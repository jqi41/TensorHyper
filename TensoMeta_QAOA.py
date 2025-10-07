#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare MetaTT-QAOA vs. Classical QAOA on MaxCut over random graphs with depolarizing noise,
now with multi-GPU support.
"""

import os
import math
import argparse
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize

# ----------------------
# 1) Tensor-Train Layer
# ----------------------
class TensorTrainLayer(nn.Module):
    def __init__(self, input_dims, output_dims, tt_ranks):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.tt_ranks = tt_ranks
        d = len(input_dims)
        assert len(output_dims) == d and len(tt_ranks) == d + 1
        self.tt_cores = nn.ParameterList()
        for k in range(d):
            r0, r1 = tt_ranks[k], tt_ranks[k + 1]
            n_k, m_k = input_dims[k], output_dims[k]
            core = nn.Parameter(torch.randn(r0, n_k, m_k, r1) * 0.1)
            self.tt_cores.append(core)
        self.bias = nn.Parameter(torch.zeros(int(np.prod(output_dims))))

    def forward(self, x):
        bsz = x.size(0)
        x_rs = x.view(bsz, *self.input_dims)
        batch = 'b'
        letters = [chr(i) for i in range(ord('a'), ord('z') + 1) if chr(i) != batch]
        d = len(self.input_dims)
        iL = letters[:d]
        oL = letters[d:2 * d]
        rL = letters[2 * d:2 * d + d + 1]
        inp = batch + ''.join(iL)
        cores_subscripts = [f"{rL[k]}{iL[k]}{oL[k]}{rL[k + 1]}" for k in range(d)]
        outp = batch + ''.join(oL)
        eins_expr = inp + ',' + ','.join(cores_subscripts) + '->' + outp
        out = torch.einsum(eins_expr, x_rs, *self.tt_cores)
        return out.reshape(bsz, -1) + self.bias

# -----------------------------
# 2) Graph → Feature Converter
# -----------------------------
def graph_to_features(graph, hist_bins=10):
    deg_list = [d for _, d in graph.degree()]
    hist, _ = np.histogram(deg_list, bins=hist_bins, range=(0, hist_bins))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return hist

def get_maxcut_edges(graph):
    return [(i, j) for i, j in graph.edges()]

# -----------------------------
# 3) Exact QAOA Expectation
# -----------------------------
def exact_qaoa_expectation(gamma, beta, edges, n_qubits, dp_noise=0.001, device=None):
    # everything stays on `device`
    device = device or gamma.device
    n = n_qubits
    dim = 1 << n
    amp0 = torch.ones(dim, dtype=torch.complex64, device=device) / math.sqrt(dim)

    bits = torch.arange(dim, device=device).unsqueeze(1)
    idx  = torch.arange(n, device=device).unsqueeze(0)
    bit_matrix  = ((bits >> idx) & 1).float()
    spin_matrix = 1.0 - 2.0 * bit_matrix

    Cz = torch.zeros(dim, dtype=torch.float32, device=device)
    for (i, j) in edges:
        si = spin_matrix[:, i]; sj = spin_matrix[:, j]
        Cz += (1.0 - si * sj) * 0.5
    Cz = Cz.to(gamma.dtype)

    phase  = torch.exp(-1j * gamma.view(1) * Cz)
    amp1   = amp0 * phase

    # build mixer
    c = torch.cos(beta * 2.0)
    s = torch.sin(beta * 2.0)
    Rx = torch.tensor([[c, -1j*s],[-1j*s, c]], device=device, dtype=torch.complex64)
    U_mix = Rx
    for _ in range(n - 1):
        U_mix = torch.kron(U_mix, Rx)

    # depolarizing noise
    U_mix_noisy = (1 - dp_noise) * U_mix + dp_noise * torch.eye(U_mix.size(0), device=device, dtype=torch.complex64)
    psi_out      = U_mix_noisy @ amp1

    probs = (psi_out.abs() ** 2).real
    exp_HC = torch.zeros((), dtype=torch.float32, device=device)
    for (i, j) in edges:
        si = spin_matrix[:, i]; sj = spin_matrix[:, j]
        zz = (si * sj).to(probs.dtype)
        exp_HC += torch.sum(probs * zz) * (-0.5) + 0.5
    return exp_HC

# -----------------------------
# 4) Classical QAOA Baseline
# -----------------------------
def classical_qaoa_maxcut(graph, p=1, shots=200):
    n = graph.number_of_nodes()
    edges = get_maxcut_edges(graph)
    def objective(params):
        gamma, beta = float(params[0]), float(params[1])
        total = 0.0
        for _ in range(shots):
            sample = np.random.randint(0, 2, size=n)
            total += sum(1 for (i,j) in edges if sample[i] != sample[j])
        return - (total / shots)

    x0  = np.random.uniform(0, np.pi, 2*p)
    res = minimize(objective, x0, method="COBYLA")
    g, b = float(res.x[0]), float(res.x[1])
    return -objective((g,b)), (g, b)

# -----------------------------
# 5) MetaTT-QAOA Model
# -----------------------------
class MetaTTQAOA(nn.Module):
    def __init__(self, input_dims, output_dims, tt_ranks):
        super().__init__()
        self.tt = TensorTrainLayer(input_dims, output_dims, tt_ranks)

    def forward(self, x):
        raw = self.tt(x)
        return raw.view(-1)  # γ, β

# -----------------------------
# 6) Training on a single graph
# -----------------------------
def train_metatt_on_graph(graph, epochs, lr, dp_noise, device, device_ids):
    edges   = get_maxcut_edges(graph)
    feat_np = graph_to_features(graph, hist_bins=10)
    feat    = torch.tensor(feat_np, dtype=torch.float32, device=device).unsqueeze(0)

    model = MetaTTQAOA(input_dims=[10], output_dims=[1,1], tt_ranks=[1,4,1]).to(device)
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        gamma, beta = model(feat)
        exp_hc = exact_qaoa_expectation(gamma, beta, edges, graph.number_of_nodes(), dp_noise, device=device)
        loss = -exp_hc
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"[MetaTT] Epoch {epoch:03d}  ⟨H_C⟩={exp_hc.item():.4f}")

    with torch.no_grad():
        gamma, beta = model(feat)
        final = exact_qaoa_expectation(gamma, beta, edges, graph.number_of_nodes(), dp_noise, device=device)
    return gamma.item(), beta.item(), final.item()

# -----------------------------
# 7) Compare across many graphs
# -----------------------------
def compare_across_graphs(n_graphs, n_nodes, p_edge, dp_noise, device, device_ids):
    np.random.seed(1234); torch.manual_seed(1234)
    classical_results = []
    metatt_results   = []
    for idx in range(1, n_graphs + 1):
        G = nx.erdos_renyi_graph(n_nodes, p_edge)
        if not nx.is_connected(G):
            # keep largest connected component
            comp = max(nx.connected_components(G), key=len)
            G = G.subgraph(comp).copy()

        cut_c, (g_c, b_c) = classical_qaoa_maxcut(G, p=1, shots=200)
        classical_results.append(cut_c)
        print(f"\nGraph {idx:02d} – Classical QAOA ≈ {cut_c:.2f}")

        g_tt, b_tt, exp_tt = train_metatt_on_graph(
            G, epochs=100, lr=0.02, dp_noise=dp_noise,
            device=device, device_ids=device_ids
        )
        metatt_results.append(exp_tt)
        print(f"         MetaTT-QAOA ⇒ γ={g_tt:.3f}, β={b_tt:.3f},  ⟨H_C⟩≈{exp_tt:.3f}")

    print("\n==== Final Averages ====")
    print(f"Avg Classical QAOA MaxCut  ≈ {np.mean(classical_results):.4f}")
    print(f"Avg MetaTT-QAOA MaxCut     ≈ {np.mean(metatt_results):.4f}")

# -----------------------------
# 8) Entrypoint & GPU Setup
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus",    type=str, default="",  help="Comma-separated list of CUDA device IDs to use (e.g. '0,1,2').")
    parser.add_argument("--n_graphs", type=int, default=10, help="Number of random graphs")
    parser.add_argument("--n_nodes",  type=int, default=12, help="Nodes per graph")
    parser.add_argument("--p_edge",   type=float, default=0.5, help="Edge probability")
    parser.add_argument("--dp_noise", type=float, default=0.001, help="Depolarizing noise")
    args = parser.parse_args()

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    n_gpus = torch.cuda.device_count()
    device_ids = list(range(n_gpus))
    device = torch.device("cuda" if n_gpus > 0 else "cpu")

    print(f"Using device: {device}  ({n_gpus} GPU(s) detected: {device_ids})")
    compare_across_graphs(
        n_graphs=args.n_graphs,
        n_nodes=args.n_nodes,
        p_edge=args.p_edge,
        dp_noise=args.dp_noise,
        device=device,
        device_ids=device_ids
    )
