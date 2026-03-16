#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorHyper-QAOA vs Classical proxy on MaxCut with multi-channel noise.
Adds Zero-Noise Extrapolation (ZNE) via gate folding and Readout Error Mitigation (REM).

Noise channels:
  - Single-qubit: depolarizing (depol), dephasing (dephase), Pauli X/Y/Z (pauli_px, pauli_py, pauli_pz)
  - Two-qubit Pauli on edges: p_twopauli ∈ {XX, YY, ZZ}
  - Mixer over-rotation: overrot_sigma (Gaussian on beta)
  - Readout error: p_readout (symmetric bit-flip model)

Mitigations:
  - ZNE: odd scales S ∈ {1,3,5} with U·U†·U folding on both cost phase and mixer; Richardson extrapolation
  - REM: analytic inverse correction on <Z_i Z_j>
"""

import math
import argparse
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize

# =========================
# 1) Tensor-Train Hypernet
# =========================
class TensorTrainLayer(nn.Module):
    def __init__(self, input_dims, output_dims, tt_ranks):
        super().__init__()
        assert len(input_dims) == len(output_dims)
        assert len(tt_ranks) == len(input_dims) + 1
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.tt_ranks = tt_ranks

        self.tt_cores = nn.ParameterList()
        for k in range(len(input_dims)):
            r0, r1 = tt_ranks[k], tt_ranks[k + 1]
            n_k, m_k = input_dims[k], output_dims[k]
            core = nn.Parameter(torch.randn(r0, n_k, m_k, r1) * 0.1)
            self.tt_cores.append(core)

        self.bias = nn.Parameter(torch.zeros(int(np.prod(output_dims))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        x_rs = x.view(bsz, *self.input_dims)

        batch = 'b'
        letters = [chr(i) for i in range(ord('a'), ord('z') + 1) if chr(i) != batch]
        d = len(self.input_dims)
        iL = letters[:d]
        oL = letters[d: 2 * d]
        rL = letters[2 * d: 2 * d + d + 1]

        inp = batch + ''.join(iL)
        cores = [f"{rL[k]}{iL[k]}{oL[k]}{rL[k+1]}" for k in range(d)]
        outp = batch + ''.join(oL)
        eins = inp + ',' + ','.join(cores) + '->' + outp

        out = torch.einsum(eins, x_rs, *self.tt_cores)
        return out.reshape(bsz, -1) + self.bias


class MetaTTQAOA(nn.Module):
    def __init__(self, input_dims, output_dims, tt_ranks):
        super().__init__()
        self.tt = TensorTrainLayer(input_dims, output_dims, tt_ranks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.tt(x)  # [B, 2]
        return raw.view(-1)  # [gamma, beta]


# ==============================
# 2) Graph → Feature conversion
# ==============================
def graph_to_features(graph: nx.Graph, hist_bins: int = 10) -> np.ndarray:
    deg_list = [d for _, d in graph.degree()]
    hist, _ = np.histogram(deg_list, bins=hist_bins, range=(0, hist_bins))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return hist  # length = hist_bins


def get_maxcut_edges(graph: nx.Graph):
    return [(i, j) for i, j in graph.edges()]


# =========================================
# 3) Gradient-safe statevector primitives
# =========================================
def _pair_indices(q: int, dim: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.arange(dim, device=device)
    mask0 = ((idx >> q) & 1) == 0
    idx0 = idx[mask0]
    idx1 = idx0 | (1 << q)
    return idx0, idx1

def apply_rx_layer(state: torch.Tensor, beta: torch.Tensor, n: int) -> torch.Tensor:
    dim = state.numel(); device = state.device
    c = torch.cos(2.0 * beta); s = torch.sin(2.0 * beta)
    for q in range(n):
        idx0, idx1 = _pair_indices(q, dim, device)
        a = state.index_select(0, idx0)
        b = state.index_select(0, idx1)
        a_new = c * a + (-1j * s) * b
        b_new = (-1j * s) * a + c * b
        state = state.clone()
        state = state.scatter(0, idx0, a_new)
        state = state.scatter(0, idx1, b_new)
    return state

def apply_X(state: torch.Tensor, q: int, n: int) -> torch.Tensor:
    dim = state.numel(); device = state.device
    idx0, idx1 = _pair_indices(q, dim, device)
    a = state.index_select(0, idx0); b = state.index_select(0, idx1)
    new_state = state.clone()
    new_state = new_state.scatter(0, idx0, b)
    new_state = new_state.scatter(0, idx1, a)
    return new_state

def apply_Z(state: torch.Tensor, q: int, n: int) -> torch.Tensor:
    dim = state.numel(); device = state.device
    idx = torch.arange(dim, device=device)
    mask1 = ((idx >> q) & 1) == 1
    new_state = state.clone()
    new_state = new_state.scatter(0, idx[mask1], -state.index_select(0, idx[mask1]))
    return new_state

def apply_Y(state: torch.Tensor, q: int, n: int) -> torch.Tensor:
    dim = state.numel(); device = state.device
    idx0, idx1 = _pair_indices(q, dim, device)
    a = state.index_select(0, idx0)  # |...0_q...>
    b = state.index_select(0, idx1)  # |...1_q...>
    a_new = 1j * b
    b_new = -1j * a
    new_state = state.clone()
    new_state = new_state.scatter(0, idx0, a_new)
    new_state = new_state.scatter(0, idx1, b_new)
    return new_state

def apply_two_qubit_pauli(state: torch.Tensor, i: int, j: int, which: str, n: int) -> torch.Tensor:
    if which == "XX":
        state = apply_X(state, i, n); state = apply_X(state, j, n)
    elif which == "YY":
        state = apply_Y(state, i, n); state = apply_Y(state, j, n)
    else:  # "ZZ"
        state = apply_Z(state, i, n); state = apply_Z(state, j, n)
    return state


# ===========================================
# 4) ZNE folding helpers and extrapolation
# ===========================================
def folding_pattern(scale: int) -> list[int]:
    """
    For odd integer scale s (1,3,5,...) return sequence of +1/-1 multipliers
    to realize U * U† * U * ... such that ideal unitary == U while noise scales ~ s.
    Example:
      s=1 -> [ +1 ]
      s=3 -> [ +1, -1, +1 ]
      s=5 -> [ +1, -1, +1, -1, +1 ]
    """
    assert scale % 2 == 1 and scale >= 1
    patt = []
    sign = +1
    for _ in range(scale):
        patt.append(sign)
        sign *= -1
    # If even count of negatives, ends on -1; but pattern for odd s ends on +1 as above.
    return patt

def richardson_extrapolate(scales, values, order: str = "linear"):
    """
    scales: list of odd ints (e.g., [1,3] or [1,3,5])
    values: list of E(s) at those scales
    returns E(s=0) estimate (zero-noise limit)
    """
    s = torch.tensor(scales, dtype=torch.float64)
    v = torch.stack(values).to(dtype=torch.float64)
    if order == "linear" or len(scales) == 2:
        # Fit E(s) ≈ a + b*s and return a
        A = torch.stack([torch.ones_like(s), s], dim=1)  # [n,2]
        sol, _ = torch.lstsq(v.unsqueeze(1), A)  # deprecated, but fine; alternatively use torch.linalg.lstsq
        a = sol[0, 0]
        return a.to(values[0].dtype)
    else:
        # Quadratic fit: E(s) ≈ a + b*s + c*s^2; a is intercept
        A = torch.stack([torch.ones_like(s), s, s**2], dim=1)  # [n,3]
        sol = torch.linalg.lstsq(A, v).solution  # [3]
        a = sol[0]
        return a.to(values[0].dtype)


# ===========================================================
# 5) Expectation core with ZNE folding and REM toggles
# ===========================================================
def exact_qaoa_expectation_one_scale(
    gamma: torch.Tensor,
    beta: torch.Tensor,
    edges,
    n_qubits: int,
    noise_cfg: dict,
    mc_shots: int,
    scale: int,
    rem: bool,
    Cz_precomp: torch.Tensor = None,
    spin_matrix: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute ⟨H_C⟩ at a given noise 'scale' via gate folding (odd integer).
    REM toggles analytic inverse correction of readout error.
    """
    assert scale % 2 == 1 and scale >= 1
    device = gamma.device
    n = n_qubits
    dim = 1 << n

    depol = float(noise_cfg.get('depol', 0.0))
    deph  = float(noise_cfg.get('dephase', 0.0))
    px    = float(noise_cfg.get('pauli_px', 0.0))
    py    = float(noise_cfg.get('pauli_py', 0.0))
    pz    = float(noise_cfg.get('pauli_pz', 0.0))
    sig   = float(noise_cfg.get('overrot_sigma', 0.0))
    p2    = float(noise_cfg.get('p_twopauli', 0.0))
    rerr  = float(noise_cfg.get('p_readout', 0.0))

    # Precompute spin_matrix, Cz if not given
    if spin_matrix is None:
        idx = torch.arange(dim, device=device).unsqueeze(1)
        qidx = torch.arange(n, device=device).unsqueeze(0)
        bit_matrix = ((idx >> qidx) & 1).float()
        spin_matrix = 1.0 - 2.0 * bit_matrix
    if Cz_precomp is None:
        Cz = torch.zeros(dim, dtype=torch.float32, device=device)
        for (i, j) in edges:
            Cz += 0.5 * (1.0 - spin_matrix[:, i] * spin_matrix[:, j])
        Cz_precomp = Cz.to(gamma.dtype)

    patt = folding_pattern(scale)
    acc = torch.zeros((), dtype=torch.float32, device=device)

    for _ in range(mc_shots):
        # Start in |+...+>
        state = torch.ones(dim, dtype=torch.complex64, device=device) / math.sqrt(dim)

        # Folded cost+mixer sequence; U_cost(±γ) and U_mix(±β_eff)
        for sgn in patt:
            # cost phase
            state = state * torch.exp(-1j * (sgn * gamma).view(1) * Cz_precomp)
            # mixer with over-rotation applied to |beta|
            beta_eff = (sgn * beta) + (torch.randn((), device=device) * sig if sig > 0 else 0.0)
            state = apply_rx_layer(state, beta_eff, n)

            # single-qubit noise after each sublayer
            for q in range(n):
                if torch.rand((), device=device) < px:   state = apply_X(state, q, n)
                if torch.rand((), device=device) < py:   state = apply_Y(state, q, n)
                if torch.rand((), device=device) < pz:   state = apply_Z(state, q, n)
                if torch.rand((), device=device) < depol:
                    k = torch.randint(0, 3, (), device=device)
                    state = apply_X(state, q, n) if k == 0 else (apply_Y(state, q, n) if k == 1 else apply_Z(state, q, n))
                if torch.rand((), device=device) < deph:
                    state = apply_Z(state, q, n)

            # two-qubit noise after sublayer
            if p2 > 0.0:
                for (i, j) in edges:
                    if torch.rand((), device=device) < p2:
                        which = ["XX", "YY", "ZZ"][torch.randint(0, 3, (), device=device).item()]
                        state = apply_two_qubit_pauli(state, i, j, which, n)

        probs = (state.abs() ** 2).real

        # Readout error scaling (apply or mitigate)
        base_scale = (1.0 - 2.0 * rerr) ** 2
        use_scale = 1.0 if rem else base_scale
        inv_scale = (1.0 / max(base_scale, 1e-8)) if rem else 1.0

        exp_HC = torch.zeros((), dtype=torch.float32, device=device)
        for (i, j) in edges:
            zz = (spin_matrix[:, i] * spin_matrix[:, j]).to(probs.dtype)
            true_corr = torch.sum(probs * zz)  # correlation without readout error
            corr_used = torch.clamp(true_corr * use_scale * inv_scale, -1.0, 1.0)  # if rem=True, this is ~true_corr
            exp_HC += 0.5 * (1.0 - corr_used)

        acc = acc + exp_HC

    return acc / mc_shots


def exact_qaoa_expectation(
    gamma: torch.Tensor,
    beta: torch.Tensor,
    edges,
    n_qubits: int,
    noise_cfg=None,
    mc_shots: int = 8,
    use_zne: bool = False,
    zne_scales=(1, 3, 5),
    zne_order: str = "linear",
    use_rem: bool = False,
) -> torch.Tensor:
    """
    Wrapper: compute ⟨H_C⟩ with optional ZNE (folding+extrapolation) and REM.
    Returns a scalar tensor (requires grad if gamma/beta require grad).
    """
    if noise_cfg is None:
        noise_cfg = dict(
            depol=0.0, dephase=0.0,
            pauli_px=0.0, pauli_py=0.0, pauli_pz=0.0,
            overrot_sigma=0.0, p_twopauli=0.0, p_readout=0.0
        )

    device = gamma.device
    n = n_qubits
    dim = 1 << n
    # Precompute spin_matrix and Cz once for reuse across scales
    idx = torch.arange(dim, device=device).unsqueeze(1)
    qidx = torch.arange(n, device=device).unsqueeze(0)
    bit_matrix = ((idx >> qidx) & 1).float()
    spin_matrix = 1.0 - 2.0 * bit_matrix
    Cz = torch.zeros(dim, dtype=torch.float32, device=device)
    for (i, j) in edges:
        Cz += 0.5 * (1.0 - spin_matrix[:, i] * spin_matrix[:, j])
    Cz = Cz.to(gamma.dtype)

    if not use_zne:
        return exact_qaoa_expectation_one_scale(
            gamma, beta, edges, n_qubits, noise_cfg, mc_shots, scale=1, rem=use_rem,
            Cz_precomp=Cz, spin_matrix=spin_matrix
        )

    # ZNE: evaluate at multiple scales and extrapolate
    scales = list(zne_scales)
    vals = []
    for s in scales:
        v = exact_qaoa_expectation_one_scale(
            gamma, beta, edges, n_qubits, noise_cfg, mc_shots, scale=s, rem=use_rem,
            Cz_precomp=Cz, spin_matrix=spin_matrix
        )
        vals.append(v)
    # Richardson extrapolation to zero-noise (s→0)
    return richardson_extrapolate(scales, vals, order=zne_order)


# ==============================================
# 6) Classical "proxy" optimizer for comparison
# ==============================================
def classical_qaoa_maxcut(graph: nx.Graph, p: int = 1, shots: int = 200):
    n = graph.number_of_nodes()
    edges = get_maxcut_edges(graph)

    def objective(params):
        gamma, beta = float(params[0]), float(params[1])
        total = 0.0
        for _ in range(shots):
            sample = np.random.randint(0, 2, size=n)
            cut_val = sum(1 for (i, j) in edges if sample[i] != sample[j])
            total += cut_val
        return - (total / shots)

    x0 = np.random.uniform(0, np.pi, 2 * p)
    res = minimize(objective, x0, method="COBYLA")
    best_gamma, best_beta = float(res.x[0]), float(res.x[1])
    best_cut = -objective((best_gamma, best_beta))
    return best_cut, (best_gamma, best_beta)


# ==================================
# 7) Training / experiment routines
# ==================================
DEFAULT_NOISE = {
    'depol': 1e-3, 'dephase': 1e-3,
    'pauli_px': 0.0, 'pauli_py': 0.0, 'pauli_pz': 0.0,
    'overrot_sigma': 0.0,
    'p_twopauli': 1e-3,
    'p_readout': 1e-2
}

def train_metatt_on_graph(
    graph: nx.Graph,
    epochs: int = 100,
    lr: float = 0.02,
    hist_bins: int = 10,
    noise_cfg: dict | None = None,
    mc_shots: int = 8,
    tt_input_dims=(2, 5),
    tt_output_dims=(1, 2),
    tt_ranks=(1, 4, 1),
    use_zne: bool = False,
    zne_scales=(1, 3, 5),
    zne_order: str = "linear",
    use_rem: bool = False,
    verbose: bool = True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = graph.number_of_nodes()
    edges = get_maxcut_edges(graph)

    feat_np = graph_to_features(graph, hist_bins=hist_bins)
    prod_in = int(np.prod(tt_input_dims))
    if len(feat_np) < prod_in:
        pad = np.zeros(prod_in, dtype=np.float32); pad[:len(feat_np)] = feat_np; feat_np = pad
    elif len(feat_np) > prod_in:
        feat_np = feat_np[:prod_in]
    feat = torch.tensor(feat_np, dtype=torch.float32, device=device).unsqueeze(0)

    model = MetaTTQAOA(
        input_dims=list(tt_input_dims),
        output_dims=list(tt_output_dims),
        tt_ranks=list(tt_ranks)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    cfg = DEFAULT_NOISE if noise_cfg is None else noise_cfg

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        gamma, beta = model(feat)  # tensors require grad
        exp_hc = exact_qaoa_expectation(
            gamma, beta, edges, n, cfg, mc_shots,
            use_zne=use_zne, zne_scales=zne_scales, zne_order=zne_order, use_rem=use_rem
        )
        loss = -exp_hc
        loss.backward()
        optimizer.step()

        if verbose and (epoch % max(1, (epochs // 5)) == 0):
            print(f"  [TensorHyper] Epoch {epoch:03d}  ⟨H_C⟩={exp_hc.item():.4f}")

    with torch.no_grad():
        gamma, beta = model(feat)
        final_exp = exact_qaoa_expectation(
            gamma, beta, edges, n, cfg, mc_shots,
            use_zne=use_zne, zne_scales=zne_scales, zne_order=zne_order, use_rem=use_rem
        ).item()
    return float(gamma.item()), float(beta.item()), final_exp


def compare_across_graphs(
    n_graphs: int = 5,
    n_nodes: int = 12,
    p_edge: float = 0.5,
    noise_cfg: dict | None = None,
    mc_shots: int = 8,
    epochs: int = 100,
    lr: float = 0.02,
    use_zne: bool = False,
    zne_scales=(1, 3, 5),
    zne_order: str = "linear",
    use_rem: bool = False,
    seed: int = 1234
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    classical_results, metatt_results = [], []
    cfg = DEFAULT_NOISE if noise_cfg is None else noise_cfg

    for idx in range(1, n_graphs + 1):
        G = nx.erdos_renyi_graph(n=n_nodes, p=p_edge, seed=seed + idx)
        if not nx.is_connected(G):
            comp = max(nx.connected_components(G), key=len)
            G = G.subgraph(comp).copy()

        cut_classical, (g_cl, b_cl) = classical_qaoa_maxcut(G, p=1, shots=200)
        classical_results.append(cut_classical)
        print(f"\nGraph {idx:02d}  – Classical proxy MaxCut ≈ {cut_classical:>5.2f}")

        gamma_tt, beta_tt, exp_tt = train_metatt_on_graph(
            G, epochs=epochs, lr=lr, noise_cfg=cfg, mc_shots=mc_shots,
            use_zne=use_zne, zne_scales=zne_scales, zne_order=zne_order, use_rem=use_rem,
            verbose=True
        )
        metatt_results.append(exp_tt)
        print(f"          MetaTT-QAOA ⇒ γ={gamma_tt:.3f}, β={beta_tt:.3f},  ⟨H_C⟩≈{exp_tt:.3f}")

    print("\n==== Final Averages over all graphs ====")
    print(f"  Avg Classical QAOA (proxy)  ≈ {float(np.mean(classical_results)):.4f}")
    print(f"  Avg TensorHyper-QAOA ⟨H_C⟩       ≈ {float(np.mean(metatt_results)):.4f}")


# ==========
# 8)  CLI
# ==========
def build_argparser():
    p = argparse.ArgumentParser(description="MetaTT-QAOA vs Classical proxy with multi-channel noise + ZNE/REM")
    p.add_argument("--n_graphs", type=int, default=10)
    p.add_argument("--n_nodes", type=int, default=14)
    p.add_argument("--p_edge", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--mc_shots", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234)
    # Noise
    p.add_argument("--depol", type=float, default=1e-3)
    p.add_argument("--dephase", type=float, default=1e-3)
    p.add_argument("--pauli_px", type=float, default=0.001)
    p.add_argument("--pauli_py", type=float, default=0.001)
    p.add_argument("--pauli_pz", type=float, default=0.001)
    p.add_argument("--overrot_sigma", type=float, default=0.005)
    p.add_argument("--p_twopauli", type=float, default=2e-3)
    p.add_argument("--p_readout", type=float, default=1e-2)
    # Mitigations
    p.add_argument("--use_zne", action="store_true", help="Enable ZNE (gate folding + extrapolation)")
    p.add_argument("--zne_scales", type=int, nargs="+", default=[1, 3, 5], help="Odd scales for folding")
    p.add_argument("--zne_order", type=str, default="linear", choices=["linear", "quadratic"], help="ZNE extrapolation order")
    p.add_argument("--use_rem", action="store_true", help="Enable analytic readout error mitigation")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()

    noise_cfg = {
        'depol': args.depol,
        'dephase': args.dephase,
        'pauli_px': args.pauli_px,
        'pauli_py': args.pauli_py,
        'pauli_pz': args.pauli_pz,
        'overrot_sigma': args.overrot_sigma,
        'p_twopauli': args.p_twopauli,
        'p_readout': args.p_readout
    }

    compare_across_graphs(
        n_graphs=args.n_graphs,
        n_nodes=args.n_nodes,
        p_edge=args.p_edge,
        noise_cfg=noise_cfg,
        mc_shots=args.mc_shots,
        epochs=args.epochs,
        lr=args.lr,
        use_zne=args.use_zne,
        zne_scales=tuple(args.zne_scales),
        zne_order=args.zne_order,
        use_rem=args.use_rem,
        seed=args.seed
    )
