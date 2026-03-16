#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Improved TensorHyper-VQC for LiH
=============================================

Key improvements over the previous version:
1. Stronger ansatz:
   - Hartree-Fock-style initial state
   - deeper hardware-efficient circuit
   - bidirectional ring entanglement
2. Stronger optimization:
   - multi-start classical VQE
   - warm-started residual TT optimization
3. Better TensorHyper design:
   - TT network generates a bounded correction to a good classical parameter vector
   - optional final local polishing

This version focuses on improving variational performance first.
After the variational gap is reduced, the noisy evaluation/mitigation layer can be
re-attached on top of this stronger backbone.

Dependencies:
  pip install cirq numpy scipy torch openfermion openfermionpyscf pyscf
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import List, Tuple

import cirq
import numpy as np
import scipy.sparse.linalg as spla
from scipy.optimize import minimize
import torch
import torch.nn as nn

from openfermion import MolecularData
from openfermion.transforms import jordan_wigner, freeze_orbitals, get_fermion_operator
from openfermion.linalg import get_sparse_operator
from openfermionpyscf import run_pyscf


# ============================================================
# 1) LiH reduced Hamiltonian
# ============================================================
def get_lih_reduced_hamiltonian(bond_length: float = 1.6):
    geometry = [['Li', (0, 0, 0)], ['H', (0, 0, bond_length)]]
    molecule = MolecularData(
        geometry=geometry,
        basis='sto-3g',
        multiplicity=1,
        charge=0,
    )
    molecule = run_pyscf(molecule, run_scf=True, run_fci=True)

    interaction_op = molecule.get_molecular_hamiltonian()
    fermion_op = get_fermion_operator(interaction_op)

    frozen_fermion = freeze_orbitals(
        fermion_op,
        occupied=[0, 1],
        unoccupied=list(range(6, 12)),
    )

    qubit_ham = jordan_wigner(frozen_fermion)
    H_sparse = get_sparse_operator(qubit_ham)

    dim = H_sparse.shape[0]
    n_qubits = int(np.log2(dim))
    print(f"[INFO] Reduced LiH Hamiltonian -> {n_qubits} qubits, shape {H_sparse.shape}")
    assert H_sparse.shape == (16, 16), f"Expected (16,16), got {H_sparse.shape}"

    return H_sparse


# ============================================================
# 2) Ansatz configuration
# ============================================================
NUM_QUBITS = 4
qubits = cirq.LineQubit.range(NUM_QUBITS)
simulator = cirq.Simulator()


@dataclass
class AnsatzConfig:
    n_layers: int = 4
    hf_bitstring: str = "1100"   # user can try "0011" if qubit ordering differs
    use_reverse_ring: bool = True


def param_count(cfg: AnsatzConfig) -> int:
    # per layer: RX, RY, RZ on each qubit => 3 * 4 = 12
    return NUM_QUBITS * cfg.n_layers * 3


def prepare_hf_state(circuit: cirq.Circuit, hf_bitstring: str):
    assert len(hf_bitstring) == NUM_QUBITS
    for i, bit in enumerate(hf_bitstring):
        if bit == "1":
            circuit.append(cirq.X(qubits[i]))


def build_ansatz(params: np.ndarray, cfg: AnsatzConfig) -> cirq.Circuit:
    expected = param_count(cfg)
    assert len(params) == expected, f"Expected {expected} params, got {len(params)}"

    circuit = cirq.Circuit()
    prepare_hf_state(circuit, cfg.hf_bitstring)

    idx = 0
    for _ in range(cfg.n_layers):
        # single-qubit rotations
        for q in qubits:
            circuit.append(cirq.rx(params[idx])(q))
            idx += 1
            circuit.append(cirq.ry(params[idx])(q))
            idx += 1
            circuit.append(cirq.rz(params[idx])(q))
            idx += 1

        # forward ring entanglement
        for i in range(NUM_QUBITS):
            circuit.append(cirq.CNOT(qubits[i], qubits[(i + 1) % NUM_QUBITS]))

        # reverse ring entanglement
        if cfg.use_reverse_ring:
            for i in reversed(range(NUM_QUBITS)):
                circuit.append(cirq.CNOT(qubits[(i + 1) % NUM_QUBITS], qubits[i]))

    return circuit


def energy_from_params(params: np.ndarray, H_sparse, cfg: AnsatzConfig) -> float:
    circuit = build_ansatz(params, cfg)
    result = simulator.simulate(circuit)
    state = result.final_state_vector
    psi = state.reshape(-1, 1)
    return float(np.vdot(psi, H_sparse @ psi).real)


# ============================================================
# 3) Classical VQE with multi-start
# ============================================================
def generate_initial_points(
    dim: int,
    n_restarts: int,
    seed: int = 1234,
    include_zero: bool = True,
    scale: float = 0.2,
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    xs = []
    if include_zero:
        xs.append(np.zeros(dim, dtype=np.float64))

    # small random points
    for _ in range(max(0, n_restarts - len(xs))):
        xs.append(rng.normal(loc=0.0, scale=scale, size=dim))

    return xs


def classical_vqe_multistart(
    H_sparse,
    cfg: AnsatzConfig,
    n_restarts: int = 12,
    maxiter: int = 300,
    seed: int = 1234,
    method: str = "COBYLA",
):
    dim = param_count(cfg)

    def objective(x):
        return energy_from_params(x, H_sparse, cfg)

    best_x = None
    best_e = float("inf")

    starts = generate_initial_points(dim, n_restarts=n_restarts, seed=seed, include_zero=True)

    for k, x0 in enumerate(starts, start=1):
        res = minimize(
            objective,
            x0,
            method=method,
            options={"maxiter": maxiter, "disp": False},
        )
        if res.fun < best_e:
            best_e = float(res.fun)
            best_x = res.x.copy()
        print(f"[Classical restart {k:02d}/{len(starts)}] energy = {res.fun:.8f} Ha")

    assert best_x is not None
    return best_x, best_e


# ============================================================
# 4) Residual TensorHyper-VQC (TT correction generator)
# ============================================================
class ResidualTTNetwork(nn.Module):
    """
    TT network generates a correction vector delta_theta of size PARAM_COUNT.
    Final parameters:
        theta = base_theta + delta_scale * tanh(delta_theta)

    This is much easier to optimize than generating theta from scratch.
    """

    def __init__(self, dims: List[int], ranks: List[int], delta_scale: float = 0.35):
        super().__init__()
        assert len(dims) + 1 == len(ranks)
        self.dims = dims
        self.ranks = ranks
        self.delta_scale = delta_scale

        self.cores = nn.ParameterList([
            nn.Parameter(torch.randn(r1, d, r2) * 0.05)
            for r1, d, r2 in zip(ranks[:-1], dims, ranks[1:])
        ])

    def forward(self) -> torch.Tensor:
        res = self.cores[0][0]  # [d1, r]
        for core in self.cores[1:]:
            temp = torch.einsum("xr,rds->xds", res, core)
            res = temp.reshape(-1, core.shape[2])
        out = res.squeeze(-1)
        return self.delta_scale * torch.tanh(out)


def choose_tt_shape(total_dim: int):
    """
    Factor total_dim into a compact TT output shape.
    For 4 layers -> 48 params, a good default is [4, 4, 3].
    For other values, use a simple fallback.
    """
    if total_dim == 48:
        return [4, 4, 3], [1, 4, 4, 1]
    if total_dim == 36:
        return [3, 4, 3], [1, 4, 4, 1]
    if total_dim == 24:
        return [4, 6], [1, 4, 1]

    # fallback: try near-square factorization
    for a in range(2, total_dim + 1):
        if total_dim % a == 0:
            b = total_dim // a
            return [a, b], [1, 4, 1]
    return [total_dim], [1, 1]


def set_tt_from_flat(tt_net: ResidualTTNetwork, flat_params: np.ndarray):
    offset = 0
    for core in tt_net.cores:
        r1, d, r2 = core.shape
        size = r1 * d * r2
        vals = flat_params[offset: offset + size].reshape(r1, d, r2)
        core.data.copy_(torch.from_numpy(vals.astype(np.float32)))
        offset += size


def flatten_tt(tt_net: ResidualTTNetwork) -> np.ndarray:
    arrs = []
    for core in tt_net.cores:
        arrs.append(core.detach().cpu().numpy().reshape(-1))
    return np.concatenate(arrs, axis=0)


def params_from_tt(tt_net: ResidualTTNetwork, base_theta: np.ndarray) -> np.ndarray:
    delta = tt_net().detach().cpu().numpy()
    assert delta.shape[0] == base_theta.shape[0]
    return base_theta + delta


def tensorhyper_vqe_residual(
    H_sparse,
    cfg: AnsatzConfig,
    base_theta: np.ndarray,
    tt_dims: List[int],
    tt_ranks: List[int],
    delta_scale: float = 0.35,
    maxiter: int = 300,
    n_restarts: int = 8,
    seed: int = 1234,
    method: str = "COBYLA",
):
    dim = param_count(cfg)
    assert base_theta.shape[0] == dim
    assert int(np.prod(tt_dims)) == dim, f"TT output dims {tt_dims} do not match parameter count {dim}"

    best_theta = None
    best_energy = float("inf")
    best_tt = None

    rng = np.random.default_rng(seed)

    for restart in range(1, n_restarts + 1):
        tt_net = ResidualTTNetwork(tt_dims, tt_ranks, delta_scale=delta_scale)
        if restart == 1:
            x0 = flatten_tt(tt_net)
        else:
            x0 = rng.normal(0.0, 0.05, size=sum(c.numel() for c in tt_net.cores))

        def objective(flat_params):
            set_tt_from_flat(tt_net, flat_params)
            theta = params_from_tt(tt_net, base_theta)
            return energy_from_params(theta, H_sparse, cfg)

        res = minimize(
            objective,
            x0,
            method=method,
            options={"maxiter": maxiter, "disp": False},
        )

        set_tt_from_flat(tt_net, res.x)
        theta = params_from_tt(tt_net, base_theta)
        e = energy_from_params(theta, H_sparse, cfg)

        if e < best_energy:
            best_energy = float(e)
            best_theta = theta.copy()
            best_tt = tt_net

        print(f"[TensorHyper restart {restart:02d}/{n_restarts}] energy = {e:.8f} Ha")

    assert best_theta is not None and best_tt is not None
    return best_theta, best_energy, best_tt


# ============================================================
# 5) Optional post-polish
# ============================================================
def local_polish(
    theta0: np.ndarray,
    H_sparse,
    cfg: AnsatzConfig,
    maxiter: int = 120,
    method: str = "COBYLA",
):
    def objective(x):
        return energy_from_params(x, H_sparse, cfg)

    res = minimize(
        objective,
        theta0,
        method=method,
        options={"maxiter": maxiter, "disp": False},
    )
    return res.x.copy(), float(res.fun)


# ============================================================
# 6) Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Improved TensorHyper-VQC for LiH")

    parser.add_argument("--bond_length", type=float, default=1.6)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--hf_bitstring", type=str, default="1100")
    parser.add_argument("--no_reverse_ring", action="store_true")

    parser.add_argument("--classical_restarts", type=int, default=12)
    parser.add_argument("--classical_maxiter", type=int, default=300)

    parser.add_argument("--tt_restarts", type=int, default=8)
    parser.add_argument("--tt_maxiter", type=int, default=300)
    parser.add_argument("--delta_scale", type=float, default=0.35)

    parser.add_argument("--post_polish", action="store_true")
    parser.add_argument("--polish_maxiter", type=int, default=120)

    parser.add_argument("--seed", type=int, default=1234)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = AnsatzConfig(
        n_layers=args.n_layers,
        hf_bitstring=args.hf_bitstring,
        use_reverse_ring=not args.no_reverse_ring,
    )

    H_sparse = get_lih_reduced_hamiltonian(bond_length=args.bond_length)
    E_exact = spla.eigsh(H_sparse, k=1, which='SA')[0][0].real

    print(f"Exact LiH ground energy: {E_exact:.8f} Ha")
    print(f"[INFO] Ansatz layers: {cfg.n_layers}")
    print(f"[INFO] HF bitstring: {cfg.hf_bitstring}")
    print(f"[INFO] Parameter count: {param_count(cfg)}")

    # Step 1: strong classical warm start
    classical_theta, classical_e = classical_vqe_multistart(
        H_sparse=H_sparse,
        cfg=cfg,
        n_restarts=args.classical_restarts,
        maxiter=args.classical_maxiter,
        seed=args.seed,
        method="COBYLA",
    )

    print(f"\nBest Classical VQE energy: {classical_e:.8f} Ha")
    print(f"Classical error: {abs(classical_e - E_exact):.8f} Ha")

    # Step 2: residual TensorHyper-VQC
    total_dim = param_count(cfg)
    tt_dims, tt_ranks = choose_tt_shape(total_dim)
    print(f"[INFO] TT dims: {tt_dims}, TT ranks: {tt_ranks}")

    tt_theta, tt_e, _ = tensorhyper_vqe_residual(
        H_sparse=H_sparse,
        cfg=cfg,
        base_theta=classical_theta,
        tt_dims=tt_dims,
        tt_ranks=tt_ranks,
        delta_scale=args.delta_scale,
        maxiter=args.tt_maxiter,
        n_restarts=args.tt_restarts,
        seed=args.seed,
        method="COBYLA",
    )

    print(f"\nTensorHyper-VQC energy: {tt_e:.8f} Ha")
    print(f"TensorHyper-VQC error: {abs(tt_e - E_exact):.8f} Ha")

    if args.post_polish:
        polished_theta, polished_e = local_polish(
            theta0=tt_theta,
            H_sparse=H_sparse,
            cfg=cfg,
            maxiter=args.polish_maxiter,
            method="COBYLA",
        )
        print(f"\nPost-polished energy: {polished_e:.8f} Ha")
        print(f"Post-polished error: {abs(polished_e - E_exact):.8f} Ha")
    else:
        polished_e = None

    print("\nSummary")
    print("-------")
    print(f"Exact energy            : {E_exact:.8f} Ha")
    print(f"Best classical VQE      : {classical_e:.8f} Ha")
    print(f"TensorHyper-VQC         : {tt_e:.8f} Ha")
    if polished_e is not None:
        print(f"Post-polished           : {polished_e:.8f} Ha")


if __name__ == "__main__":
    main()
