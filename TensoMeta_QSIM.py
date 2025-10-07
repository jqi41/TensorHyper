#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LiH VQE + MetaTT-VQC (FINAL corrected)

- LiH 4-qubit Hamiltonian → 16×16 via frozen orbitals
- Classical VQE baseline (COBYLA)
- MetaTT-VQC: TT network generates 24 parameters
"""

import cirq
import numpy as np
import scipy.sparse.linalg as spla
from scipy.optimize import minimize
import torch
import torch.nn as nn

from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from openfermion.transforms import jordan_wigner, freeze_orbitals, get_fermion_operator
from openfermion.linalg import get_sparse_operator

# 1. LiH reduced Hamiltonian (4 qubits)
def get_lih_reduced_hamiltonian(bond_length=1.6):
    geometry = [['Li', (0, 0, 0)], ['H', (0, 0, bond_length)]]
    molecule = MolecularData(
        geometry=geometry,
        basis='sto-3g',
        multiplicity=1,
        charge=0
    )
    molecule = run_pyscf(molecule, run_scf=True, run_fci=True)

    # Convert to FermionOperator
    interaction_op = molecule.get_molecular_hamiltonian()
    fermion_op = get_fermion_operator(interaction_op)

    # Freeze core spin-orbitals [0,1] and virtual spin-orbitals [6..11]
    frozen_fermion = freeze_orbitals(
        fermion_op,
        [0, 1],              # freeze core spin-orbitals
        list(range(6, 12))   # drop high-lying virtual spin-orbitals
    )

    # Map to qubits and build sparse matrix
    qubit_ham = jordan_wigner(frozen_fermion)
    H_sparse = get_sparse_operator(qubit_ham)

    # Sanity check: should be 4 qubits → 16×16
    dim = H_sparse.shape[0]
    n_qubits = int(np.log2(dim))
    print(f"[INFO] Reduced LiH Hamiltonian → {n_qubits} qubits, shape {H_sparse.shape}")
    assert H_sparse.shape == (16, 16), f"Expected (16,16), got {H_sparse.shape}"
    return H_sparse

# 2. Load Hamiltonian & exact energy
H_sparse = get_lih_reduced_hamiltonian()
E_exact = spla.eigsh(H_sparse, k=1, which='SA')[0][0].real
print(f"Exact LiH ground energy: {E_exact:.6f} Ha")

# 3. Ansatz & energy
NUM_QUBITS = 4
NUM_LAYERS = 2
PARAM_COUNT = NUM_QUBITS * NUM_LAYERS * 3  # 24 parameters

qubits = cirq.LineQubit.range(NUM_QUBITS)
simulator = cirq.Simulator()

def ansatz(params):
    assert len(params) == PARAM_COUNT, f"Expected {PARAM_COUNT} params, got {len(params)}"
    circuit = cirq.Circuit()
    idx = 0
    for _ in range(NUM_LAYERS):
        for q in qubits:
            circuit.append(cirq.rx(params[idx])(q)); idx += 1
            circuit.append(cirq.ry(params[idx])(q)); idx += 1
            circuit.append(cirq.rz(params[idx])(q)); idx += 1
        for i in range(NUM_QUBITS):
            circuit.append(cirq.CNOT(qubits[i], qubits[(i+1) % NUM_QUBITS]))
    return circuit

def energy(params):
    circuit = ansatz(params)
    result = simulator.simulate(circuit)
    state = result.final_state_vector
    assert state.shape[0] == 16, f"Statevector length {state.shape[0]}, expected 16"
    psi = state.reshape(-1, 1)
    return np.vdot(psi, H_sparse @ psi).real

# 4. Classical VQE baseline
def classical_vqe():
    x0 = np.zeros(PARAM_COUNT)
    res = minimize(energy, x0, method='COBYLA')
    return res.x, res.fun

# 5. MetaTT-VQC: Tensor-Train network
class TTNetwork(nn.Module):
    def __init__(self, dims, ranks):
        super().__init__()
        self.cores = nn.ParameterList([
            nn.Parameter(torch.randn(r1, d, r2) * 0.1)
            for r1, d, r2 in zip(ranks[:-1], dims, ranks[1:])
        ])

    def forward(self):
        res = self.cores[0][0]
        for core in self.cores[1:]:
            res_flat = res.reshape(-1, res.shape[-1])
            temp = torch.einsum('xr, rds -> xds', res_flat, core)
            res = temp.reshape(-1, core.shape[2])
        params = res.squeeze(-1).detach().numpy()
        assert len(params) == PARAM_COUNT, f"TT output {len(params)} params, expected {PARAM_COUNT}"
        return params

dims = [4, 6]   # 4*6 = 24 parameters
ranks = [1, 2, 1]
tt_net = TTNetwork(dims, ranks)

def metatt_objective(flat_params):
    offset = 0
    for core in tt_net.cores:
        r1, d, r2 = core.shape
        size = r1 * d * r2
        vals = flat_params[offset:offset+size].reshape(r1, d, r2)
        core.data.copy_(torch.from_numpy(vals.astype(np.float32)))
        offset += size
    return energy(tt_net())

def metatt_vqe():
    total = sum(core.numel() for core in tt_net.cores)
    x0 = np.random.randn(total) * 0.1
    res = minimize(metatt_objective, x0, method='COBYLA')
    metatt_objective(res.x)  # finalize TT
    return tt_net(), res.fun

# 6. Run & compare
if __name__ == "__main__":
    _, E_cl = classical_vqe()
    print(f"Classical VQE energy: {E_cl:.6f} Ha")

    _, E_tt = metatt_vqe()
    print(f"MetaTT-VQC energy   : {E_tt:.6f} Ha")
