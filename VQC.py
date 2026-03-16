#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 09:20:50 2023

@author: junqi
"""

# PyTorch
import torch
from torch import nn

# Torch Quantum
import torchquantum as tq
import torchquantum.functional as tqf


class VQC(tq.QuantumModule):
    """
    A variational quantum circuit (VQC, Variational Ansatz) consists of three parts: 
    (1) tensor product encoder; (2) variational ansatz; (3) measurement. We can 
    create an encoder by passing a list of gates to tq.GeneralEncoder. Each entry 
    in the list contains input_idx, func, and wires. Here, each qubit has a Pauli-Y 
    gate, which can convert the classical input data into the quantum states. Then, 
    we choose the variational ansatz such that each quantum channel is mutually entangled 
    and Pauli-X,Y,Z gates rotated by arbitrary angles. Finally, we perform Pauli-Z 
    measurements on each qubit on each qubit by creating a tq.MeasureAll module 
    and passing tq.PauliZ to it. The measure function will return four expectation 
    values from the qubits. 
    """
    def __init__(self, 
                 n_wires: int = 8,
                 n_qlayers: int = 1,
                 add_fc: bool = False,
                 out_features: int = 2):
        super().__init__()
        self.n_wires = n_wires 
        self.n_qlayers = n_qlayers
        self.add_fc = add_fc
            
        # Setting up tensor product encoder
        enc_cnt = list()
        for i in range(self.n_wires):
            cnt = {'input_idx': [i], 'func': 'ry', 'wires': [i]}
            enc_cnt.append(cnt)
        self.encoder = tq.GeneralEncoder(enc_cnt)
        
        # We create trainable model parameters, which are stored in dict 
        self.params_rx_dct = tq.QuantumModuleDict()
        self.params_ry_dct = tq.QuantumModuleDict()
        self.params_rz_dct = tq.QuantumModuleDict()
            
        for k in range(self.n_qlayers):
            for i in range(self.n_wires):
                self.params_rx_dct[f"{i + k * self.n_wires}"] = tq.RX(has_params=True, trainable=True)
                self.params_ry_dct[f"{i + k * self.n_wires}"] = tq.RY(has_params=True, trainable=True)
                self.params_rz_dct[f"{i + k * self.n_wires}"] = tq.RZ(has_params=True, trainable=True)
        # The observables are Hermitian operator based on Pauli-Z 
        self.measure = tq.MeasureAll(tq.PauliZ)
        
    @tq.static_support 
    def forward(self, 
                x: torch.Tensor, 
                q_device: tq.QuantumDevice):
        """
        1. To convert tq QuantumModule to qiskit or run in the static model,
        we need to:
            (1) add @tq.static_support before the forward
            (2) make sure to add
                static=self.static_mode and 
                parent_graph=self.graph
                to all the tqf functions, such as tqf.hadamard below
        """
        self.q_device = q_device
        self.encoder(self.q_device, x)
            
        for k in range(self.n_qlayers):
            for i in range(self.n_wires):
                self.params_rx_dct[f"{i + k * self.n_wires}"] 
                self.params_ry_dct[f"{i + k * self.n_wires}"] 
                self.params_rz_dct[f"{i + k * self.n_wires}"] 
            
            for i in range(self.n_wires):
                if i == self.n_wires-1:
                    tqf.cnot(self.q_device, wires=[i, 0], static=self.static_mode,
                             parent_graph=self.graph)
                else:
                    tqf.cnot(self.q_device, wires=[i, i+1], static=self.static_mode,
                             parent_graph=self.graph)
             
        if not self.add_fc:
            return self.measure(self.q_device)
        else:
            qc_out = self.measure(self.q_device)
            return self.fc_layer(qc_out)
        
    def reset_quantum_device(self, bsz):
        self.q_device.reset_states(bsz)

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_model(self, path: str):
        self.load_state_dict(torch.load(path))


class MPS_VQC(tq.QuantumModule):
    """
    An implementation of MPS-VQC. The VQC follows a matrix product state (MPS) structure in which the quantum circuits
    are used to create the quantum version of MPS.
    """
    def __init__(self,
                 n_wires: int = 8,
                 n_qlayers: int = 1,
                 tensor_product_enc = True,
                 add_fc: bool = False,
                 out_features: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.n_qlayers = n_qlayers

        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

        # Setting up tensor product encoder
        if tensor_product_enc:
            enc_cnt = list()
            for i in range(self.n_wires):
                cnt = {'input_idx': [i], 'func': 'ry', 'wires': [i]}
                enc_cnt.append(cnt)
            self.encoder = tq.GeneralEncoder(enc_cnt)
        else:
            self.encoder = tq.AmplitudeEncoder()

        # We create trainable model parameters, which are stored in dict
        self.params_rx_dct = tq.QuantumModuleDict()
        self.params_ry_dct = tq.QuantumModuleDict()
        self.params_rz_dct = tq.QuantumModuleDict()

        for k in range(self.n_qlayers):
            for i in range(self.n_wires):
                self.params_rx_dct[f"{i + k * self.n_wires}"] = tq.RX(has_params=True, trainable=True)
                self.params_ry_dct[f"{i + k * self.n_wires}"] = tq.RY(has_params=True, trainable=True)
                self.params_rz_dct[f"{i + k * self.n_wires}"] = tq.RZ(has_params=True, trainable=True)

        self.measure = tq.MeasureAll(tq.PauliZ)
        self.add_fc = add_fc
        if add_fc:
            self.fc_layer = torch.nn.Linear(self.n_wires, out_features)

    @tq.static_support
    def forward(self,
                x: torch.Tensor,
                q_device: tq.QuantumDevice):
        self.q_device = q_device
        self.encoder(self.q_device, x)

        for k in range(self.n_qlayers):
            for i in range(self.n_wires):
                self.params_rx_dct[f"{i + k * self.n_wires}"](self.q_device, wires=i)
                self.params_ry_dct[f"{i + k * self.n_wires}"](self.q_device, wires=i)
                self.params_rz_dct[f"{i + k * self.n_wires}"](self.q_device, wires=i)
                if i < self.n_wires - 1:
                    tqf.cnot(self.q_device, wires=[i, i + 1], static=self.static_mode,
                                parent_graph=self.graph)
        
        if not self.add_fc:
            return self.measure(self.q_device)
        else:
            qc_out = self.measure(self.q_device)
            return self.fc_layer(qc_out)
        
    def reset_quantum_device(self, bsz):
        self.q_device.reset_states(bsz)

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_model(self, path: str):
        self.load_state_dict(torch.load(path))


class MLP_MPS_VQC(tq.QuantumModule):
    """
    An implementation of MLP_MPS_VQC, which is an interplay of quantum and classical neural network. 
    The weight of hidden layer is controlled by a variaional quantum circuit. The hidden layer's weight 
    values are first transformed into quantum states using Amplitude Encoding, and the parametric VQC 
    outputs a new weight matrix for the hidden layer. The MLP_MPS_VQC resolves the potential problems
    VQC for machine learning, like the scaling of qubits, dimensionality, and trainability issues. 
    """
    def __init__(self,
                 n_wires: int = 8,
                 n_qlayers: int = 1,
                 input_dims: int = 16,
                 hidden_units: int = 128,
                 out_features: int = 2,
                 add_fc: bool = True):
        super().__init__()
        self.n_wires = n_wires
        self.n_qlayers = n_qlayers
        self.hidden_units = hidden_units
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        
        # Initialize weights
        self.W1 = torch.randn(input_dims, hidden_units, requires_grad=False)
        self.W2 = torch.nn.Linear(hidden_units, out_features)
        self.encoder = tq.AmplitudeEncoder()
        
        # Trainable quantum parameters
        self.params_rx_dct = tq.QuantumModuleDict()
        self.params_ry_dct = tq.QuantumModuleDict()
        self.params_rz_dct = tq.QuantumModuleDict()

        for k in range(self.n_qlayers):
            for i in range(self.n_wires):
                self.params_rx_dct[f"{i + k * self.n_wires}"] = tq.RX(has_params=True, trainable=True)
                self.params_ry_dct[f"{i + k * self.n_wires}"] = tq.RY(has_params=True, trainable=True)
                self.params_rz_dct[f"{i + k * self.n_wires}"] = tq.RZ(has_params=True, trainable=True)

        self.measure = tq.MeasureAll(tq.PauliZ)
        self.add_fc = add_fc
        if add_fc:
            self.fc_layer = nn.Linear(self.n_wires, self.hidden_units)
        
    def forward(self, 
                x: torch.Tensor,
                q_device: tq.QuantumDevice,
                is_train: bool = True,
                W1 = None):
        self.q_device = q_device
        if is_train:
            if W1 == None:
                W1 = self.W1
            self.encoder(self.q_device, self.W1)
            for k in range(self.n_qlayers):
                for i in range(self.n_wires):
                    self.params_rx_dct[f"{i + k * self.n_wires}"](self.q_device, wires=i)
                    self.params_ry_dct[f"{i + k * self.n_wires}"](self.q_device, wires=i)
                    self.params_rz_dct[f"{i + k * self.n_wires}"](self.q_device, wires=i)
                    
                    if i < self.n_wires - 1:
                        tqf.cnot(self.q_device, wires=[i, i + 1], static=self.static_mode,
                                 parent_graph=self.graph)
            
            qc_out = self.measure(self.q_device)
            if self.add_fc:
                qc_out = self.fc_layer(qc_out)    
            h = torch.relu(torch.matmul(x, qc_out))
            out = self.W2(h)
            return out, qc_out
        else:
            h = torch.relu(torch.matmul(x, W1))
            out = self.W2(h)
            return out

    def reset_quantum_device(self, bsz):
        self.q_device.reset_states(bsz)

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_model(self, path: str):
        self.load_state_dict(torch.load(path))


if __name__ == "__main__":
    
    test_signal = torch.randn(5000, 16)
    test_signal_label = test_signal_label = torch.randint(low=0, high=2, size=(5000,))
    dev = tq.QuantumDevice(n_wires=12, bsz=test_signal.shape[0])
    mlp_mps_vqc = MLP_MPS_VQC(n_wires=12, n_qlayers=1, hidden_units=128, input_dims=16,
                      add_fc=True, out_features=2)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp_mps_vqc.parameters(), lr=0.01)
    W1 = torch.randn(16, 128, requires_grad=False)
    
    for epoch in range(1000):
        optimizer.zero_grad()
        
        out, W1 = mlp_mps_vqc(test_signal, q_device=dev, is_train=True, W1=W1)
        loss = criterion(out, test_signal_label)
        
        loss.backward()
        optimizer.step()
        
        print(f'loss= {loss.item(): .4f}')
        
    with torch.no_grad():
        out = mlp_mps_vqc(test_signal, q_device=dev, is_train=False, W1=W1)
        _, predicted = torch.max(out.data, 1)
        correct = (predicted == test_signal_label).sum().item()
        accuracy = correct / test_signal_label.size(0)
        
        print(f'Test Accuracy: {accuracy:.4f}')
    
