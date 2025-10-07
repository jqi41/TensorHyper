#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MetaTT-VQC for Quantum-Dot Classification
  – TensorTrainLayer for parameter generation
  – VQC, MPS_VQC, TTN_VQC consuming generated angles
  – Meta-wrappers: VQCParamVQC, MPSParamVQC, TTNParamVQC w/ residual global angles for ring-VQC
  – DataLoader over 50×50 noisy diagrams
  – Adam + LR scheduler
Requires: torch, torchquantum, numpy
"""

import math
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchquantum as tq
import torchquantum.functional as tqf
from torch.utils.data import TensorDataset, DataLoader


# reproducibility
seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)

##################################
# TensorTrainLayer Definition
##################################
TT_INIT_SCALE = 0.01

class TensorTrainLayer(nn.Module):
    def __init__(self, input_dims, output_dims, tt_ranks):
        super().__init__()
        d = len(input_dims)
        assert len(output_dims)==d and len(tt_ranks)==d+1
        self.input_dims, self.output_dims, self.tt_ranks = input_dims, output_dims, tt_ranks
        self.tt_cores = nn.ParameterList()
        for k in range(d):
            r0, r1 = tt_ranks[k], tt_ranks[k+1]
            n_k, m_k = input_dims[k], output_dims[k]
            core = nn.Parameter(torch.randn(r0,n_k,m_k,r1)*TT_INIT_SCALE)
            nn.init.xavier_uniform_(core)
            self.tt_cores.append(core)
        self.bias = nn.Parameter(torch.zeros(math.prod(output_dims)))

    def forward(self, x):
        # x: [bsz, prod(input_dims)]
        bsz = x.size(0)
        x_rs = x.view(bsz, *self.input_dims)
        batch='b'
        letters=[chr(i) for i in range(ord('a'),ord('z')+1) if chr(i)!=batch]
        d=len(self.input_dims)
        iL, oL, rL = letters[:d], letters[d:2*d], letters[2*d:2*d+d+1]
        inp = batch + ''.join(iL)
        cores = [f"{rL[k]}{iL[k]}{oL[k]}{rL[k+1]}" for k in range(d)]
        outp = batch + ''.join(oL)
        eins = inp + ',' + ','.join(cores) + '->' + outp
        out = torch.einsum(eins, x_rs, *self.tt_cores)
        
        return out.reshape(bsz, -1) + self.bias
    
##################################
# Base VQC Definition (ring entangler)
##################################
class VQC(tq.QuantumModule):
    def __init__(self, n_wires=12, n_qlayers=2,
                 tensor_product_enc=True, add_fc=False,
                 out_features=2, noise_prob=0.01):
        super().__init__()
        self.n_wires, self.n_qlayers = n_wires, n_qlayers
        self.noise_prob, self.add_fc = noise_prob, add_fc
        # global angles (L, W, 3)
        self.angles = nn.Parameter(torch.randn(n_qlayers,n_wires,3)*0.1)

        if tensor_product_enc:
            cfg=[{'input_idx':[i],'func':'ry','wires':[i]} for i in range(n_wires)]
            self.encoder = tq.GeneralEncoder(cfg)
        else:
            self.encoder = tq.AmplitudeEncoder()

        self.measure = tq.MeasureAll(tq.PauliZ)
        if add_fc:
            self.fc = nn.Linear(n_wires, out_features)
         
            
    def reset_quantum_device(self, bsz):
        self.q_device.reset_states(bsz)

    def depolarize(self):
        for i in range(self.n_wires):
            if torch.rand((),device=self.q_device.device)<self.noise_prob:
                err=torch.randint(0, 3, (), device=self.q_device.device).item()
                op = tqf.x if err==0 else (tqf.y if err==1 else tqf.z)
                op(self.q_device,wires=i,static=self.static_mode,parent_graph=self.graph)

    def entangle_ring(self):
        for i in range(self.n_wires-1):
            tqf.cnot(self.q_device,wires=[i,i+1],static=self.static_mode,parent_graph=self.graph)
        tqf.cnot(self.q_device,wires=[self.n_wires-1,0],static=self.static_mode,parent_graph=self.graph)

    @tq.static_support
    def forward(self, x, q_device, angles=None):
        self.q_device = q_device
        bsz = x.size(0)
        self.reset_quantum_device(bsz)
        self.encoder(self.q_device, x)
        use_batch = angles is not None
        for k in range(self.n_qlayers):
            for w in range(self.n_wires):
                if use_batch:
                    r,y,z = angles[:,k,w,0], angles[:,k,w,1], angles[:,k,w,2]
                else:
                    r,y,z = self.angles[k,w]
                tqf.rx(self.q_device,wires=w,params=r,static=self.static_mode,parent_graph=self.graph)
                tqf.ry(self.q_device,wires=w,params=y,static=self.static_mode,parent_graph=self.graph)
                tqf.rz(self.q_device,wires=w,params=z,static=self.static_mode,parent_graph=self.graph)
            self.entangle_ring()
            self.depolarize()
        out = self.measure(self.q_device)
        return self.fc(out) if self.add_fc else out


##################################
# MPS_VQC & TTN_VQC (inherit ring code but override entanglers)
##################################
class MPS_VQC(VQC):
    """Nearest‐neighbor chain entangler."""
    @tq.static_support
    def forward(self, x, q_device, angles=None):
        self.q_device = q_device
        bsz = x.size(0)
        self.reset_quantum_device(bsz)
        self.encoder(self.q_device, x)
        use_batch = angles is not None
        for k in range(self.n_qlayers):
            for w in range(self.n_wires):
                if use_batch:
                    r = angles[:, k, w, 0]
                    y = angles[:, k, w, 1]
                    z = angles[:, k, w, 2]
                else:
                    r, y, z = self.angles[k, w]
                tqf.rx(self.q_device, wires=w, params=r, static=self.static_mode, parent_graph=self.graph)
                tqf.ry(self.q_device, wires=w, params=y, static=self.static_mode, parent_graph=self.graph)
                tqf.rz(self.q_device, wires=w, params=z, static=self.static_mode, parent_graph=self.graph)
                if w < self.n_wires - 1:
                    tqf.cnot(self.q_device, wires=[w, w+1], static=self.static_mode, parent_graph=self.graph)
            self.depolarize()
        out = self.measure(self.q_device)
        return self.fc(out) if self.add_fc else out
    
    
class TTN_VQC(VQC):
    """Binary‐tree entangler."""
    def entangle_ttn(self):
        half = self.n_wires // 2
        for i in range(half):
            tqf.cnot(self.q_device, wires=[i, i+half], static=self.static_mode, parent_graph=self.graph)

    @tq.static_support
    def forward(self, x, q_device, angles=None):
        self.q_device = q_device
        bsz = x.size(0)
        self.reset_quantum_device(bsz)
        self.encoder(self.q_device, x)
        use_batch = angles is not None
        for k in range(self.n_qlayers):
            for w in range(self.n_wires):
                if use_batch:
                    r = angles[:, k, w, 0]
                    y = angles[:, k, w, 1]
                    z = angles[:, k, w, 2]
                else:
                    r, y, z = self.angles[k, w]
                tqf.rx(self.q_device, wires=w, params=r, static=self.static_mode, parent_graph=self.graph)
                tqf.ry(self.q_device, wires=w, params=y, static=self.static_mode, parent_graph=self.graph)
                tqf.rz(self.q_device, wires=w, params=z, static=self.static_mode, parent_graph=self.graph)
            self.depolarize()
        self.entangle_ttn()
        self.depolarize()
        out = self.measure(self.q_device)
        return self.fc(out) if self.add_fc else out
    

##################################
# MetaTT Wrappers
##################################
class VQCParamVQC(nn.Module):
    """TT → ring-VQC with residual global angles"""
    def __init__(self, input_dim, tt_input_dims, tt_output_dims, tt_ranks, **vqc_kwargs):
        super().__init__()
        self.tt  = TensorTrainLayer(tt_input_dims,tt_output_dims,tt_ranks)
        self.vqc = VQC(**vqc_kwargs)
    def forward(self,x,q_device):
        bsz=x.size(0)
        # residual: global + delta
        G = self.vqc.angles.unsqueeze(0).expand(bsz,-1,-1,3)
        angles = self.tt(x).reshape(bsz, self.vqc.n_qlayers, self.vqc.n_wires, 3)
        return self.vqc(x,q_device,angles=G+angles)

class MPSParamVQC(nn.Module):
    """TT → MPS_VQC"""
    def __init__(self,input_dim,tt_input_dims,tt_output_dims,tt_ranks,**kw):
        super().__init__(); self.tt=TensorTrainLayer(tt_input_dims,tt_output_dims,tt_ranks)
        self.vqc=MPS_VQC(**kw)
    def forward(self,x,q_device):
        bsz=x.size(0)
        ang = self.tt(x).reshape(bsz, self.vqc.n_qlayers, self.vqc.n_wires, 3)
        return self.vqc(x,q_device,angles=ang)

class TTNParamVQC(nn.Module):
    """TT → TTN_VQC"""
    def __init__(self,input_dim,tt_input_dims,tt_output_dims,tt_ranks,**kw):
        super().__init__(); self.tt=TensorTrainLayer(tt_input_dims,tt_output_dims,tt_ranks)
        self.vqc=TTN_VQC(**kw)
    def forward(self,x,q_device):
        bsz=x.size(0)
        ang = self.tt(x).reshape(bsz, self.vqc.n_qlayers, self.vqc.n_wires, 3)
        return self.vqc(x,q_device,angles=ang)
    

##################################
# Quantum-Dot DataLoader
##################################
def load_quantum_dot_data():
    # replace paths with your .npy files
    X_clean = np.load("./mlqe_2023_edx/week1/dataset/csds_noiseless.npy")      # (2000,50,50)
    X_noisy = np.load("./mlqe_2023_edx/week1/dataset/csds.npy")                # (2000,50,50)
    y       = np.load("./mlqe_2023_edx/week1/dataset/labels.npy")              # (2000,)
    # use noisy for training/testing
    X = X_noisy.reshape(-1,2500).astype(np.float32)
    y = y.astype(np.int64)
    # split 90/10
    split = int(0.9*len(y))
    X_train,y_train = X[:split], y[:split]
    X_test, y_test  = X[split:], y[split:]
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))
    
    return train_ds, test_ds


##################################
# Training Script
##################################
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='Training a hybrid quantum-classical model (MLP_VQC variants) for charge stability diagram classification of quantum dots with depolarizing noise simulation.'
    )
    parser.add_argument('--save_path', metavar='DIR', default='models', help='Path to save the trained model')
    parser.add_argument('--num_qubits', default=20, help='Number of qubits in the quantum circuit', type=int)
    parser.add_argument('--batch_size', default=64, help='Batch size for training', type=int)
    parser.add_argument('--num_epochs', default=21, help='Number of training epochs', type=int)
    parser.add_argument('--depth_vqc', default=6, help='Depth (number of variational layers) of the VQC', type=int)
    parser.add_argument('--lr', default=3e-2, help='Learning rate', type=float)
    parser.add_argument('--test_kind', metavar='DIR', default='gen', help='Test type: "rep" for representation, "gen" for generalization')
    parser.add_argument('--model_kind', metavar='DIR', default='ring', help='Model type: ring, mps, tree')
    parser.add_argument('--noise_prob', default=0.000, help='noisy_error_rate', type=float)
    parser.add_argument('--tt_input_dim', default=[5, 10, 5, 10], help='tensor-train input dimensions', type=list[int])
    parser.add_argument('--tt_output_dim', default=[4, 2, 3, 9], help='tensor-train output dimensions', type=list[int])
    parser.add_argument('--tt_ranks', default=[1, 2, 5, 2, 1], help='tensor-train output dimensions', type=list[int])

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, test_ds = load_quantum_dot_data()
    train_loader = DataLoader(train_ds, batch_size=args.num_epochs, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.num_epochs)
    epochs = args.num_epochs

    # choose model: 'ring','mps','ttn'
    mode = args.model_kind
    common = dict(
        input_dim=2500,
        #tt_input_dims=[5, 10, 5, 10],
        tt_input_dims=args.tt_input_dim,
        tt_output_dims=args.tt_output_dim,
        tt_ranks=args.tt_ranks,
        n_wires=args.num_qubits,
        n_qlayers=args.depth_vqc,
        tensor_product_enc=True,
        add_fc=False,
        out_features=2,
        noise_prob=args.noise_prob
    )
    if mode=='ring':
        model = VQCParamVQC(**common).to(device)
    elif mode=='mps':
        model = MPSParamVQC(**common).to(device)
    else:
        model = TTNParamVQC(**common).to(device)

    print("Params:", sum(p.numel() for p in model.parameters()))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(1, epochs):
        # ---- Training ----
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        for X_batch, y_batch in train_loader:
             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
             optimizer.zero_grad()
             q_dev = tq.QuantumDevice(n_wires=args.num_qubits, bsz=X_batch.size(0)).to(device)
             out = model(X_batch, q_dev)
             loss = criterion(out, y_batch)
             loss.backward()
             optimizer.step()

             # accumulate train loss & accuracy
             total_loss += loss.item() * X_batch.size(0)
             preds = out.argmax(dim=1)
             correct_train += (preds == y_batch).sum().item()
             total_train += X_batch.size(0)

        scheduler.step()
        train_loss = total_loss / len(train_ds)
        train_acc  = correct_train / total_train

        # ---- Evaluation ----
        model.eval()
        total_test_loss = 0.0
        correct_test = 0
        total_test = 0
         
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                q_dev = tq.QuantumDevice(n_wires=args.num_qubits, bsz=X_batch.size(0)).to(device)
                out = model(X_batch, q_dev)

                # accumulate test loss & accuracy
                loss = criterion(out, y_batch)
                total_test_loss += loss.item() * X_batch.size(0)
                preds = out.argmax(dim=1)
                correct_test += (preds == y_batch).sum().item()
                total_test += X_batch.size(0)

        test_loss = total_test_loss / len(test_ds)
        test_acc  = correct_test / total_test

        print(f"Epoch {epoch:02d}  "
              f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}  "
              f"Test loss:  {test_loss :.4f}, Test acc:  {test_acc :.4f}")
