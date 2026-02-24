# TensorHyper-VQC 
### A Tensor-Train-Guided Hypernetwork for Robust and Scalable Variational Quantum Computing

📄 **Published in npj Quantum Information (2026)** 
DOI: https://doi.org/10.1038/s41534-025-01157-z  

---

## 🧠 Framework Architecture

<p align="center">
  <img src="image/framework.png" width="750">
</p>

<p align="center">
  <em>Figure 1: TensorHyper-VQC architecture. A classical TT network generates variational parameters injected into a fixed quantum circuit. Gradients are backpropagated only through TT-cores, mitigating barren plateaus and enhancing noise robustness.</em>
</p>

---

## 🚀 Overview 

**TensorHyper-VQC** is a tensor-train (TT) guided hypernetwork framework designed to overcome two fundamental challenges in Variational Quantum Computing (VQC):

- Barren plateaus (vanishing gradients)
- Sensitivity to quantum noise
- Poor scalability with increasing qubits and circuit depth

Instead of directly optimizing quantum gate parameters on hardware, TensorHyper-VQC delegates parameter generation to a classical Tensor-Train (TT) network.

The quantum circuit acts only as a forward-pass evaluator, while all gradient updates occur in the classical domain.

This classical–quantum decoupling results in:

- ✔ Improved trainability (NTK enhancement)
- ✔ $\mathcal{O}\left(\frac{1}{UL}\right)$ gradient variance reduction
- ✔ Stronger generalization control via low-rank structure
- ✔ Hardware-level robustness without explicit mitigation

---

## 🧠 Core Idea 

For a VQC with $U$ qubits and $L$ layers,

A conventional VQC directly optimizes:   $\textbf{w} \in \mathbb{R}^{3UL}$

TensorHyper-VQC instead parameterizes:   $\textbf{w} = \text{TT}(\textbf{z}; \mathcal{G}_1, ..., \mathcal{G}_K)$

Where:

- $\textbf{z} \sim \mathcal{N}(0, I)$
- $\mathcal{G}_1, ..., \mathcal{G}_K$ are TT-cores
- Optimization is performed only over TT-cores
- Quantum circuit performs inference only

---

## 📐 Theoretical Guarantees

TensorHyper-VQC is supported by rigorous theoretical analysis: 

### 1️⃣ Approximation Capability 

TT approximation error decreases polynomially with TT-rank: 

$ε_{\text{tt}}(r) = \mathcal{O}\left(\frac{1}{r^{(p−1)}}\right)$

Provides explicit rank–expressivity trade-off.

### 2️⃣ Neural Tangent Kernel (NTK) Analysis

Minimum NTK eigenvalue satisfies: 

---

Our codes include TensorHyper-VQC experiments for Quantum Dot Classification, Max-Cut Maximization, and LiH Molecular Simulation. 

#### Installation

The main dependencies include *pytorch*

#### 0. Downloading the dataset 
```
git clone https://gitlab.com/QMAI/mlqe_2023_edx.git
```

#### 1. Simulating experiments of Quantum Dot Classification 
python TensorHyper_QD.py 

#### 2. Simulating experiments of Max-Cut Optimization
python TensorHyper_QAOA.py

#### 3. Simulating experiments of LiH Molecular Simulation
python TensorHyper_QSIM.py
