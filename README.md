# 🧠 Custom Neural Network - House Price Prediction

**Built entirely from scratch with NumPy** — no PyTorch, no TensorFlow, no deep learning frameworks.

---

## 📌 Overview

This project implements a **regression neural network from scratch** to predict house prices. Every component is manually coded:

- ✅ Forward propagation
- ✅ Backpropagation (chain rule)
- ✅ Adam optimizer
- ✅ ReLU activation
- ✅ He weight initialization
- ✅ Mini-batch gradient descent

---

## 🎯 Key Results

| Metric | Result |
|--------|--------|
| **Training Loss Reduction** | 96% (0.1187 → 0.0046) |
| **Test Loss** | 0.0104 |
| **Architecture** | 14 → 16 → 8 → 1 |

---

## 🧠 Architecture
Input (14 features)
↓
Hidden Layer 1 (16 neurons, ReLU)
↓
Hidden Layer 2 (8 neurons, ReLU)
↓
Output Layer (1 neuron)


---

## 🛠️ What I Built From Scratch

| Component | Implementation |
|-----------|----------------|
| **Forward Pass** | Matrix multiplication + bias + ReLU |
| **Backpropagation** | Chain rule derivatives |
| **Adam Optimizer** | Momentum + RMSprop from scratch |
| **ReLU Activation** | `max(0, x)` with derivative |
| **Weight Initialization** | He initialization (`√(2/n)`) |
| **Batch Training** | Mini-batch gradient descent (batch size 32) |

## 📊 Loss Reduction

```
Epoch 0:    0.1187
Epoch 500:  0.0200
Epoch 1000: 0.0080
Epoch 1500: 0.0050
Epoch 2000: 0.0046  ← 96% reduction!
```

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/ebukagerald/housing-custom-nn.git
cd housing-custom-nn

# Install dependencies
pip install numpy pandas

# Train the model
python train.py

📁 Project Structure
housing-custom-nn/
├── model.py          # Neural network architecture
├── train.py          # Training loop
├── backprop.py       # Manual backpropagation
├── adam.py           # Adam optimizer from scratch
├── relu.py           # ReLU activation
├── data/             # Dataset
└── README.md

💡 Why This Matters
Most ML engineers use frameworks like PyTorch or TensorFlow. I built the math behind them.

This project demonstrates:

Deep understanding of neural networks

Mathematical foundations of deep learning

Ability to implement complex algorithms from scratch

Strong Python and NumPy skills

🛠️ Tech Stack
Language: Python

Libraries: NumPy, Pandas

No frameworks: Everything is custom

🔗 Links
GitHub: github.com/ebukagerald/housing-custom-nn
LinkedIn: linkedin.com/in/ebukagerald

Built with ❤️ by Ebuka Gerald | Expert ML Engineer

