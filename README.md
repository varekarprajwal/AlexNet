# 🚀 PyTorch AlexNet (CIFAR-10) — From Fundamentals to GPU-Aware Training

## 📌 Overview

This project implements an **AlexNet-inspired Convolutional Neural Network (CNN)** from scratch using PyTorch and trains it on the CIFAR-10 dataset.

Beyond a basic implementation, the project explores:
* **Deep learning fundamentals** (CNNs, training pipeline)
* **Optimization techniques** (normalization, learning rate tuning)
* **Hardware-aware execution** (CPU vs GPU vs Apple MPS)

The goal is not just to train a model, but to understand **how deep learning models behave, scale, and execute efficiently on modern hardware.**

---

## 🧠 1. Foundations — What is AlexNet?

AlexNet is a landmark CNN architecture that demonstrated the power of deep convolutional networks for image classification.

**Core components:**
* **Convolution layers** → feature extraction
* **ReLU activation** → non-linearity
* **Pooling** → spatial reduction
* **Fully connected layers** → classification

### 🔍 Why AlexNet?
* Simple yet powerful architecture
* Good balance between model complexity and training feasibility
* Ideal for understanding CNN internals

---

## ⚙️ 2. Problem Setup

### Dataset: CIFAR-10
* **Images:** 60,000
* **Classes:** 10
* **Image size:** 32×32×3

### The Challenge
AlexNet was originally designed for 224×224 images, but CIFAR-10 is much smaller.

**👉 Solution:**
* Modify the architecture to match the dataset dimensions.
* Reduce the fully connected layer size.
* Adjust the convolution design (e.g., smaller kernel sizes).

---

## 🏗️ 3. Model Architecture (Modified AlexNet)

### Key Design Decisions
* Smaller kernel sizes (3×3 instead of 11×11)
* Reduced fully connected layers
* Preserved deep hierarchical feature extraction

### 🔁 Forward Flow
```text
Input → Conv → ReLU → Pool  
      → Conv → ReLU → Pool  
      → Conv → Conv → Conv  
      → Pool → Flatten  
      → Fully Connected → Output
```
## 📦 4. Data Pipeline

### Preprocessing Steps
* `transforms.ToTensor()`
* `transforms.Normalize(mean, std)`

### Why Normalization Matters
* **Without normalization:** Gradients become unstable, and training gets stuck (observed: loss ≈ 2.30).
* **After normalization:** Stable gradients and faster convergence.

---

## 🔄 5. Training Pipeline

### Core Steps
1. Forward pass
2. Loss computation
3. Backpropagation
4. Parameter update

### Loss Function
* `CrossEntropyLoss` (Standard for multi-class classification)

### Optimizer
* **Adam optimizer**
* **Learning rate:** `0.0001`

### Training Mode
Using `model.train()` ensures:
* Dropout is active
* Proper training behavior is enforced

---

## 📊 6. Evaluation Pipeline

Using `model.eval()` ensures:
* Randomness (like Dropout) is disabled
* Consistent and deterministic inference

**Accuracy Calculation:**
```python
accuracy = correct / total
```
## 🚀 7. Results

### Training Performance

| Epoch | Loss | Accuracy |
| :---: | :---: | :---: |
| **1** | 1.69 | 47.7% |
| **2** | 1.34 | 53.7% |
| **3** | 1.18 | 59.5% |
| **4** | 1.07 | 63.4% |
| **5** | 0.97 | 65.6% |

### Observations
* **Loss decreases steadily:** The model is effectively learning features.
* **Accuracy improves consistently:** The architecture is well-suited for the CIFAR-10 complexity.
* **No overfitting in early epochs:** The regularization and architecture size are keeping the model generalizable.