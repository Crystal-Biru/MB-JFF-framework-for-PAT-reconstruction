# MB-JFF: Model-Based Joint Feature Fusion Framework for PAT Reconstruction

This repository contains the official implementation of the **Model-Based Joint Feature Fusion (MB-JFF)** framework for **photoacoustic tomography (PAT) reconstruction**.

MB-JFF is designed to achieve **high-quality photoacoustic image reconstruction under limited-view detection configurations** by integrating forward model guidance with deep feature fusion.

---

## 🏗️ Framework Overview

The proposed **MB-JFF framework** combines physical modeling with deep neural networks to improve reconstruction quality from limited-view measurements.

The framework consists of two key components:

### 1. Model-Based Module
This module incorporates a **hybrid forward operator** to introduce forward model guidance into the reconstruction pipeline.  

### 2. Joint Feature Fusion Module
This module performs **joint feature fusion** between:

- the **original sensor data**, and  
- the **model-driven residual sensor data**

By combining these complementary sources of information, the network can better recover missing structures caused by limited-view acquisition.

---

## 🛠️ Installation

### Environment Requirements

- Python ≥ 3.8  
- PyTorch ≥ 1.10  
- CUDA (recommended for GPU acceleration)

### Install Dependencies

```bash
pip install -r requirements.txt
