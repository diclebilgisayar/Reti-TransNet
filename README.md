# 👁️ Reti-TransNet: An Adaptive Gated Hybrid CNN-Transformer Framework for Diabetic Retinopathy Grading

![alt text](https://img.shields.io/badge/License-MIT-yellow.svg)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
![alt text](https://img.shields.io/badge/Framework-PyTorch-orange.svg)
![alt text](https://img.shields.io/badge/Task-Medical%20Imaging-blue.svg)

![alt text](images/figure_2.png)
<p align="center"><em>Fig 1: Overall workflow of Reti-TransNet architecture.</em></p>

This repository contains the official PyTorch implementation of the paper:
"Reti-TransNet: An Adaptive Gated Hybrid CNN-Transformer Framework with Explainability for Severity Grading of Diabetic Retinopathy".

# 📌 Abstract

Diabetic Retinopathy (DR) requires precise detection of both minute local lesions and global structural distortions. Reti-TransNet is a novel hybrid framework that synergizes the local feature extraction capability of EfficientNet-B0 with the global context modeling power of Swin Transformer. Unlike trivial concatenation strategies, we introduce a learnable "Adaptive Gated Fusion" mechanism to dynamically weight the importance of local versus global features based on image complexity.

## Key Achievements:

- 🏆 **State-of-the-Art Reliability**: Achieved a Quadratic Weighted Kappa of 0.90 on the APTOS 2019 dataset.

- 🌍 **Robust Generalization**: Demonstrated strong zero-shot performance on the external IDRiD dataset (AUC 0.949 for screening healthy patients).

- 🔍 **Explainability**: Integrated Grad-CAM++ ensures clinical transparency by localizing pathological biomarkers.

- ⚡ **Efficiency**: Trained in just 25 epochs on a single NVIDIA T4 GPU.

- 🏗️ **Architecture**: The proposed architecture consists of two parallel branches (CNN & Transformer) fused by a novel Adaptive Gating Mechanism.

# Reti-TransNet: Adaptive Gated Hybrid Framework for DR Grading

Official implementation of the paper **"Reti-TransNet: An Adaptive Gated Hybrid CNN-Transformer Framework with Explainability for Severity Grading of Diabetic Retinopathy"**.

## 📊 Experimental Results

Our model outperforms recent state-of-the-art methods in terms of reliability (Kappa) and screening safety, validated on both internal and external datasets.

### 1. Internal Validation (APTOS 2019)
| Metric | Result (95% CI) |
| :--- | :--- |
| **Quadratic Kappa ($\kappa$)** | **0.90** (0.88 - 0.92) |
| **Accuracy** | **84.17%** |
| **No DR Recall** | **97.5%** |
| **AUC (No DR)** | **0.997** |

<p align="center">
  <img src="images/Figure_4.png" width="45%" alt="Confusion Matrix"/>
  <img src="images/Figure_5.png" width="45%" alt="APTOS ROC"/>
</p>

### 2. External Validation (IDRiD - Zero-Shot)
Despite domain shift, the model maintains high screening reliability.

| Metric | Result |
| :--- | :--- |
| **AUC (No DR)** | **0.958** |
| **Kappa ($\kappa$)** | **0.76** |

<p align="center">
  <img src="images/Figure_6.png" width="60%" alt="IDRiD ROC"/>
</p>

## 🚀 How to Run

1. **Install Requirements:**
   ```bash
   pip install -r requirements.txt

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to Meta AI Research for the Reti-TransNet
- APTOS-2019 dataset creators and maintainers
- IDRiD dataset creators and maintainers
- All contributors and supporters

## Contact

For questions and feedback:
- 📧 Email:
- 🌟 Issues: [GitHub Issues](https://github.com/diclebilgisayar/Reti-TransNet/issues)
  
---
<p align="center">
  Made with ❤️ for the Medical Imaging Community
</p>
