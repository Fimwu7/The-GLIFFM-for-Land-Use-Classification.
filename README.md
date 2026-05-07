
# [Land-Use Classification of High-Resolution Remote Sensing Imagery Incorporating Global–Local Interactive Features Across Optical Domain and Land Cover Primitives](https://ieeexplore.ieee.org/document/11417990/metrics#metrics)

---

> This is the official PyTorch implementation of "[Land-Use Classification of High-Resolution Remote Sensing Imagery Incorporating Global–Local Interactive Features Across Optical Domain and Land Cover Primitives](https://ieeexplore.ieee.org/document/11417990/metrics#metrics)".
>
> Paper: [IEEE TGRS](https://ieeexplore.ieee.org/document/11417990/metrics#metrics) 
> 
# Introduction

**GLIFFM** is a novel deep learning framework designed for land-use classification. It achieves superior performance by effectively incorporating land-cover primitives and integrating global-local interactive features across the optical domain.
<img width="835" height="276" alt="Overall0729 drawio" src="https://github.com/user-attachments/assets/4a1820cf-c70c-4c5a-9697-f90545feab5b" />


---
# Requirements
```yaml
# Environments:
cuda==11.6
python==3.8.20

# Packages:
mmcv==1.7.1
timm==0.6.12
torch==1.12.1
torchvision==0.13.1
segmentation_models_pytorch
```
---
# 🤝 Acknowledgements

This project is built upon the following excellent open-source work:

* **[TransXNet](https://github.com/LMMMEng/TransXNet)**: We employ TransXNet as the backbone in our **GLIFFM** architecture to capture comprehensive global-local dynamics. We sincerely thank the authors for their inspiring work and for making their code publicly available.
    
    > **Reference:** M. Lou et al., "TransXNet: Learning Both Global and Local Dynamics With a Dual Dynamic Token Mixer for Visual Recognition," *IEEE TNNLS*, 2025.

* **[Segmentation Models PyTorch](https://github.com/qubvel-org/segmentation_models.pytorch)**: We thank the contributors of `segmentation_models_pytorch` for providing efficient implementations of various segmentation architectures.

* **Dataset Providers**: We are grateful to the providers of the **GID5** and **EuroSAT** datasets for their valuable contributions to the remote sensing community.
