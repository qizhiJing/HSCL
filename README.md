# Hierarchical Alignment for Multi-modal Representation Learning in Alzheimer’s Disease Diagnosis

## 🧠 Introduction

We propose a **Hierarchical Alignment** framework for effective multi-modal representation learning in Alzheimer’s Disease (AD) diagnosis. Our approach:

- Align features **within modalities** (intra-modal) using **supervised contrastive learning**
- Align features **across modalities** (inter-modal) via a shared space
- Introduce a **curriculum learning strategy** to transition smoothly from intra- to inter-modal alignment

This hierarchical structure ensures discriminative and well-structured representations that enhance diagnostic performance.

![](https://github.com/qizhiJing/Hierarchical-Alignment-for-Multi-modal-Representation-Learning-in-Alzheimer-s-Disease-Diagnosis/blob/master/images/-s1bvs7ct6glsj25c-001.jpg)

## 🧪 Key Features

- **Multi-modal input**: MRI, PET, and CSF features from the ADNI dataset
- **Supervised contrastive loss** for both intra- and inter-modal alignment
- **Curriculum learning** to gradually enforce hierarchical feature consistency
- **Achieved 96.74% accuracy** in AD diagnosis, outperforming SOTA methods

## 📁 Dataset

This work uses the publicly available [ADNI dataset](http://adni.loni.usc.edu/). We use three modalities:

- **MRI**: Structural imaging features
- **PET**: Functional imaging features
- **CSF**: Cerebrospinal fluid protein levels (e.g., Aβ42, T-tau, P-tau)

## 🏗️ Installation

Install required packages with:

```bash
pip install -r requirements.txt



