# A Comparative Study of Machine Learning and Deep Learning Methods for Water Pipeline Leakage Classification

This repository presents a comprehensive comparative study of traditional machine learning and deep learning approaches for **water pipeline leakage classification**, based on **real-world frequency spectrum data** collected from an urban water distribution network in South Korea.

The project was conducted in collaboration with **ETRI (Electronics and Telecommunications Research Institute)** and academic partners, with the goal of establishing **practical baselines** for leakage detection and exploring the **potential of representation learning** for future intelligent water infrastructure systems.

👉 **[Paper (PDF)](https://github.com/Manta1014/A-Comparative-Study-of-ML-and-Deep-Learning-Methods-for-Water-Pipeline-Leakage-Classification/blob/main/%E1%84%89%E1%85%A1%E1%86%BC%E1%84%89%E1%85%AE%E1%84%83%E1%85%A9%E1%84%80%E1%85%AA%E1%86%AB%20%E1%84%82%E1%85%AE%E1%84%89%E1%85%AE%20%E1%84%90%E1%85%A1%E1%86%B7%E1%84%8C%E1%85%B5%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%8B%E1%85%B1%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%86%E1%85%A5%E1%84%89%E1%85%B5%E1%86%AB%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20%E1%84%80%E1%85%B5%E1%84%87%E1%85%A5%E1%86%B8%20%E1%84%87%E1%85%B5%E1%84%80%E1%85%AD%20%E1%84%8B%E1%85%A7%E1%86%AB%E1%84%80%E1%85%AE.pdf)**

👉 **[Poster (PDF)](https://github.com/Manta1014/A-Comparative-Study-of-ML-and-Deep-Learning-Methods-for-Water-Pipeline-Leakage-Classification/blob/main/KSC25_%E1%84%89%E1%85%A1%E1%86%BC%E1%84%89%E1%85%AE%E1%84%83%E1%85%A9%20%E1%84%80%E1%85%AA%E1%86%AB%E1%84%86%E1%85%A1%E1%86%BC_poster.pdf)**
---

## 📌 Project Motivation

Water pipeline leakage is a critical issue affecting urban infrastructure resilience, water resource sustainability, and public safety. Undetected or delayed leakage can lead to:

- Significant water loss and economic damage  
- Soil subsidence and road collapse  
- Secondary contamination and safety hazards  

With the increasing proportion of aging pipelines and growing environmental uncertainty, **automatic, accurate, and scalable leakage detection systems** are becoming essential.

While traditional machine learning methods such as **K-Nearest Neighbors (K-NN)** and **Random Forest** are widely used in practice due to their simplicity and interpretability, real-world sensor data often exhibit non-stationary behavior, environmental noise, and temporal pattern shifts. This project systematically evaluates whether **deep learning–based representation learning** can overcome these limitations and under what conditions such approaches become advantageous.

---

## 📊 Dataset Description

- **Source**: Urban water distribution network (Daegu, South Korea)  
- **Sensors**: 391 frequency sensors  
- **Observation period**: 5 consecutive days per sensor  
- **Input format**:  
  - 512-dimensional frequency spectrum per day  
  - One-day spectrum used as the base input in baseline experiments  
- **Total samples**: 1,955 sensor-day instances  

### Class Labels

| Label | Description |
|------|-------------|
| 0 | Normal |
| 1 | General leakage |
| 2 | Micro (fine) leakage |

Class distribution:
- Normal: 690  
- General leakage: 705  
- Micro leakage: 560  

---

## 🧠 Methods Compared

All models are trained and evaluated under the **same preprocessing and evaluation protocol**.

### 1️⃣ K-Nearest Neighbors (K-NN)
- Distance-based classification without explicit training  
- Uses Hausdorff distance to measure shape similarity between frequency spectra  
- Final prediction via majority voting  

**Pros**
- Simple and intuitive  
- Competitive in certain leakage categories  

**Cons**
- Poor scalability in high-dimensional spaces  
- Sensitive to noise and outliers  

---

### 2️⃣ Random Forest
- Ensemble of decision trees trained via bootstrap aggregation  
- Learns frequency-wise feature importance  

**Pros**
- Most stable and interpretable baseline  
- Strong performance across all leakage classes  
- Particularly robust for micro-leakage detection  

---

### 3️⃣ Multi-Layer Perceptron (MLP)
- Fully connected neural network for representation learning  
- Input normalization with Batch Normalization  
- ReLU activations and Dropout for regularization  

**Pros**
- Captures non-linear interactions between frequency bands  
- Shows strong potential as data scale increases  

**Cons**
- Performance depends on data quantity and augmentation  

---

### 4️⃣ Convolutional Neural Network (CNN)
- Treats frequency spectrum as a 1D signal  
- Learns local spectral patterns via convolutional filters  
- Pooling layers provide translation invariance  

**Pros**
- Effective at capturing localized frequency patterns  

**Cons**
- Lower recall for general leakage in this dataset  
- Limited global context modeling  

---

## 🧪 Experimental Setup

- **Input**: 512-dimensional frequency vector  
- **Data split**:  
  - Train: 80%  
  - Validation: 10%  
  - Test: 10%  
- **Evaluation metrics**:  
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  

All experiments follow an identical protocol to ensure fair comparison.

---

## 📈 Results Summary

| Model | Accuracy | Key Observations |
|------|----------|------------------|
| K-NN | ~0.35 | Struggles in high-dimensional spectral space |
| Random Forest | **~0.93** | Most stable and practical baseline |
| MLP | ~0.87 | Competitive representation learning performance |
| CNN | ~0.76 | Strong local patterns, weaker generalization |

### Key Findings

- Random Forest remains the strongest real-world baseline  
- Deep learning models benefit from increased data and richer representations  
- Distance-based methods degrade rapidly in high-dimensional frequency spaces  

---

## 🔮 Future Work

- Graph Transformer–based leakage detection using spatial sensor correlations  
- Multi-day temporal modeling (`512 × 5 days`)  
- Real-time deployment and streaming inference  
- Explainable AI for infrastructure decision support  

---

## 📂 Repository Structure (Example)

```bash
├── data/
│   └── processed_frequency_data/
├── models/
│   ├── knn.py
│   ├── random_forest.py
│   ├── mlp.py
│   └── cnn.py
├── experiments/
│   └── evaluation.ipynb
├── results/
│   └── metrics_tables/
├── README.md
```
