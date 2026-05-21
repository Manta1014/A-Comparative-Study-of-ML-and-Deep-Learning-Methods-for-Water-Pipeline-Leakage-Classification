# A Comparative Study of Machine Learning and Deep Learning Methods for Water Pipeline Leakage Classification

This repository presents a comprehensive comparative study of **machine learning and deep learning approaches for water pipeline leakage classification**, using **real-world frequency spectrum data** collected from an urban water distribution network in South Korea.

The project was conducted in collaboration with **ETRI (Electronics and Telecommunications Research Institute)** and academic partners. The primary goal is to establish **reliable practical baselines** for leakage detection and to evaluate the **scalability and potential of deep learning and graph-based models** for intelligent water infrastructure monitoring.

👉 **[Paper (PDF)](https://github.com/Manta1014/A-Comparative-Study-of-ML-and-Deep-Learning-Methods-for-Water-Pipeline-Leakage-Classification/blob/main/%E1%84%89%E1%85%A1%E1%86%BC%E1%84%89%E1%85%AE%E1%84%83%E1%85%A9%E1%84%80%E1%85%AA%E1%86%AB%20%E1%84%82%E1%85%AE%E1%84%89%E1%85%AE%20%E1%84%90%E1%85%A1%E1%86%B7%E1%84%8C%E1%85%B5%E1%84%85%E1%85%B3%E1%86%AF%20%E1%84%8B%E1%85%B1%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%86%E1%85%A5%E1%84%89%E1%85%B5%E1%86%AB%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%20%E1%84%80%E1%85%B5%E1%84%87%E1%85%A5%E1%86%B8%20%E1%84%87%E1%85%B5%E1%84%80%E1%85%AD%20%E1%84%8B%E1%85%A7%E1%86%AB%E1%84%80%E1%85%AE.pdf)**

👉 **[Poster (PDF)](https://github.com/Manta1014/A-Comparative-Study-of-ML-and-Deep-Learning-Methods-for-Water-Pipeline-Leakage-Classification/blob/main/KSC25_%E1%84%89%E1%85%A1%E1%86%BC%E1%84%89%E1%85%AE%E1%84%83%E1%85%A9%20%E1%84%80%E1%85%AA%E1%86%AB%E1%84%86%E1%85%A1%E1%86%BC_poster.pdf)**

---

## 📌 Project Motivation

Water pipeline leakage poses a serious threat to urban infrastructure, leading to water loss, economic damage, road subsidence, and public safety risks. As pipeline networks age and environmental uncertainty increases, there is a growing demand for:

- Early and accurate leakage detection  
- Automated and scalable monitoring systems  
- Models robust to noisy and non-stationary sensor data  

While traditional machine learning methods such as **K-Nearest Neighbors (K-NN)** and **Random Forest** are widely adopted in practice due to their simplicity and interpretability, they often struggle with:

- High-dimensional frequency spectra  
- Temporal distribution shifts  
- Spatial correlations between sensors  

This project systematically compares **distance-based models, tree-based ensembles, neural networks, and graph transformers** to understand their strengths and limitations in real-world leakage detection scenarios.

---

## 📊 Dataset Description

- **Data source**: Urban water distribution network in Korea  
- **Sensors**: frequency sensors deployed across the network  
- **Observation window**: 5 consecutive days per sensor  
- **Frequency resolution**: 512 frequency bins per day  

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

## 🔬 Experimental Settings

To analyze model behavior under different assumptions, experiments were conducted under **three distinct data settings**:

### 1️⃣ Day-wise Independent Analysis
- Each day is treated as an independent sample  
- Input: `512 × 1 day` frequency spectrum  
- Focus: Short-term leakage signal detection  

### 2️⃣ Day-wise Trend Analysis
- Day-level statistics analyzed independently  
- Emphasizes per-day frequency pattern shifts  

### 3️⃣ Integrated Trend Analysis (Multi-day)
- Multiple days aggregated to capture temporal trends  
- Enables models to learn leakage progression patterns  

All experiments use an **8:1:1 split** for train / validation / test sets and are evaluated using **Accuracy, Precision, Recall, and F1-score**.

---

## 🧠 Models Evaluated

### 1️⃣ K-Nearest Neighbors (K-NN)
- Distance-based classification without training  
- Distances used:
  - Euclidean Distance  
  - Hausdorff Distance  
- Hyperparameter: `k ∈ {2, 3, 4, 5, 10, 15, 20}`  

**Observations**
- Euclidean K-NN performs well in low-k regimes for normal classes  
- Hausdorff distance is more robust to spectral shape variations  
- Performance degrades in integrated multi-day settings  

---

### 2️⃣ Random Forest
- Ensemble of decision trees with bootstrap aggregation  
- Learns frequency-wise importance implicitly  

**Observations**
- Most stable and interpretable baseline  
- Strong and consistent performance across all leakage classes  
- Particularly robust for **micro-leakage detection**  

---

### 3️⃣ Multi-Layer Perceptron (MLP)
- Fully connected neural network for representation learning  
- Techniques:
  - Batch Normalization  
  - ReLU activation  
  - Dropout regularization  

#### Hyperparameter Optimization
- **Optuna-based tuning applied**
- Tuned parameters include:
  - Learning rate  
  - Hidden layer size  
  - Dropout rate  

**Observations**
- Fine-tuned MLP significantly outperforms untuned versions  
- Demonstrates strong potential when data scale increases  

---

### 4️⃣ Convolutional Neural Network (CNN)
- Treats frequency spectrum as a 1D signal  
- Learns localized spectral patterns via convolution  

**Observations**
- Effective at capturing local frequency peaks  
- Lower recall for general leakage compared to MLP and RF  

---

### 5️⃣ Graph Transformer (Advanced Model)

To capture **spatial correlations between sensors**, a **Graph Transformer** model was implemented and evaluated.

#### Model Characteristics
- Nodes: Individual sensors  
- Edges: Similarity-based sensor relationships  
- Attention mechanism models long-range dependencies  

#### Optimized Hyperparameters (Optuna)

model_dim = 32
num_heads = 4
num_layers = 2
learning_rate = 0.00025
dropout = 0.1
batch_size = 64

**Observations**
- Graph Transformer achieves **top-tier precision and recall** for normal and micro leakage  
- Demonstrates strong robustness in trend-based settings  
- Highlights the importance of spatial context in leakage detection  

---

## 📈 Performance Summary (Representative Results)

| Model | Accuracy | Key Notes |
|------|----------|----------|
| K-NN (Euclidean) | ~0.97 (Day-wise) | Strong for normal class, sensitive to k |
| K-NN (Hausdorff) | ~0.70 | More robust to shape variation |
| Random Forest | **~0.93** | Most reliable practical baseline |
| MLP (Optuna tuned) | ~0.92 | Strong representation learning |
| CNN | ~0.88 | Good local pattern detection |
| Graph Transformer (Optuna) | **~0.96** | Best overall balance with spatial awareness |

---

## 🔍 Key Findings

- Traditional models remain strong baselines in real-world settings  
- Hyperparameter tuning is critical for deep learning performance  
- Graph-based models significantly improve robustness by leveraging sensor relationships  
- Micro-leakage detection benefits most from advanced architectures  

---

## 🔮 Future Work

- Full spatio-temporal Graph Transformer modeling (`512 × 5 days`)  
- Real-time streaming inference  
- Explainable attention visualization for infrastructure decision support  
- Deployment-oriented lightweight models  

---

## 📂 Repository Structure

Due to data security and collaboration constraints, the source code and raw sensor data are not publicly released.  
This repository focuses on **research documentation, experimental results, and reproducibility descriptions**.

```bash
A-Comparative-Study-of-ML-and-Deep-Learning-Methods-for-Water-Pipeline-Leakage-Classification/
├── README.md
│   └── Project overview, experimental settings, model comparison, and results
│
├── KSC25_상수도관망_poster.pdf
│   └── Conference poster summarizing the methodology and key experimental findings
│
├── 상수도관 누수 탐지를 위한 머신러닝 및 딥러닝 기법 비교 연구.pdf
│   └── Full research paper detailing dataset, models, experiments, and analysis
│
└── (Code and data not publicly available)
    └── Model implementations and raw sensor data are restricted due to security policies
```

🏛 Acknowledgements - This research was collaborated

Ulsan National Institute of Science and Technology (UNIST) & Electronics and Telecommunications Research Institute (ETRI)
