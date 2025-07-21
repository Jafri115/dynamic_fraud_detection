# 🔒 Dynamic Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Demo](https://img.shields.io/badge/Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/Wasifjafri/wiki-fraud-detection)
[![University](https://img.shields.io/badge/University-Hildesheim-purple.svg)](https://www.uni-hildesheim.de/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **SeqTab-OCAN**: A deep learning framework for detecting fraudulent behavior using both sequential and tabular data, trained with adversarial techniques for robustness.

---

## 🎮 Live Demo

🚀 [**Try it now on Hugging Face**](https://huggingface.co/spaces/Wasifjafri/wiki-fraud-detection)

Upload user behavior data and get instant fraud predictions through a clean web interface.

---

## 📖 Overview

Traditional fraud detection methods struggle with dynamic fraudulent behaviors and fail to effectively combine different data modalities. **SeqTab-OCAN** addresses these challenges through:

### 🌟 Key Innovation
- **Hybrid Data Fusion**: First framework to effectively combine sequential user behavior and tabular features
- **Time-Aware Processing**: Advanced attention mechanisms capture temporal relationships in user sequences
- **Adversarial Training**: OCAN with complementary GANs handles imbalanced datasets and improves outlier detection
- **Production Ready**: Complete end-to-end pipeline with REST API for real-world deployment

### 🏆 Achievement
Outperforms all established benchmarks with **92.27% Accuracy** and **92.01% F1-Score** on Wikipedia vandalism detection, representing a new state-of-the-art with improvements ranging from **0.61% to 5.67%** over verified VEWS baselines.

*This work was developed as part of a master's thesis in collaboration with the University of Hildesheim.*

---

## 🏗️ Architecture

### System Flow

```
┌─────────────────┐    ┌─────────────────┐
│  Sequential     │    │    Tabular      │
│     Data        │    │     Data        │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│ Time-Aware      │    │   Feature       │
│ Attention       │    │  Engineering    │
│   Network       │    │    Module       │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          └──────────┬───────────┘
                     ▼
          ┌─────────────────┐
          │ SeqTab Fusion   │
          │    Network      │
          └─────────┬───────┘
                    ▼
          ┌─────────────────┐
          │ OCAN Adversarial│
          │    Training     │
          └─────────┬───────┘
                    ▼
          ┌─────────────────┐
          │ Fraud Detection │
          │   Predictions   │
          └─────────────────┘
```

### Model Components

| Component            | Role                          | Technology            |
|---------------------|-------------------------------|------------------------|
| Sequential Encoder   | Learn temporal patterns       | LSTM + Attention       |
| Tabular Encoder      | Capture behavioral signals     | Dense Layers           |
| Fusion Layer         | Combine modalities             | Custom Neural Layer    |
| OCAN Module          | Detect anomalies               | GAN + One-Class SVM    |

---

## 🚀 Quick Start

### 🧪 Setup & Run

```bash
git clone https://github.com/Jafri115/dynamic_fraud_detection.git
cd dynamic_fraud_detection

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 🛠️ Core Commands

| Task           | Command                        | Description                       |
|----------------|--------------------------------|-----------------------------------|
| Train Model    | `python run_complete_pipeline.py` | Full model training pipeline      |
| Start API      | `python app.py`                | Launch FastAPI server (localhost) |
| API Docs       | Visit `localhost:8000/docs`    | Swagger/OpenAPI UI                |

---

## 📊 Dataset & Features

### 📁 UMD Wikipedia Vandalism Dataset

| Property        | Value         |
|-----------------|---------------|
| Users           | 33,000+       |
| Edits           | 770,000+      |
| Class Balance   | 20% Vandal    |
| Time Span       | 2013–2014     |
| Modalities      | Sequential + Tabular |

### 🔍 Feature Types

- **Sequential**: `edit_sequence`, `rev_time`, `time_delta_seq`  
- **Behavioral**: `meta_edit_ratio`, `reedit_score`, `unique_categories`  
- **Temporal**: `night_edit_ratio`, `fast_edit_ratio`, `weekend_edit_ratio`  
- **Session-based**: `sessions_count`, `avg_session_length`, `session_variance`


---

## 🎯 Performance

### ✅ Model Results

| Metric           | Value     |
|------------------|-----------|
| Accuracy         | **92.27%** |
| F1-Score         | **92.01%** |
| Precision        | 96.88%    |
| Recall           | 87.60%    |
| ROC AUC          | 97.08%    |
| Avg. Precision   | 97.48%    |

### 📈 Compared to VEWS Benchmarks

| VEWS Model       | Accuracy | Ours      | Improvement |
|------------------|----------|-----------|-------------|
| VEWS_WVB         | 86.60%   | 92.27%    | +5.67%      |
| VEWS_WTPM        | 87.39%   | 92.27%    | +4.88%      |
| VEWS_Combined    | 87.82%   | 92.27%    | +4.45%      |
| VEWS_Temporal    | 91.66%   | 92.27%    | +0.61%      |

---

## 📁 Project Structure

```
dynamic_fraud_detection/
├── src/                   # Core modules
│   ├── components/        # Feature engineering
│   ├── models/            # ML models
│   ├── pipeline/          # Training/prediction
│   └── utils/             # Helpers/utilities
├── app.py                 # FastAPI server
├── run_complete_pipeline.py
├── requirements.txt
├── tests/                 # Unit tests
├── notebooks/             # EDA & experimentation
```

---

## 📚 Research & Citation

Conducted as part of a Master's Thesis at **University of Hildesheim**.

```bibtex
@mastersthesis{murtaza2024seqtab,
  title={Dynamic Fraud Detection: Machine Learning Approaches for Identifying Anomalous Activities},
  author={Murtaza, Wasif},
  year={2024},
  school={University of Hildesheim},
  url={https://github.com/Jafri115/dynamic_fraud_detection}
}
```

---

## 🤝 Contributing

We welcome contributions!

- 🐛 Report issues
- 💡 Suggest enhancements
- 🧪 Improve tests
- 📚 Add docs or examples

Open a [GitHub issue](https://github.com/Jafri115/dynamic_fraud_detection/issues) or start a [discussion](https://github.com/Jafri115/dynamic_fraud_detection/discussions).

---

## 👤 Author

<div align="center">

**Wasif Murtaza**  
*Master’s Student, Data Science*  
[GitHub](https://github.com/Jafri115) • [Email](mailto:swasifmurtaza@gmail.com) • [Demo](https://huggingface.co/spaces/Wasifjafri/wiki-fraud-detection)

</div>
