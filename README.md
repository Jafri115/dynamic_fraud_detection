# 🔒 Dynamic Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Research](https://img.shields.io/badge/Research-Fraud%20Detection-red.svg)](https://github.com/Jafri115/dynamic_fraud_detection)
[![University](https://img.shields.io/badge/University-Hildesheim-purple.svg)](https://www.uni-hildesheim.de/)
[![License](https://img.shields.io/badge/License-Academic-yellow.svg)](LICENSE)
[![Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Demo-yellow)](https://huggingface.co/spaces/Wasifjafri/wiki-fraud-detection)

> **SeqTab-OCAN**: A state-of-the-art machine learning framework for fraud detection combining sequential and tabular data using deep learning and adversarial networks.

## 🎮 Live Demo

🚀 **[Try the Interactive Demo on Hugging Face →](https://huggingface.co/spaces/Wasifjafri/wiki-fraud-detection)**

Experience real-time fraud detection with our deployed model. Upload user data and get instant predictions through an intuitive web interface.

---

## 📖 Overview

Traditional fraud detection methods struggle with dynamic fraudulent behaviors and fail to effectively combine different data modalities. **SeqTab-OCAN** addresses these challenges through:

### 🌟 Key Innovation
- **Hybrid Data Fusion**: First framework to effectively combine sequential user behavior and tabular features
- **Time-Aware Processing**: Advanced attention mechanisms capture temporal relationships in user sequences
- **Adversarial Training**: OCAN with complementary GANs handles imbalanced datasets and improves outlier detection
- **Production Ready**: Complete end-to-end pipeline with REST API for real-world deployment

### 🏆 Achievement
Outperforms all baseline models with **93.96% F1-Score** and **98.17% AUC-PR** on Wikipedia vandalism detection.

*This work was developed as part of a master's thesis in collaboration with the University of Hildesheim.*

## 📋 Table of Contents

- [🎮 Live Demo](#-live-demo)
- [📖 Overview](#-overview)
- [🚀 Quick Start](#-quick-start)
- [🏗️ Architecture](#️-architecture)
- [📊 Dataset & Features](#-dataset--features)
- [🎯 Performance](#-performance)
- [💻 Development](#-development)
- [📚 Research](#-research)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🚀 Quick Start

### ⚡ One-Click Setup

```bash
# Clone and setup
git clone https://github.com/Jafri115/dynamic_fraud_detection.git
cd dynamic_fraud_detection

# Create environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 🎯 Usage

| Task | Command | Description |
|------|---------|-------------|
| **Train Model** | `python run_complete_pipeline.py` | Complete training pipeline |
| **Start API** | `python app.py` | Launch FastAPI server (localhost:8000) |
| **API Docs** | Visit `localhost:8000/docs` | Interactive API documentation |

## 🏗️ Architecture

### System Overview

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

### 🔄 Processing Pipeline

1. **Data Ingestion**: Raw Wikipedia edit logs → Structured sequences + features
2. **Feature Engineering**: Temporal, behavioral, and statistical feature extraction  
3. **Sequence Processing**: Time-aware attention captures temporal dependencies
4. **Feature Fusion**: Advanced neural fusion of sequential and tabular representations
5. **Adversarial Training**: OCAN handles class imbalance and improves robustness
6. **Prediction**: Real-time fraud probability scoring

### 🧠 Model Architecture

| Component | Purpose | Technology |
|-----------|---------|------------|
| **Sequential Encoder** | Temporal pattern recognition | LSTM + Attention |
| **Tabular Encoder** | Statistical feature processing | Dense Networks |
| **Fusion Layer** | Multi-modal integration | Custom Architecture |
| **OCAN Module** | Adversarial anomaly detection | GANs + One-Class |

## 📊 Dataset & Features

### 🗃️ UMDWikipedia Dataset

| Metric | Value | Description |
|--------|-------|-------------|
| **Users** | 33,000+ | Mix of vandals and legitimate editors |
| **Edits** | 770,000 | Wikipedia edit transactions |
| **Distribution** | 20% vandal / 80% benign | Realistic imbalanced scenario |
| **Time Span** | 2013-2014 | Historical Wikipedia data |

### 🔧 Engineered Features

Our feature engineering transforms raw edit logs into meaningful behavioral indicators:

#### Temporal Features
- `edit_frequency` - Average time between consecutive edits
- `night_edits` - Activity during 18:00-06:00 (suspicious pattern)
- `day_edits` - Activity during 06:00-18:00 (normal pattern)
- `weekend_edits` - Weekend activity levels

#### Behavioral Features  
- `total_edits` - User's complete edit history
- `unique_pages` - Diversity of edited content
- `reverted_ratio` - Proportion of reverted contributions
- `cluebot_reverts` - Automated detection reversals
- `category_diversity` - Breadth of topic engagement

### 💾 Data Processing

```python
# Example: Feature extraction pipeline
from src.components.data_transformation import DataTransformer

# Initialize transformer
transformer = DataTransformer()

# Process raw edit sequences
features = transformer.extract_features(user_edit_logs)
sequences = transformer.create_sequences(user_edit_logs)

# Combined dataset ready for training
processed_data = transformer.combine_features(features, sequences)
```

## 🎯 Performance

### 🏆 Benchmark Results

| Model | Data Type | Precision | Recall | **F1-Score** | **AUC-PR** | AUC-ROC |
|-------|-----------|-----------|---------|-------------|-----------|---------|
| OCAN Baseline | Sequential | 91.17±0.7% | 90.97±0.8% | 91.07±0.3% | 88.38±0.4% | 97.10±0.3% |
| Tab-RL | Tabular | 90.42±0.2% | 79.96±0.4% | 84.87±0.2% | 92.40±0.1% | 90.79±0.3% |
| Seq-RL | Sequential | 94.70±1.8% | 90.26±1.5% | 92.41±0.3% | 97.18±1.1% | 97.54±0.5% |
| SeqTab-RL | Seq+Tab | 95.29±1.8% | 89.31±3.2% | 92.25±1.1% | 97.35±0.1% | 97.32±0.2% |
| **SeqTab-OCAN** | **Seq+Tab** | **93.07±0.03%** | **94.87±0.02%** | **🥇 93.96±0.01%** | **🥇 98.17±0.02%** | **93.79±0.02%** |

### 📈 Key Achievements

| Metric | Achievement | Significance |
|--------|-------------|--------------|
| **🎯 F1-Score** | **93.96%** *(±0.01%)* | Best overall performance with exceptional stability |
| **🎪 AUC-PR** | **98.17%** *(±0.02%)* | Excellent precision-recall trade-off for imbalanced data |
| **⚡ Stability** | **Ultra-low variance** | Extremely robust across different data splits |
| **🔄 Reproducibility** | **Consistent results** | Reliable performance in production environments |

### 🔍 Analysis

- **✅ Superior Balance**: Achieves optimal precision-recall trade-off
- **✅ Robust Training**: Minimal variance indicates stable learning
- **✅ Real-world Ready**: Performance maintained across diverse scenarios  
- **✅ Scalable**: Efficient processing of large-scale sequential data

## 💻 Development

### 🗂️ Project Structure

```
dynamic_fraud_detection/
├── 📁 src/                     # Core source code
│   ├── components/             # Data processing pipeline
│   ├── models/                 # ML model definitions  
│   ├── pipeline/               # Training & prediction pipelines
│   ├── training/               # Training utilities & configs
│   └── utils/                  # Helper functions & tools
├── 🧪 tests/                   # Comprehensive test suite
├── 📁 data/                    # Dataset storage (structure only)
├── 📁 notebooks/               # Research & analysis notebooks  
├── 📁 scripts/                 # Training & evaluation scripts
├── ⚙️ app.py                   # FastAPI production server
├── 📁 run_complete_pipeline.py # End-to-end training pipeline
└── 📋 requirements.txt         # Python dependencies
```

### 🔧 Development Setup

```bash
# 1. Clone repository
git clone https://github.com/Jafri115/dynamic_fraud_detection.git
cd dynamic_fraud_detection

# 2. Setup development environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Optional: development tools

# 4. Setup pre-commit hooks (optional)
pre-commit install

# 5. Run tests to verify setup
pytest tests/ -v
```

## 📚 Research

### 🎓 Academic Context

This work was conducted as part of a **Master's Thesis** at the **University of Hildesheim**, Germany, focusing on advanced machine learning approaches for fraud detection in dynamic environments.

### 📄 Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{murtaza2024seqtab,
  title={Dynamic Fraud Detection: Machine Learning Approaches for Identifying Anomalous Activities},
  author={Murtaza, Wasif},
  year={2024},
  school={University of Hildesheim},
  type={Master's Thesis},
  url={https://github.com/Jafri115/dynamic_fraud_detection}
}
```

### 🔬 Research Contributions

1. **Novel Architecture**: First framework to effectively combine sequential and tabular data for fraud detection
2. **Time-Aware Processing**: Advanced attention mechanisms for temporal pattern recognition
3. **Adversarial Training**: OCAN adaptation for handling severe class imbalance
4. **Real-world Validation**: Comprehensive evaluation on Wikipedia vandalism detection
5. **Production Deployment**: Complete end-to-end system with REST API

### 📊 Related Work

- **OCAN**: One-Class Adversarial Networks for anomaly detection
- **Attention Mechanisms**: Time-aware processing for sequential data
- **Multi-modal Learning**: Fusion of heterogeneous data types
- **Fraud Detection**: Machine learning approaches for financial crime

---

## 🤝 Contributing

We welcome contributions from the research and developer community!

### 🌟 Ways to Contribute

- **🐛 Bug Reports**: Found an issue? Open a GitHub issue
- **💡 Feature Requests**: Have ideas? Share them in discussions  
- **📝 Documentation**: Improve docs, add examples, write tutorials
- **🧪 Testing**: Add test cases, improve coverage
- **🔬 Research**: Extend the model, try new datasets
- **💻 Code**: Fix bugs, optimize performance, add features

### 📞 Get in Touch

- **Issues**: [GitHub Issues](https://github.com/Jafri115/dynamic_fraud_detection/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Jafri115/dynamic_fraud_detection/discussions)
- **Email**: swasifmurtaza@gmail.com

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### 🙏 Acknowledgments

- **University of Hildesheim** - Research supervision and academic support
- **UMD** - Providing the Wikipedia vandalism dataset  
- **Open Source Community** - Tools, libraries, and frameworks used
- **Research Community** - Prior work in fraud detection and anomaly detection

---

## 👨‍💻 Author

<div align="center">

**Wasif Murtaza**  
*Master's Student in Data Science*

[![Email](https://img.shields.io/badge/Email-swasifmurtaza@gmail.com-red?style=for-the-badge&logo=gmail)](mailto:swasifmurtaza@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-@Jafri115-black?style=for-the-badge&logo=github)](https://github.com/Jafri115)
[![University](https://img.shields.io/badge/University-Hildesheim-purple?style=for-the-badge)](https://www.uni-hildesheim.de/)
[![Demo](https://img.shields.io/badge/🤗%20Demo-Hugging%20Face-yellow?style=for-the-badge)](https://huggingface.co/spaces/Wasifjafri/wiki-fraud-detection)

---


</div>
