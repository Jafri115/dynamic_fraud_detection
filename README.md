# 🔒 Dynamic Fraud Detection## 📋 Table of Contents

- [Live Demo](#-live-demo)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Performance](#-performance)
- [API Usage](#-api-usage)
- [Contributing](#-contributing)
- [License](#-license)
- [Citation](#-citation)n](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Research](https://img.shields.io/badge/Research-Fraud%20Detection-red.svg)](https://github.com/Jafri115/dynamic_fraud_detection)
[![University](https://img.shields.io/badge/University-Hildesheim-purple.svg)](https://www.uni-hildesheim.de/)
[![License](https://img.shields.io/badge/License-Academic-yellow.svg)](LICENSE)
[![Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Demo-yellow)](https://huggingface.co/spaces/Wasifjafri/wiki-fraud-detection)

> Advanced fraud detection framework combining sequential and tabular data using deep learning and adversarial networks.

**SeqTab-OCAN** is a state-of-the-art machine learning framework designed to detect fraudulent activities by leveraging both sequential user behavior patterns and tabular features. This project was developed as part of a master's thesis in collaboration with the University of Hildesheim.

## 🎮 Live Demo

🚀 **Try it now**: [**Interactive Demo on Hugging Face**](https://huggingface.co/spaces/Wasifjafri/wiki-fraud-detection)

Experience SeqTab-OCAN in action! Upload user data and get real-time fraud detection predictions through our interactive web interface.

## 🌟 Key Features

- **🧠 Hybrid Architecture**: Combines sequential and tabular data for comprehensive fraud detection
- **⏰ Time-Aware Processing**: Captures temporal relationships in user behavior sequences  
- **⚖️ Imbalanced Data Handling**: One-Class Adversarial Networks (OCAN) with GANs for outlier detection
- **📊 Superior Performance**: Outperforms baseline models across multiple metrics
- **🚀 Production Ready**: FastAPI-based REST API for real-time predictions
- **🔧 Modular Design**: Easy to extend and customize for different use cases

## � Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Performance](#-performance)
- [API Usage](#-api-usage)
- [Contributing](#-contributing)
- [License](#-license)
- [Citation](#-citation)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/Jafri115/dynamic_fraud_detection.git
cd dynamic_fraud_detection

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## ⚡ Quick Start

### Training the Model

```bash
# Run the complete training pipeline
python run_complete_pipeline.py
```

### Starting the API Server

```bash
# Launch the FastAPI server
python app.py

# Server will be available at http://localhost:8000
# API documentation at http://localhost:8000/docs
```

### Running Tests

```bash
# Execute test suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 🏗️ Architecture

SeqTab-OCAN combines multiple components for effective fraud detection:

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
## 📊 Dataset

### UMDWikipedia Dataset
Our experiments utilize the UMDWikipedia dataset, adapted for fraud detection research:

- **📈 Scale**: 33,000+ users, 770,000 edits
- **🎯 Distribution**: 20% vandalism, 80% benign behavior  
- **🔧 Features**: 9 engineered behavioral features
- **💡 Innovation**: Novel adaptation of Wikipedia data for fraud detection

### Engineered Features

| Feature                    | Description                                                                  |
|----------------------------|------------------------------------------------------------------------------|
| **Total Edits**             | Total number of edits made by a user.                                        |
| **Unique Pages**            | Number of unique pages edited by a user.                                     |
| **Edit Frequency**          | Average time in seconds between consecutive edits by a user.                 |
| **Night Edits**             | Number of edits made by a user during nighttime (18:00 - 06:00).             |
| **Day Edits**               | Number of edits made by a user during daytime (06:00 - 18:00).               |
| **Weekend Edits**           | Number of edits made by a user during weekends.                              |

These engineered features help capture behavioral patterns that are useful for detecting fraudulent or abnormal activities, drawing from both sequential and tabular aspects of the dataset.

## 🎯 Performance

### Benchmark Results

SeqTab-OCAN achieves state-of-the-art performance on the UMDWikipedia dataset:

| Model | Data Type | Precision | Recall | F1-Score | AUC-PR | AUC-ROC |
|-------|-----------|-----------|--------|----------|--------|---------|
| OCAN Baseline | Sequential | 0.9117±0.007 | 0.9097±0.008 | 0.9107±0.003 | 0.8838±0.004 | 0.971±0.003 |
| Tab-RL | Tabular | 0.9042±0.002 | 0.7996±0.004 | 0.8487±0.002 | 0.9240±0.001 | 0.9079±0.003 |
| Seq-RL | Sequential | 0.9470±0.018 | 0.9026±0.015 | 0.9241±0.003 | 0.9718±0.011 | 0.9754±0.005 |
| SeqTab-RL | Seq + Tab | 0.9529±0.018 | 0.8931±0.032 | 0.9225±0.011 | 0.9735±0.001 | 0.9732±0.002 |
| **SeqTab-OCAN** | **Seq + Tab** | **0.9307±0.0003** | **0.9487±0.0002** | **0.9396±0.0001** | **0.9817±0.0002** | **0.9379±0.0002** |

### Key Insights

- ✅ **Best F1-Score**: 0.9396 with minimal variance (±0.0001)
- ✅ **Highest AUC-PR**: 0.9817, indicating excellent precision-recall trade-off
- ✅ **Balanced Performance**: Strong performance across all evaluation metrics
- ✅ **Robust Results**: Extremely low standard deviation demonstrates model stability

## 🔌 API Usage

### Starting the Server

```bash
python app.py
```

### Making Predictions

```python
import requests

# Example prediction request
data = {
    "user_features": {
        "total_edits": 150,
        "unique_pages": 45,
        "reverted_ratio": 0.12,
        # ... other features
    },
    "sequence_data": [
        {"timestamp": "2024-01-01T10:00:00", "action": "edit", "page_id": 123},
        # ... sequence data
    ]
}

response = requests.post("http://localhost:8000/predict", json=data)
prediction = response.json()
```

### API Endpoints

- `POST /predict` - Make fraud predictions
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation


## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd dynamic_fraud_detection

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Train the complete pipeline
python run_complete_pipeline.py

# Start API server
python app.py

# Run tests
pytest tests/
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ --cov=src

# Code formatting
black src/ tests/
flake8 src/ tests/
```

## 📁 Project Structure

```
dynamic_fraud_detection/
├── 📂 src/                     # Source code
│   ├── 📂 components/          # Data processing components
│   ├── 📂 models/              # Model definitions
│   ├── 📂 pipeline/            # Training and prediction pipelines
│   ├── 📂 training/            # Training utilities
│   └── 📂 utils/               # Helper functions
├── 📂 tests/                   # Test suite
├── 📂 data/                    # Data directory (structure only)
├── 📂 notebooks/               # Jupyter notebooks for analysis
├── 📂 scripts/                 # Training and evaluation scripts
├── 🐍 app.py                   # FastAPI application
├── 🐍 run_complete_pipeline.py # Complete training pipeline
├── 📋 requirements.txt         # Production dependencies
└── 📄 README.md               # This file
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{murtaza2024seqtab,
  title={Dynamic Fraud Detection: Machine Learning Approaches for Identifying Anomalous Activities},
  author={Murtaza, Wasif},
  year={2024},
  school={University of Hildesheim},
  type={Master's Thesis}
}
```

## 👨‍💻 Author

**Wasif Murtaza**  
📧 Email: swasifmurtaza@gmail.com  
🎓 University of Hildesheim  
🔗 GitHub: [@Jafri115](https://github.com/Jafri115)

## 🙏 Acknowledgments

- University of Hildesheim for project supervision
- UMD for providing the Wikipedia vandalism dataset
- Open-source community for tools and libraries used

---

<div align="center">
  <sub>Built with ❤️ for fraud detection research</sub>
</div>
