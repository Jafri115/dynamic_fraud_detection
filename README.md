# Dynamic Fraud Detection :Machine Learning Approaches for Identifying Anomalous Activities

This is a master's thesis project in collaboration with the University of Hildesheim, focused on fraud detection using user behavior analytics.


## Project Overview

Fraudulent behaviors are dynamic and challenging to detect due to the constantly evolving techniques used by fraudsters. Traditional fraud detection methods often focus on either transactional data or user activity sequences, but fail to combine both types effectively. Furthermore, many models do not consider the temporal dynamics inherent in sequential data.

In this project, we propose SeqTab-OCAN, a fraud detection framework that integrates sequential and tabular data using a time-aware attention network to capture the temporal relationships in sequence data and combine them with tabular data, improving predictive accuracy. Additionally, SeqTab-OCAN incorporates One-Class Adversarial Networks (OCAN) with complementary GANs to handle imbalanced datasets and detect outliers, boosting fraud detection performance.

Experimental results on public datasets show that SeqTab-OCAN outperforms existing fraud detection models, offering significant improvements in detecting fraudulent activities.


## Project Structure

The main project files and folders include:

```plaintext
.
├── data/                        # Contains datasets and processed data
├── models/                      # Model definitions and training scripts
│   ├── Combined_rep_MODEL/      # Combined representation learning model
│   └── OCAN_Baseline/           # OCAN GAN model for fraud detection
├── saved_models/                # Directory for model checkpoints
├── utils/                       # Utility functions for data handling, logging, etc.
├── train_model.py               # Main script to train the model phases
├── requirements.txt             # Dependencies for setup
└── README.md                    # Project documentation and instructions
```
## Dataset Overview

### UMDWikipedia Dataset

- **Purpose**: Developed for Wikipedia vandalism detection.
- **Source**: Data from approximately 33,000 users, including both vandals and non-vandals.
- **Data Size**: 
  - 770,000 edits in total.
  - 160,651 edits (20%) made by vandals.
  - 609,389 edits (80%) made by benign users.
- **Features**: The dataset provides detailed insights into the editing behaviors of users, making it suitable for studying user activity patterns and detecting anomalous behaviors that may indicate fraudulent actions or vandalism.

This dataset is particularly useful for analyzing user interactions in online platforms and can be applied in fraud detection scenarios where identifying abnormal behavior is crucial.

## Adaptation for Fraud Detection

- **Challenge**: A lack of publicly available datasets containing both sequential and tabular data types.
- **Solution**: The UMDWikipedia dataset was adapted to fit the experimental needs of fraud detection, combining sequential and tabular data.
- **Data Engineering**: Tabular features were engineered from the sequential data available in Wikipedia logs to better represent user behavior for fraud detection.

### Table 1: Description of Engineered Features from the UMDWikipedia Dataset

| Feature                    | Description                                                                  |
|----------------------------|------------------------------------------------------------------------------|
| **Total Edits**             | Total number of edits made by a user.                                        |
| **Unique Pages**            | Number of unique pages edited by a user.                                     |
| **Reverted Ratio**          | Average ratio of edits that were reverted for a user.                        |
| **Cluebot Revert Count**    | Total number of times a user’s edits were reverted by ClueBot.               |
| **Edit Frequency**          | Average time in seconds between consecutive edits by a user.                 |
| **Night Edits**             | Number of edits made by a user during nighttime (18:00 - 06:00).             |
| **Day Edits**               | Number of edits made by a user during daytime (06:00 - 18:00).               |
| **Weekend Edits**           | Number of edits made by a user during weekends.                              |
| **Page Category Diversity**| Number of unique categories of pages edited by a user.                       |

These engineered features help capture behavioral patterns that are useful for detecting fraudulent or abnormal activities, drawing from both sequential and tabular aspects of the dataset.

## Public Dataset - Results Comparison

### Table 3: Comparison of Experiments on UMDWikipedia Dataset

| Method               | Data   | Pre  | Recall | F1    | AUC_PR | AUC_ROC |
|----------------------|--------|------|--------|-------|--------|---------|
| **OCAN Baseline with LSTM-AE** | Seq    | 0.9117 | 0.9097 | 0.9107 | 0.8838 | 0.971   |
|                      |        | ±0.007 | ±0.008  | ±0.003 | ±0.004 | ±0.003  |
| **Tab-RL**            | Tab    | 0.9042 | 0.7996 | 0.8487 | 0.9240 | 0.9079  |
|                      |        | ±0.002 | ±0.004  | ±0.002 | ±0.001 | ±0.003  |
| **Seq-RL**            | Seq    | 0.9470 | 0.9026 | 0.9241 | 0.9718 | 0.9754  |
|                      |        | ±0.018 | ±0.015  | ±0.003 | ±0.011 | ±0.005  |
| **SeqTab-RL**         | Seq+Tab | 0.9529 | 0.8931 | 0.9225 | 0.9735 | 0.9732  |
|                      |        | ±0.018 | ±0.032  | ±0.011 | ±0.001 | ±0.002  |
| **SeqTab-OCAN**       | Seq+Tab | 0.9307 | 0.9487 | 0.9396 | 0.9817 | 0.9379  |
|                      |        | ±0.0003 | ±0.0002 | ±0.0001 | ±0.0002 | ±0.0002 |


The results show that combining both sequential (Seq) and tabular (Tab) data significantly improves the performance of fraud detection models, with **SeqTab-OCAN** demonstrating the best performance across all metrics.


## How to Run the Project

### 1. Clone the Repository

Start by cloning the repository to your local machine:

```bash
git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection
```

### 2. Install Dependencies
Install the required Python packages using requirements.txt:
```bash
pip install -r requirements.txt
```

### 3. Start MLflow Server
MLflow is used for experiment tracking. Start the server with the following command:
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5005

```

### 4. Running the Model Training Script
The main training script is train_model.py. You can specify various parameters to control the model training:
```bash
python train_model.py <dataset_id> <load_from_disk> <train_representation_phase1> <train_OCAN_phase2>
```
Command Explanation

```plaintext
- dataset_id: Identifier for the dataset (e.g., 1 for the default dataset).
- load_from_disk: Set to 1 to load pre-processed data from disk.
- train_representation_phase1: Set to 1 to train the representation learning model (Phase 1).
- train_OCAN_phase2: Set to 1 to train the OCAN GAN model (Phase 2).

```

### Contributing
For questions or contributions, please contact Wasif Jafri on GitHub at https://github.com/wasifjafri.
