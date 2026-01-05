# Real-Time Network Intrusion Detection with Incremental Learning

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.9.6-blue)

## Introduction

This repository contains the source code for the Practical Project of the **Aprendizagem Aplicada à Segurança (AAS)** course.

The objective of this project is to **detect and classify network intrusions in real-time** using **machine learning**. The system processes streaming network traffic data, continuously adapting to new patterns through online learning while maintaining high detection performance.

The project implements a dual-model architecture:

- **Binary Classification**: SGD (Stochastic Gradient Descent) classifier for real-time attack/benign detection with incremental learning via `partial_fit()`
- **Multi-Class Classification**: Random Forest classifier for specific attack type identification

**Authors:**

- André Oliveira (107637)
- Alexandre Cotorobai (107849)

## Prerequisites

- Python 3.9.6
- Pip 21.2.4

## Installation

It is strongly recommended to run this project inside a virtual environment to avoid dependency conflicts.

### 1. Clone Repository

```bash
git clone https://github.com/andreaoliveira9/Projeto-AAS.git
```

### 2. Go to Project Directory

```bash
cd Projeto-AAS
```

### 3. Create and Activate Virtual Environment

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## Dataset

This project utilizes the **NF-CICIDS2018-v3** dataset, a comprehensive network intrusion detection dataset containing NetFlow features extracted from network traffic captures.
    
  - **Source:** [NF-CICIDS2018-v3 Dataset](https://staff.itee.uq.edu.au/marius/NIDS_datasets/)
  - **Setup:** Place the dataset files in the `data/` folder. If the dataset is large, run the provided script to download it:
    ```bash
      python scripts/download_data.py
    ```

  After download, the dataset is preprocessed and split into training/test/evaluation
  sets using the provided notebooks. The preprocessing step also generates reusable artifacts (encoders, scalers, imputers) for real-time data transformation.

- **Preprocessed Splits:** (stored in `data/preprocessed/`)
  - `train_data.csv`: 14,511 samples (60% benign, 40% attacks) for initial training
  - `test_data.csv`: 104,484 samples (preprocessed features) for streaming simulation
  - `test_raw.csv`: 130,605 samples (raw features) for real-time preprocessing testing
  - `eval_data.csv`: 26,121 samples for model validation during streaming
  
- **Preprocessing Artifacts** (stored in `data/preprocessing_artifacts/`):
  - `encoders.pkl`: Categorical feature encoders
  - `scaler.pkl`: Feature scaling transformations
  - `imputer.pkl`: Missing value imputation strategies
  - `feature_config.pkl`: Feature configuration and metadata
## Usage

### 1. Data Preprocessing and Analysis

After downloading the dataset, run the data analysis notebook to preprocess it and prepare the training/test/evaluation splits:

```bash
jupyter notebook notebooks/data_analysis.ipynb
```

This notebook will:

- Load and analyze the raw dataset
- Remove data leakage features (timing-based features not available in real-time)
- Balance the dataset (60% benign, 40% attacks)
- Feature engineering
- Create train/test/evaluation splits
- Generate preprocessed CSV files in `data/processed/`
- Generate preprocessing artifacts in `data/preprocessing_artifacts/` (encoders, scalers, imputers, feature config)

### 2. Model Training

Train both the binary and multi-class classifiers:

```bash
jupyter notebook notebooks/model_training.ipynb
```

This notebook will:

- Train 8 different binary classifiers (selects SGD for incremental learning)
- Train 5 multi-class classifiers (selects Random Forest for attack type detection)
- Save trained models to `notebooks/models` directory
- Generate performance visualizations

**Note:** Copy the trained model files to the `models/` directory for use in the detection system.

### 3. Real-Time Detection System

Run the intrusion detection system with streaming packet simulation and cleared logs:

```bash
python ./src/run_detection_system.py --clear-logs
```

**Parameters:**

- `--model`: Path to binary classification model (default: `./models/sgd_model_binary.pkl`)
- `--test`: Test dataset for streaming simulation (default: `./data/processed/test_data.csv`)
  - Use `test_data.csv` for preprocessed data (faster)
  - Use `test_raw.csv` for raw data with real-time preprocessing
- `--eval`: Evaluation dataset for performance monitoring (default: `./data/processed/eval_data.csv`)
- `--preprocessing-artifacts`: Path to preprocessing artifacts directory (enables real-time preprocessing)
- `--rate`: Packets per second streaming rate (default: 100)
- `--retrain-batch`: Number of samples before retraining consideration (default: 1000)
- `--eval-interval`: Number of packets between performance evaluations (default: 500)
- `--output`: Directory for logs and metrics (default: `logs`)
- `--clear-logs`: Clear previous logs before starting
- `--attack-classifier`: Path to multi-class attack classifier (optional)
- `--no-preprocessing`: Disable real-time preprocessing even if artifacts are available

### 4. Visualization Dashboard

Launch the Streamlit dashboard to visualize detection metrics in real-time or analyze past sessions:

```bash
streamlit run ./src/dashboard/streamlit_dashboard.py
```

**Note:** The dashboard reads logs from `src/logs/` directory. You can:

- Run it **during system execution** for real-time monitoring
- Run it **after execution** for offline analysis of completed sessions

## Major Results

### Binary Classification (Attack Detection)

#### Initial Evaluation (Before Online Learning)

The following table summarizes the **initial performance** of our SGD classifier on the evaluation set (26,121 samples):

| Class        | Precision | Recall | F1-Score | Support |
| :----------- | :-------: | :----: | :------: | :-----: |
| **Benign**   |   0.81    |  0.94  |   0.87   | 15,673  |
| **Attack**   |   0.88    |  0.67  |   0.76   | 10,448  |
| **Accuracy** |           |        | **0.83** | 26,121  |
| **MCC**      |           |        | **0.64** |         |

#### After Online Learning (39,579 packets, 32 retraining cycles)

After continuous adaptation through incremental learning, the model achieved significantly improved performance:

| Metric         |  Value  |
| :------------- | :-----: |
| **Accuracy**   | **0.89** |
| **MCC**        | **0.78** |
| **Precision**  |  0.81   |
| **Recall**     |  0.94   |
| **F1-Score**   |  0.87   |

**Key Improvements:**
- Accuracy increased from 0.83 to 0.89 (+6%)
- MCC improved from 0.64 to 0.78 (+14%)
- Recall improved from 0.67 to 0.94 (+27%), significantly reducing false negatives
- The model successfully adapted to new attack patterns through 32 retraining cycles

### Multi-Class Classification (Attack Type Detection)

The following table shows the top 3 models evaluated for attack type classification:

| Model                 | Accuracy | Precision | Recall | F1-Score |  MCC   |
| :-------------------- | :------: | :-------: | :----: | :------: | :----: |
| **Random Forest**     |  0.9856  |  0.9849   | 0.9856 |  0.9851  | 0.9760 |
| **Gradient Boosting** |  0.9823  |  0.9831   | 0.9823 |  0.9826  | 0.9706 |
| **Decision Tree**     |  0.9729  |  0.9768   | 0.9729 |  0.9744  | 0.9554 |

**Selected Model:** Random Forest (98.56% accuracy, 0.976 MCC)

### Per-Attack Type Performance (Random Forest)

| Attack Type              | Precision | Recall | F1-Score | Samples |
| :----------------------- | :-------: | :----: | :------: | :-----: |
| Benign                   |   0.99    |  0.99  |   0.99   | 15,673  |
| DDOS_attack-HOIC         |   1.00    |  1.00  |   1.00   |  4,483  |
| FTP-BruteForce           |   1.00    |  1.00  |   1.00   |  1,680  |
| DDoS_attacks-LOIC-HTTP   |   1.00    |  1.00  |   1.00   |  1,200  |
| SSH-Bruteforce           |   1.00    |  1.00  |   1.00   |   844   |
| Bot                      |   0.93    |  0.99  |   0.96   |   946   |
| Infilteration            |   0.83    |  0.71  |   0.77   |   861   |
| DoS_attacks-SlowHTTPTest |   0.85    |  0.91  |   0.88   |   434   |

### Key Findings

- The **shadow model architecture prevents model degradation** during incremental learning, with automated validation before model promotion
- **Low False Negative Rate** for critical attacks (DDoS, Brute Force) is maintained through strict MCC thresholds
- The system successfully handles **concept drift** through continuous monitoring and adaptive retraining
- **Attack type classification provides actionable intelligence** for security analysts, enabling targeted response strategies

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
