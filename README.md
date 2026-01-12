# Steam Purchase Prediction

This project predicts whether a user will purchase a specific game on Steam using user behavior data and game metadata.  
The task is formulated as a **binary classification problem**, combining user activity signals with game attributes to model purchase likelihood.

The project emphasizes **clean data preprocessing, feature engineering, and model evaluation**, and is structured as a reproducible machine learning pipeline rather than a notebook-only workflow.

---

## Problem Formulation

- **Task**: Predict whether a user will purchase a given Steam game
- **Type**: Binary classification
- **Labels**:
  - `1` → user purchases the game
  - `0` → user does not purchase the game

---

## Features

### Game Features
- Price
- Genre (multi-label, one-hot encoded)
- Sentiment score
- Release year / game age
- Metascore (when available)

### User Features
- Total playtime
- Average playtime
- Number of games owned
- User genre preference distribution
- Consumption level based on historical purchases

---

## Models

- **Logistic Regression**
  - Interpretable baseline
  - Fast to train
- **Random Forest**
  - Captures nonlinear interactions
  - Handles feature interactions automatically

---

## Evaluation Metrics

- **Accuracy: 0.90**
- **ROC AUC: 0.96**
  - Chosen due to class imbalance
  - Focuses on ranking buyers above non-buyers

---

## Dataset

This project uses the Steam dataset from UCSD:

https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data

Please download:
- **Version 1**: Review Data (6.7 MB)
- **Version 2**: Item Metadata (2.7 MB)

Place the downloaded files in:
`data/raw`
