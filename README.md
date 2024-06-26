# creditfraud

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Project Description
This project focuses on accurately identifying fraudulent transactions within a dataset of credit card transactions. The dataset features are anonymized and scaled, preserving privacy while still allowing for effective analysis. Our main objectives include understanding the dataset's distribution, creating a balanced subset using the NearMiss algorithm, evaluating various classifiers for their accuracy, developing a neural network model for comparison, and addressing common issues in dealing with imbalanced datasets.

Feature Details:
PCA Transformation: The dataset documentation indicates that all features, except for time and amount, have undergone a Principal Component Analysis (PCA) transformation, a technique used for reducing dimensionality.

Scaling: It is important to note that before performing PCA, features need to be scaled. In this dataset, it is assumed that all 'V' features have been scaled appropriately by the dataset creators.

Goals:
Data Distribution Analysis: Examine the dataset to understand the distribution of fraud and non-fraud transactions.
Data Balancing: Create a balanced dataset using the NearMiss algorithm to ensure a 50/50 ratio of fraud to non-fraud transactions.
Classifier Evaluation: Implement and evaluate various classifiers to determine which provides the highest accuracy.
Neural Network Development: Build a neural network and compare its performance with the best-performing classifier.
Handling Imbalanced Data: Gain insights into common mistakes and best practices when dealing with imbalanced datasets.
Steps to Achieve These Goals:
Data Analysis and Visualization: Perform exploratory data analysis (EDA) to visualize and understand the dataset.
Data Preprocessing: Clean and preprocess the data, including handling missing values, scaling features, and applying the PCA transformation if necessary.
Model Training: Train multiple machine learning models, including logistic regression, decision trees, random forests, and support vector machines.
Model Evaluation: Evaluate the models using appropriate metrics such as accuracy, precision, recall, and F1-score.
Neural Network Implementation: Develop and train a neural network model, then compare its performance to traditional classifiers.
Results Interpretation: Analyze and interpret the results, highlighting the strengths and weaknesses of each model in detecting fraudulent transactions.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for frauddetection
│                         and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── frauddetection                <- Source code for use in this project.
    │
    ├── __init__.py    <- Makes frauddetection a Python module
    │
    ├── data           <- Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── features       <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    │
    ├── models         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py
```

## Setup Instructions

To get started with this project, follow the steps below to set up your development environment using Conda.

### 1. Clone the Repository

```sh
git clone https://github.com/your-username/creditfraud.git
cd creditfraud
```

### 2. Create & Activate Conda Environment

```sh
conda env create -f environment.yml
conda activate creditfraud
```



--------

