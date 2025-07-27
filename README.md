# Breast Cancer Voting Classifier – ML ensemble for reliable breast cancer detection

This repository contains a single Jupyter notebook `breast_cancer_analytics.ipynb`.
It demonstrates how to train an ensemble on the UCI Breast Cancer dataset from
`scikit-learn`. The objective is to achieve high accuracy while **minimizing false negatives**
so that malignant tumors are not missed.


## Notebook Overview

- Load the dataset and split it into training and test sets
- Invert the target labels so that `1` represents malignant tumors
- Hyperparameter tuning using `GridSearchCV` for:
  - Support Vector Classifier (SVC)
  - RandomForestClassifier
  - MLPClassifier
- Combine the models into hard and soft `VotingClassifier` ensembles
- Additional ensemble approaches: 
  - Voting with per-model probability thresholds (hard and soft voting variants)
- Evaluate all models and ensembles using classification reports and confusion matrices


The final ensemble outperforms each of its individual base models in terms of accuracy and recall on the test data.


## Running the Notebook

1. Install the required libraries:
   ```bash
   pip install scikit-learn imbalanced-learn pandas numpy matplotlib seaborn
   ```
2. Launch the notebook and run the cells in order:
   ```bash
   jupyter notebook breast_cancer_analytics.ipynb
   ```


The notebook walks through all steps from data preparation to ensemble evaluation.


## Outlook

In future work, the selection of decision thresholds for each model and the ensemble will be automated
