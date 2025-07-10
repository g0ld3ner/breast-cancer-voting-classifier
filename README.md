# Breast Cancer Voting Classifier

This repository contains a single Jupyter notebook `Brustkrebs_advanced.ipynb`.
It demonstrates how to train an ensemble on the UCI Breast Cancer dataset from
`scikit-learn`. The objective is to reach high accuracy while minimizing
**false negatives** so that malignant tumors are not missed.

## Notebook Overview

- Load the dataset and split it into training and test sets
- Invert the target labels so that `1` represents malignant tumors
- Hyperparameter search using `GridSearchCV` for:
  - Support Vector Classifier (SVC)
  - RandomForestClassifier
  - MLPClassifier
  with an Fβ score (β = 50) that heavily penalizes false negatives
- Apply `SMOTE` to balance the classes
- Combine the best models in a soft `VotingClassifier`
- Evaluate the ensemble with classification reports and confusion matrices

The resulting ensemble achieves roughly 96% accuracy on the test data while
prioritizing recall to avoid overlooking cancer cases.

## Running the Notebook

1. Install the required libraries:
   ```bash
   pip install scikit-learn imbalanced-learn pandas numpy matplotlib seaborn
   ```
2. Launch the notebook and run the cells in order:
   ```bash
   jupyter notebook Brustkrebs_advanced.ipynb
   ```

The notebook walks through all steps from data preparation to evaluating the
Voting Classifier.
