# Dry Bean Classification — Machine Learning Assignment 2

## a. Problem Statement
The objective is to build and compare five supervised classification algorithms that predict the variety of a dry bean using numerical measurements extracted from bean images.

## b. Dataset Description
- Dataset: Dry Bean
- Source: UCI Machine Learning Repository
- UCI Dataset ID: 602
- Instances: 13,611
- Input features: 16 numerical measurements
- Target: `Class`
- Classes: Barbunya, Bombay, Cali, Dermason, Horoz, Seker and Sira
- Missing values: None reported by UCI

Source: https://archive.ics.uci.edu/dataset/602/dry

## c. GitHub Repository Link
Add your public repository URL here after creating the repository.

## d. Models Used and Evaluation

The following five models are trained on the same stratified training split:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine

Metrics use weighted Precision, Recall and F1 for multiclass evaluation. AUC is weighted one-vs-rest ROC AUC. MCC is the multiclass Matthews Correlation Coefficient.

### Model Comparison Table
After running `python train.py`, copy the values from `results/metrics.csv` into this table.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Support Vector Machine | | | | | | |

### Observations

The five classification algorithms were evaluated on the same test dataset containing 2,723 instances.

1. Support Vector Machine: Achieved the highest accuracy of 92.21%, AUC of 0.9937 and F1 score of 0.9222. It demonstrated the best overall performance among the five models.
2. Logistic Regression: Achieved 92.07% accuracy and 0.9934 AUC, performing very close to SVM.
3. K-Nearest Neighbors: Achieved 91.66% accuracy and 0.9812 AUC, showing competitive classification performance.
4. Gaussian Naive Bayes: Achieved 89.79% accuracy and 0.9902 AUC. Its high AUC indicates strong class-ranking performance despite lower accuracy.
5. Decision Tree: Achieved 89.17% accuracy and 0.9330 AUC, the lowest performance among the five models.

### Overall Winner

Support Vector Machine was the best-performing model based on the test results, achieving the highest accuracy, AUC, precision, recall, F1 score and MCC. Therefore, SVM was selected as the strongest classifier for this experiment.


## Application
The Streamlit app accepts a labeled test CSV, lets the evaluator choose one of the five saved models, and displays Accuracy, AUC, Precision, Recall, F1, MCC, a confusion matrix and a classification report. The app loads trained models and does not retrain on uploaded test data.

Live Streamlit App Link: Add after deployment.

## Reproducibility
Install dependencies with `pip install -r requirements.txt`.
Run training with `python train.py`.
Launch the app with `streamlit run app.py`.

## Files
- `train.py`: downloads data, trains models, evaluates them and saves artifacts.
- `app.py`: interactive Streamlit evaluation interface.
- `test_data.csv`: held-out test set with features and true target.
- `models/`: saved classifiers and metadata.
- `results/metrics.csv`: evaluation comparison table.
