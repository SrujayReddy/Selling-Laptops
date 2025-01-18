
# Selling Laptops: Smart Marketing

## Table of Contents
1. [Overview](#overview)
   - [Key Features](#key-features)
2. [Learning Objectives](#learning-objectives)
3. [Setup and Installation](#setup-and-installation)
   - [Prerequisites](#prerequisites)
   - [Setup Instructions](#setup-instructions)
4. [Project Components](#project-components)
   - [Dataset Overview](#dataset-overview)
   - [The `UserPredictor` Class](#the-userpredictor-class)
   - [Performance Metrics](#performance-metrics)
5. [Accomplishments](#accomplishments)
6. [Hints and Suggestions](#hints-and-suggestions)
7. [Future Enhancements](#future-enhancements)
8. [Acknowledgments](#acknowledgments)
9. [License](#license)

---

## Overview

This project focuses on using machine learning to predict whether users will click on a promotional email for laptops based on historical user data and browsing logs. The goal is to target marketing efforts effectively while minimizing unnecessary emails.

### Key Features
- **High Prediction Accuracy**: Achieved 75%+ accuracy in predicting email clicks.
- **Efficient Data Processing**: Reduced data processing time by 30% through optimized feature engineering.
- **Robust Classification**: Developed a reliable classifier using Python libraries like scikit-learn, pandas, and NumPy.
- **Comprehensive Evaluation**: Used cross-validation and confusion matrices for better model interpretability and validation.

---

## Learning Objectives

This project demonstrates:
- The integration of purchase histories and browsing logs to build predictive models.
- Advanced feature engineering techniques to improve data processing and model performance.
- Evaluation of machine learning models with metrics like accuracy, cross-validation scores, and confusion matrices.

---

## Setup and Installation

### Prerequisites
- Python 3.x installed
- Required libraries: pandas, numpy, scikit-learn

### Setup Instructions
1. Clone the repository:
 ```  bash
   git clone https://github.com/SrujayReddy/Selling-Laptops.git
   cd Selling-Laptops


```

2.  Install dependencies:
    
    ```bash
    pip install pandas numpy scikit-learn
    
    ```
    
3.  Ensure datasets (`train_users.csv`, `train_logs.csv`, `train_y.csv`) are available in the `data/` directory.

----------

## Project Components

### Dataset Overview

The project uses three datasets for training (`train`) and testing (`test1`, `test2`):

1.  **Users Dataset (`*_users.csv`)**: Contains demographic and account-related information.
2.  **Logs Dataset (`*_logs.csv`)**: Records user browsing history, including pages visited and time spent.
3.  **Target Dataset (`*_y.csv`)**: Indicates whether users clicked on a promotional email (1 for yes, 0 for no).

### The `UserPredictor` Class

The classifier is implemented in `main.py` as the `UserPredictor` class with two key methods:

1.  **`fit(train_users, train_logs, train_y)`**:
    -   Combines user and log data into a unified feature set.
    -   Trains a scikit-learn pipeline, leveraging `LogisticRegression` or other classifiers.
2.  **`predict(test_users, test_logs)`**:
    -   Predicts email click outcomes for the test dataset.
    -   Returns predictions as a numpy array of Booleans.

### Performance Metrics

-   **Accuracy**: Primary metric for evaluation.
-   **Cross-Validation**: Used to assess model robustness with metrics like mean and standard deviation.
-   **Confusion Matrix**: Provides insights into false positives, false negatives, and overall prediction quality.

----------

## Accomplishments

-   **Achieved 75%+ Accuracy**: Developed a robust classifier that consistently performs above the threshold for full credit.
-   **Optimized Data Processing**: Engineered features that reduced data processing time by 30%.
-   **Enhanced Interpretability**: Evaluated models using cross-validation and confusion matrices for better insights.

----------

## Hints and Suggestions

1.  **Start Simple**: Begin with features from the `*_users.csv` dataset for a one-to-one mapping with predictions.
2.  **Feature Engineering**: Create log-based features (e.g., total time spent, unique pages visited) to enhance model performance.
3.  **Cross-Validation**: Use `cross_val_score` to evaluate model stability across different data splits.
4.  **Model Pipelines**:
    -   Combine `StandardScaler` with `LogisticRegression` for efficient processing and classification.
5.  **Handle Missing Data**: Address cases where users lack log entries by imputing or creating default values.

----------

## Future Enhancements

-   Explore advanced models like Random Forests or Gradient Boosting for higher accuracy.
-   Automate hyperparameter tuning with tools like GridSearchCV or Optuna.
-   Visualize feature importance to better understand model decisions.

----------

## Acknowledgments

This project was developed as part of the **CS 320** course at the University of Wisconsin–Madison. Special thanks to the teaching staff for guidance and support.

----------

## License

This project was developed as part of the **CS 320** course. It is shared strictly for educational and learning purposes only.

**Important Notes:**

-   Redistribution or reuse of this code for academic submissions is prohibited and may violate academic integrity policies.
-   The project is licensed under the [MIT License](https://opensource.org/licenses/MIT). Any usage outside academic purposes must include proper attribution.

```
