# Home Credit Default Risk Prediction

## Overview

This project aims to predict whether a client will default on their loan using various machine learning models. The project involves data preprocessing, exploration, feature selection, model building, and evaluation.

## Project Structure

```
.
├── Datasets
│   ├── train_df_filtered.csv
│   ├── test_df_filtered.csv
│   ├── app_train.csv
│   └── app_test.csv
├── Plots
│   ├── correlation_of_columns.png
│   ├── features_importances_m2.png
│   ├── features_importances_XGB.png
│   ├── features_importances_RF.png
│   ├── features_importances_HGB.png
│   ├── classification_report.png
│   ├── confusion_matrix_percent.png
│   ├── ROC.png
│   ├── ROC_xgb.png
│   ├── classification_report_RF.png
│   ├── confusion_matrix_RF.png
│   ├── ROC_RF.png
│   ├── classification_report_HGB.png
│   ├── confusion_matrix_HGB.png
│   └── ROC_HGB.png
├── models
│   └── HGB_model.pkl
└── README.md
```

## Dependencies

The project requires the following Python libraries:

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report
import joblib
from yellowbrick.model_selection import FeatureImportances
from yellowbrick.classifier import ClassificationReport, ConfusionMatrix
```

## Conclusion

The project highlights the challenges of predicting loan defaults with moderate accuracy. Future efforts should focus on improving model performance through advanced techniques and better data handling.