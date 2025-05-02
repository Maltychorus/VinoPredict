=============================================================
Wine Quality Analysis 
=============================================================



.. py:module:: wine_analysis
   :synopsis: Analysis and Machine Learning for Wine Quality Prediction

This module provides functions for **data loading, visualization, and machine learning models** to predict wine quality.

Wine Data Preview
=============================================================
The table below shows information about wine samples. Each sample is like a mini lab report for one glass of wine. Scientists use this kind of data to figure out how good a wine is based on what’s inside it.

Each row represents a single wine sample, and each column gives us details about its ingredients or characteristics. The dataset includes 1,143 wine samples, each described by 13 different features.

-------------------------------------------------------------
Column Descriptions
-------------------------------------------------------------

+------------------------+--------------------------------------------------------------+
| **Column Name**        | **What It Tells Us**                                         |
+========================+==============================================================+
| fixed acidity          | Natural acids that give wine its sour or tart taste.         |
+------------------------+--------------------------------------------------------------+
| volatile acidity       | If this is too high, the wine might smell like vinegar.      |
+------------------------+--------------------------------------------------------------+
| citric acid            | Adds freshness and lemony flavor.                            |
+------------------------+--------------------------------------------------------------+
| residual sugar         | Sugar left in wine — more sugar means sweeter taste.         |
+------------------------+--------------------------------------------------------------+
| chlorides              | Salt content in the wine.                                    |
+------------------------+--------------------------------------------------------------+
| free sulfur dioxide    | Helps keep wine fresh and free from bacteria.                |
+------------------------+--------------------------------------------------------------+
| total sulfur dioxide   | Total amount of preservative chemicals.                      |
+------------------------+--------------------------------------------------------------+
| density                | How thick or watery the wine is.                             |
+------------------------+--------------------------------------------------------------+
| pH                     | Acidity level — lower pH means more acidic.                  |
+------------------------+--------------------------------------------------------------+
| sulphates              | Acts as a preservative and adds a dry or sharp note.         |
+------------------------+--------------------------------------------------------------+
| alcohol                | Alcohol content in the wine.                                 |
+------------------------+--------------------------------------------------------------+
| quality                | A score from 0–10 showing how good the wine is.              |
+------------------------+--------------------------------------------------------------+
| Id                     | Just a tag number for each sample.                           |
+------------------------+--------------------------------------------------------------+


-------------------------------------------------------------
Quick Peak at the Data
-------------------------------------------------------------

.. image:: C:/Users/Gurseerat/docs/source/images/wine_data_preview.png
   :width: 1000px
   :align: center



-------------------------------------------------------------
Exploratory Visualizations
-------------------------------------------------------------

**1. Correlation Heatmap** -  
Shows how strongly each feature is related to wine quality and to each other. This helps in identifying redundant or useful variables.

.. image:: C:/Users/Gurseerat/docs/source/images/Correlation_Heatmap.png
   :width: 1000px
   :align: center


**2. Feature Importance (Random Forest)** -  
Ranks the features based on how much they influence the model's prediction of wine quality.

.. image:: C:/Users/Gurseerat/docs/source/images/feature_importance.png
   :width: 1000px
   :align: center


-------------------------------------------------------------
Model Evaluation
-------------------------------------------------------------

**1. Confusion Matrix** -  
Illustrates how well the model is performing across each predicted quality score.

.. image:: C:/Users/Gurseerat/docs/source/images/Confusion_Matrix.png
   :width: 800px
   :align: center

**2. ROC-AUC Curve** -  
Displays model performance at different classification thresholds. The area under the curve (AUC) indicates how well the model can distinguish between classes. A value of 1 means perfect classification, while 0.5 indicates random guessing.

.. image:: C:/Users/Gurseerat/docs/source/images/roc_auc_curve.png
   :width: 800px
   :align: center   

**3.Classification Report** - This section presents the classification report for the Random Forest model used in predicting wine quality. The report includes precision, recall, and F1-score for each quality level in the test dataset.

.. image:: C:/Users/Gurseerat/docs/source/images/classification_report_heatmap.png
   :width: 800px
   :align: center

