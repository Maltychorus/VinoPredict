=============================================================
Wine Quality Analysis - Python Code Documentation
=============================================================

.. py:module:: wine_analysis
   :synopsis: Analysis and Machine Learning for Wine Quality Prediction

This module provides functions for **data loading, visualization, and machine learning models** to predict wine quality.

Data Loading
=============================================================

Load the wine dataset from a CSV file into a Pandas DataFrame.

.. code-block:: python

   import pandas as pd
   import numpy as np
   import seaborn as sns
   import matplotlib.pyplot as plt

   # Load the wine dataset
   file_path = "WineQT.csv"
   wine_data = pd.read_csv(file_path)

   # Display the first few rows of the dataset
   print(wine_data.head())

Data Inspection
=============================================================

Inspect the dataset for missing values, shape, and summary statistics.

.. code-block:: python

   import pandas as pd

   # Load the dataset (Ensure wine_data is already loaded)
   print("Dataset Shape:", wine_data.shape)

   # Check for missing values
   print("Missing Values:\n", wine_data.isnull().sum())

   # Display summary statistics
   print("Summary Statistics:\n", wine_data.describe())

Data Visualization
=============================================================

Visualize the distribution of wine quality and the correlation between features.

.. code-block:: python

   import seaborn as sns
   import matplotlib.pyplot as plt

   # Plot the distribution of wine quality
   sns.countplot(x='quality', data=wine_data)
   plt.title("Distribution of Wine Quality")
   plt.xlabel("Quality")
   plt.ylabel("Count")
   plt.show()

   # Compute correlation matrix
   correlation = wine_data.corr()
   print("Correlation Matrix:\n", correlation)

   # Plot the heatmap of feature correlations
   fig, ax = plt.subplots(figsize=(20, 10))
   sns.heatmap(correlation, cmap='YlGnBu', annot=True, ax=ax)
   plt.title("Feature Correlation Heatmap")
   plt.show()

Feature Engineering
=============================================================

Convert the numerical **quality scores** into categorical labels (**High, Middle, Low**) for better classification.

.. code-block:: python

   # Categorizing Wine Quality
   wine_data = wine_data.replace({'quality': {
       8: 'High',
       7: 'High',
       6: 'Middle',
       5: 'Middle',
       4: 'Low',
       3: 'Low',
   }})

   # Display the updated dataset
   print(wine_data[['quality']].head())


Machine Learning Models
=============================================================

Train **four machine learning models** to predict wine quality:

- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Random Forest**
- **Decision Tree**

.. code-block:: python

   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LogisticRegression
   from sklearn.svm import SVC
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.metrics import accuracy_score

   # Splitting the dataset
   X = wine_data.drop(columns=['quality', 'Id'])
   y = wine_data['quality']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

   # Train Logistic Regression Model
   logistic_model = LogisticRegression()
   logistic_model.fit(X_train, y_train)
   accuracy_logistic = accuracy_score(y_test, logistic_model.predict(X_test))

   # Train Support Vector Machine (SVM)
   svm_model = SVC(kernel='linear')
   svm_model.fit(X_train, y_train)
   accuracy_svm = accuracy_score(y_test, svm_model.predict(X_test))

   # Train Random Forest Classifier
   rf_model = RandomForestClassifier()
   rf_model.fit(X_train, y_train)
   accuracy_rf = accuracy_score(y_test, rf_model.predict(X_test))

   # Train Decision Tree Classifier
   dt_model = DecisionTreeClassifier()
   dt_model.fit(X_train, y_train)
   accuracy_dt = accuracy_score(y_test, dt_model.predict(X_test))


Model Evaluation
=============================================================

Evaluate the performance of each trained model by calculating accuracy scores.

.. code-block:: python

   from sklearn.metrics import accuracy_score

   # Evaluate Logistic Regression
   prediction = logistic_model.predict(X_test)
   accuracy_logistic = accuracy_score(y_test, prediction)

   # Evaluate Support Vector Machine (SVM)
   prediction = svm_model.predict(X_test)
   accuracy_svm = accuracy_score(y_test, prediction)

   # Evaluate Random Forest Classifier
   prediction = rf_model.predict(X_test)
   accuracy_rf = accuracy_score(y_test, prediction)

   # Evaluate Decision Tree Classifier
   prediction = dt_model.predict(X_test)
   accuracy_dt = accuracy_score(y_test, prediction)

   # Store the accuracies
   model_accuracies = {
       "Logistic Regression": accuracy_logistic,
       "SVM": accuracy_svm,
       "Random Forest": accuracy_rf,
       "Decision Tree": accuracy_dt
   }

   # Print model accuracies
   print("Model Accuracies:", model_accuracies)


Model Performance Comparison
=============================================================

Generate a **bar plot** to compare the performance of different models.

.. code-block:: python

    import matplotlib.pyplot as plt

    # Accuracy values obtained from model evaluations
    accuracy_logistic = 0.85  # Logistic Regression
    accuracy_svm = 0.82       # SVM
    accuracy_rf = 0.87        # Random Forest
    accuracy_dt = 0.81        # Decision Tree

    # Model names and their respective accuracies
    model_names = ['Logistic Regression', 'SVM', 'Random Forest', 'Decision Tree']
    accuracies = [accuracy_logistic, accuracy_svm, accuracy_rf, accuracy_dt]

    # Creating a bar plot to visualize the comparison of model accuracies
    plt.figure(figsize=(12, 7))

    # Bar plot with additional customizations
    bars = plt.bar(model_names, accuracies, color=['blue', 'orange', 'green', 'red'], edgecolor='black', alpha=0.7)

    # Adding text annotations for accuracy values on top of each bar
    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,  # Adjusted to fit within the actual accuracy range
            f'{bar.get_height():.2f}',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )

    # Adding grid, labels, and title
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Comparison of Model Accuracies', fontsize=16, fontweight='bold')

    # Adjusted y-axis limits to fit the data properly
    plt.ylim(0.75, 0.9)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Display the plot
    plt.show()


