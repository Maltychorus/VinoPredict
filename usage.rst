==========
Code Block 
==========

Data Analysis Code-Block for Wine Quality Analysis Dataset
==========================================================

.. code-block:: python

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Load dataset
    wine_data = pd.read_csv('WineQT.csv')
    wine_data.head()

    # Data Inspection
    wine_data.shape
    wine_data.isnull().sum()
    wine_data.describe()

    # Data Visualization - Count Plot
    sns.countplot(x='quality', data=wine_data)

    # Correlation Heatmap
    correlation = wine_data.corr()
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(correlation, cmap='YlGnBu', annot=True)

    # Categorizing Wine Quality
    wine_data = wine_data.replace({'quality': {
        8: 'High',
        7: 'High',
        6: 'Middle',
        5: 'Middle',
        4: 'Low',
        3: 'Low',
    }})

    # Updated Count Plot
    sns.countplot(x='quality', data=wine_data)

    # Pie Chart for Quality Distribution
    data = [945, 159, 39]
    quality = ['Middle', 'High', 'Low']
    fig = plt.figure(figsize=(10, 10))
    plt.pie(data, labels=quality)

    # Citric Acid vs Quality
    sns.boxplot(x='quality', y='citric acid', data=wine_data)

    # Scatterplot - Alcohol Content vs Volatile Acidity
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='alcohol', y='volatile acidity', hue='quality', 
                    data=wine_data, palette='coolwarm', s=100)

    # Set Plot Labels and Title
    plt.title('Alcohol Content vs Volatile Acidity by Quality Group', fontsize=14)
    plt.xlabel('Alcohol Content', fontsize=12)
    plt.ylabel('Volatile Acidity', fontsize=12)
    plt.show()

    # Machine Learning Models
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = wine_data.drop(columns=['quality', 'Id'])
    y = wine_data['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    from sklearn.svm import SVC
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score, classification_report
    prediction = svm_model.predict(X_test)
    accuracy_svm = accuracy_score(y_test, prediction)

    from sklearn.linear_model import LogisticRegression
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train)

    prediction = logistic_model.predict(X_test)
    accuracy_logistic = accuracy_score(y_test, prediction)

    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    prediction = rf_model.predict(X_test)
    accuracy_rf = accuracy_score(y_test, prediction)

    from sklearn.tree import DecisionTreeClassifier
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)

    prediction = dt_model.predict(X_test)
    accuracy_dt = accuracy_score(y_test, prediction)

    # Model Accuracy Comparison
    model_names = ['Logistic Regression', 'SVM', 'Random Forest', 'Decision Tree']
    accuracies = [accuracy_logistic, accuracy_svm, accuracy_rf, accuracy_dt]

    plt.figure(figsize=(12, 7))
    bars = plt.bar(model_names, accuracies, color=['blue', 'orange', 'green', 'red'], edgecolor='black', alpha=0.7)

    # Adding text annotations for accuracy values on top of each bar
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Comparison of Model Accuracies', fontsize=16, fontweight='bold')

    plt.ylim(0.75, 0.9)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.show()



