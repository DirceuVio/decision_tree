# Decision Tree Implementation in Python

This repository contains a Python implementation of a Decision Tree Classifier and a Decision Tree Regressor. Decision Trees are versatile and widely used machine learning algorithms for both classification and regression tasks.

## Purpose
I created this project with the intention of gaining an in-depth understanding of how Decision Trees work and how to implement them from scratch. It serves as a learning exercise and a practical example of Decision Tree implementation.

## Usage

You can use the provided Decision Tree Classifier and Regressor classes in your projects. Here's how to get started:

### Decision Tree Classifier

```python
from decision_tree import DecisionTreeClassifier

# Load your dataset and split it into features (X) and labels (y)
# X_train, y_train, X_test, y_test = ...

# Create a DecisionTreeClassifier instance
clf = DecisionTreeClassifier(X_train, y_train)

# Fit the classifier to the training data
clf.fit()

# Make predictions on new data
predictions = clf.predict(X_test)
```

### Decision Tree Regressor
```python
from decision_tree import DecisionTreeRegressor

# Load your dataset and split it into features (X) and target values (y)
# X_train, y_train, X_test, y_test = ...

# Create a DecisionTreeRegressor instance
regressor = DecisionTreeRegressor(X_train, y_train)

# Fit the regressor to the training data
regressor.fit()

# Make predictions on new data
predictions = regressor.predict(X_test)
```