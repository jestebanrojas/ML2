#----------------------------------------------------------------------#
#----------------------------------------------------------------------#
## 5. Train a naive logistic regression on raw MNIST images to distinguish between 0s and 8s. We are calling 
# this our baseline. What can you tell about the baseline performance?
#----------------------------------------------------------------------#
#----------------------------------------------------------------------#


import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1,parser='auto')


X, y = mnist.data.astype('float32'), mnist.target.astype('int')

# Keep only digits 0 and 8
indices = np.logical_or(y == 0, y == 8)
X = X[indices]
y = y[indices]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a naive logistic regression model
model = LogisticRegression(solver='liblinear', max_iter=100)
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
report = classification_report(y_test, y_pred)
print('\nClassification Report:\n', report)