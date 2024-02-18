from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA, PCA
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest



# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1,parser='auto')
X, y = mnist.data.astype('float32'), mnist.target.astype('int')

# Keep only digits 0 and 8
indices = np.logical_or(y == 0, y == 8)
X = X[indices]
y = y[indices]


# Outlier detection
clf = IsolationForest(contamination=0.05)  # Puedes ajustar el nivel de contaminación según tus necesidades
outliers = clf.fit_predict(X)

# Eliminate Outliers
X_clean = X[outliers != -1]
y_clean = y[outliers != -1]



#scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

# quantity of singular values to be considered
n_componentes = 2


# PCA KErnel
#Not used because the accuracy was highly affected
#kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
#X_kpca = kpca.fit_transform(X_scaled)

# create an object from PCA
mnist_pca = PCA(n_components=n_componentes)
# fit the data
mnist_pca.fit(X_scaled)
# transform the data using the PCA object
X_transformed = mnist_pca.transform(X_scaled)




# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_clean, test_size=0.2, random_state=42)
print('inicio')
# Train a naive logistic regression model
model = LogisticRegression(solver='liblinear', max_iter=100)
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('\n PCA 4 features')
print(f"Accuracy: {accuracy:.4f}")
report = classification_report(y_test, y_pred)
print('\nClassification Report:\n', report)
