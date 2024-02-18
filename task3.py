#----------------------------------------------------------------------#
#----------------------------------------------------------------------#
## 3. Let’s create the unsupervised Python package
#----------------------------------------------------------------------#
#----------------------------------------------------------------------#

    #----------------------------------------------------------------------#
    # --- Implement SVD from scratch using Python and NumP
    #----------------------------------------------------------------------#



# the code with the class SVD is allocated on ../unsupervised/svd
import numpy as np
from unsupervised.dim_red import svd


# Example matrix
columns=4
rows=3
A = np.random.randint(10, size=(rows, columns))

# quantity of singular values to be considered
n_componentes = 3

# create an object from svd
svd_1 = svd.SVD(n_components=n_componentes)

# Apply the svd transformation
A_transformada = svd_1.fit_transform(A)
# test: build the original matrix
A_reconstruida = svd_1.inverse_transform(A_transformada)

print("Matriz original:")
print(A)
print("\nMatriz transformada:")
print(A_transformada)
print("\nMatriz reconstruida:")
print(A_reconstruida)
print("\nVt:")
print(svd_1.components_)

    #----------------------------------------------------------------------#
    # --- Implement PCA from scratch using Python and NumPy
    #----------------------------------------------------------------------#
# the code with the class pca is allocated on ../unsupervised/pca



from unsupervised.dim_red import pca




# create a PCA object with n components
pca = pca.PCA(n_components=n_componentes)

# fit the data
pca.fit(A)
# transform the data using the PCA object
X_transformed = pca.transform(A)


# re-construct the data
X_reconstructed = np.dot(X_transformed, pca.components_) + pca.mean_


print("\nX_transformed:")
print(X_transformed)
print("\nX_reconstructed:")
print(X_reconstructed)



    #----------------------------------------------------------------------#
    # --- Implement t-SNE from scratch using Python and NumPy
    #----------------------------------------------------------------------#

"""
from unsupervised.t_sne import t_sne

from sklearn.manifold import TSNE

#A_transformed = t_sne.t_sne(A, n_dimensions=2, perplexity=2)



A_transformed = t_sne.t_sne(A, n_dimensions=2, n_iterations=250, perplexity=1)

print("\nA_transformed:")
print(A_transformed)
"""


import logging

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

from unsupervised.dim_red import t_sne

logging.basicConfig(level=logging.DEBUG)

X, y = make_classification(
    n_samples=500, n_features=10, n_informative=5, n_redundant=0, random_state=1111, n_classes=2, class_sep=2.5
)

p = t_sne.TSNE(2, max_iter=500)
X = p.fit_transform(X)

colors = ["red", "green"]
for t in range(2):
    t_mask = (y == t).astype(bool)
    plt.scatter(X[t_mask, 0], X[t_mask, 1], color=colors[t])

plt.show()