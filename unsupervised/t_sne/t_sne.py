#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import numpy as np
#from sklearn.datasets import fetch_openml


#import numpy as np
#import matplotlib.pyplot as plt






def calculate_P(X, perplexity=30.0, tol=1e-5, max_iter=1000):
    # X: Conjunto de datos en el espacio de alta dimensión
    # perplexity: Hiperparámetro de t-SNE

    n, _ = X.shape
    print("\n x.shape:")
    print(n)
    P = np.zeros((n, n))

    # Buscar la mejor sigma para cada punto
    for i in range(n):

        print("\n iteracion 1 calculate p:")
        print(i)
        sigma_low, sigma_high = 0.0, np.inf
        target_perplexity = np.log2(perplexity)


        z=0
        # Búsqueda binaria para encontrar sigma que alcanza la perplexity objetivo
        for _ in range(max_iter):
           # print("\n iteracion 2 calculate p:")
           # z=z+1
            #print(z)

            sigma = (sigma_low + sigma_high) / 2.0
            distances = np.sum(X[i] ** 2) - 2 * np.dot(X[i], X.T) + np.sum(X ** 2, axis=1)
            P[i] = np.exp(-distances / (2.0 * sigma ** 2))
            P[i, i] = 0.0
            sum_Pi = np.sum(P[i])
            entropy = -np.sum(P[i] / sum_Pi * np.log2(P[i] / sum_Pi + 1e-12))

            if np.abs(entropy - target_perplexity) < tol:
                break

            if entropy > target_perplexity:
                sigma_high = sigma
            else:
                sigma_low = sigma

    # Normalizar las probabilidades condicionales
    P = (P + P.T) / (2.0 * n)
    P = np.maximum(P, 1e-12)  # Evitar divisiones por cero

    return P


def calculate_Q(Y):
    # Y: Ubicaciones de los puntos en el espacio de baja dimensión

    n, _ = Y.shape
    Q = np.zeros((n, n))

    # Calcula las distancias euclidianas entre puntos en el espacio de baja dimensión
    distances = np.sum(Y ** 2, axis=1, keepdims=True) - 2 * np.dot(Y, Y.T) + np.sum(Y ** 2, axis=1, keepdims=True).T

    # Calcula las probabilidades de similitud conjunta Qij utilizando la función t-distribution
    Q = 1 / (1 + distances)
    np.fill_diagonal(Q, 0.0)  # Asegura que Qii sea cero

    # Normaliza las probabilidades conjuntas
    Q = Q / np.sum(Q)
    
    return Q

def gradient(P, Q, Y):
    # P: Probabilidades de similitud condicional
    # Q: Probabilidades de similitud conjunta
    # Y: Ubicaciones de los puntos en el espacio de baja dimensión

    n, _ = Y.shape
    grad = np.zeros_like(Y)

    for i in range(n):

        print("\n iteracion gradient:")
        print(i)

        # Calcula la diferencia de ubicaciones de los puntos
        diff = Y[i] - Y
        # Calcula la diferencia de probabilidades
        PQ_diff = np.expand_dims((P[i] - Q[i]) * Q[i], axis=-1)
        # Suma ponderada de las diferencias
        grad[i] = 4 * np.sum(PQ_diff * diff, axis=0)

    return grad

def tsne(X, num_dimensions=30, learning_rate=50.0, perplexity=30.0, num_iterations=250):
    # Inicializa las ubicaciones de los puntos en el espacio de baja dimensión
    Y = np.random.randn(X.shape[0], num_dimensions)

    for iteration in range(num_iterations):
        print("\n iteracion tsne:")
        print(iteration)

        # Calcula las probabilidades de similitud condicional (Pij)
        P = calculate_P(X, perplexity)

        # Calcula las probabilidades de similitud conjunta (Qij)
        Q = calculate_Q(Y)

        # Calcula los gradientes y ajusta las posiciones de los puntos
        grad = gradient(P, Q, Y)
        Y -= learning_rate * grad

    return Y
"""
"""
import numpy as np

def compute_pairwise_distances(X):
    # Compute pairwise Euclidean distances
    distances = np.sum(X**2, axis=1, keepdims=True) - 2 * np.dot(X, X.T) + np.sum(X**2, axis=1, keepdims=True).T
    np.fill_diagonal(distances, 0)  # Set diagonal elements to zero
    return distances

def calculate_P(Y, perplexity, epsilon=1e-10):
    # Compute pairwise distances
    distances = compute_pairwise_distances(Y)

    # Adjust perplexity
    beta = 1.0 / perplexity

    # Initialize conditional probabilities matrix
    P = np.zeros((Y.shape[0], Y.shape[0]))

    # Compute conditional probabilities for each point
    for i in range(Y.shape[0]):
        print("\n iteracion 1 calculate p:")
        print(i)
        # Compute conditional probabilities for point i
        exp_distances = np.exp(-beta * (distances[i] - np.max(distances[i]))) + epsilon  # Add epsilon to prevent overflow
        P[i] = exp_distances / np.sum(exp_distances)



    # Set diagonal elements to zero
    np.fill_diagonal(P, 0)

    # Symmetrize P
    P = (P + P.T) / (2.0 * Y.shape[0])

    return P


def t_sne(X, n_dimensions=2, perplexity=30.0, learning_rate=200.0, n_iterations=1000, epsilon=1e-10):
    # Initialize low-dimensional embedding randomly
    Y = np.random.randn(X.shape[0], n_dimensions)

    # Perform t-SNE optimization
    for iteration in range(n_iterations):
        print("\n iteracion 1 tsne")
        print(iteration)


        # Compute conditional probabilities
        P = calculate_P(Y, perplexity, epsilon=epsilon)


        # Check for NaN values in P
        if np.isnan(P).any():
            print("NaN values detected in P. Stopping optimization.")
            break

        # Compute gradient
        grad = 4 * np.dot((P - P.T) * (1 - compute_pairwise_distances(Y)), Y)


        # Update embedding
        Y -= learning_rate * grad

        # Print cost every 50 iterations
        if (iteration + 1) % 50 == 0:
            epsilon = 1e-12  # Pequeña cantidad para evitar división por cero
            cost = np.sum(P * np.log((P + epsilon) / (compute_pairwise_distances(Y) + epsilon)))
            print(f"Cost: {cost}")



    return Y


