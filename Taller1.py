#Taller No. 1
#Machine Learning 2
#Por Juan Esteban Rojas

#----------------------------------------------------------------------#
#----------------------------------------------------------------------#
## 1. Simulate any random rectangular matrix A 
#----------------------------------------------------------------------#
#----------------------------------------------------------------------#


#Library Import

import numpy as np
import pandas as pd

#Script to create a random values Rectangular Matrix.
columns=4
rows=3
A = np.random.randint(10, size=(rows, columns))

print("Next is displayed the random rectangular Matrix:\n")
print(A)

    #----------------------------------------------------------------------#
    # What is the rank and trace of A?
    #----------------------------------------------------------------------#

#Rank
print("\n\n The rank of the A Matrix is:\n")
print(np.linalg.matrix_rank(A))

#Trace
print("\n\n The rank of the A Matrix is:\n")
print(np.trace(A))


    #----------------------------------------------------------------------#
    #What is the determinant of A?
    #----------------------------------------------------------------------#
try:
    detA=np.linalg.det(A)
    print("\n\n The determinant of A Matrix is:\n")
    print(detA)
except Exception as err:
    print("\n\n Oops!  It was not able to calculate the matrix determinant, it is not a square matrix")

    #----------------------------------------------------------------------#
    # Can you invert A? How?
    #----------------------------------------------------------------------#

# Calculating the inverse of the matrix
try:
    
    invA=np.linalg.inv(A)
    print("\n\n The inverse of A Matrix is:\n")
    print(invA)
except Exception as err:
    print("\n\n Oops!  it is not a square matrix, so there is not possible to calculate tje inverse matrix, but you can calculate the pseudo inverse")
    psinvA=np.linalg.pinv(A)
    print("\n\n The pseudoinverse of A Matrix is:\n")
    print(psinvA)

    #----------------------------------------------------------------------#
    # --- How are eigenvalues and eigenvectors of A’A and AA’ related? What interesting differences can you notice between both?
    #----------------------------------------------------------------------#

#Calculating A Transpose

ATransp=A.transpose()
print("A Transpose: \n",ATransp)
print("Original A Matrix: \n",A)

AAT=np.dot(A,ATransp)
ATA=np.dot(ATransp,A)

print("AA': \n",np.dot(A,ATransp))
print("A'A: \n",np.dot(ATransp,A))


#Lets call U as the eingenvectors matrix of AA' and V the eigenvectors of A'A 


eigenvalues_AAT, U = np.linalg.eigh(np.dot(A,ATransp))
print("eigenvalues AA':",eigenvalues_AAT)
print("eigenvectors:\n",U)

eigenvalues_ATA, V = np.linalg.eigh(np.dot(ATransp,A))
print("eigenvalues A'A:",eigenvalues_ATA)
print("eigenvectors:\n",V)

"""
As we can see, the eigenvalues are the same for A'A and AA'.A
Now, we can probe that A=U.S.V'

"""

# order U and V matrix
index=np.argsort(-eigenvalues_AAT)
print("index",index)
eigenvalues_AAT = eigenvalues_AAT[index]
U=U[:,index]


index=np.argsort(-eigenvalues_ATA)
print("index",index)
eigenvalues_ATA = eigenvalues_ATA[index]
V=V[:,index]

#Build a diagonal matrix based on the eigenvalues
S = np.zeros((rows,columns))
S[:,:-1] = np.sqrt(np.diag(eigenvalues_AAT))
S[np.isnan(S)] = 0

#Aply the operation A=U.S.V'
A3=np.dot(np.dot(U,S),V.transpose())
print("A3:",A3)
#So we can see that A'A and AA' are related in order tha we can reconstruct the original A matrix 
# through the A'A, AA' eigenvectors and the eigenvalues.

print("U,V,S",U,V.transpose(),S)

U, S1, Vh = np.linalg.svd(A, full_matrices=True)


S = np.zeros((3,4))
S[:,:-1] = np.diag(S1)
S[np.isnan(S)] = 0

A2=np.dot(np.dot(U,S),Vh)

print("A2: \n",A2)
print("U,V,S",U,Vh,S)


#----------------------------------------------------------------------#
#----------------------------------------------------------------------#
## 2. Add a steady, well-centered picture of your face to a shared folder alongside your classmates
#----------------------------------------------------------------------#
#----------------------------------------------------------------------#


    #----------------------------------------------------------------------#
    # --- edit your picture to be 256x256 pixels, grayscale (single channel)
    #----------------------------------------------------------------------#

from PIL import Image, ImageOps 

#Load the image
image = Image.open(r"foto/juan_Rojas.jpg") 
#Image resizing and greyscale convertion
my_image = image.resize((256, 256))
my_image = ImageOps.grayscale(my_image) 
#Save the knew Image
my_image.save('foto/Juan_Rojas1.jpg')

    #----------------------------------------------------------------------#
    # --- Plot your edited face
    #----------------------------------------------------------------------#

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#Create a subplot to allocate the pictures
fig, ax = plt.subplot_mosaic([
    ['JuaRojas', 'Promedio', 'Diferencia']
], figsize=(7, 3.5))


#The image will be plotted later
ax['JuaRojas'].imshow(my_image, cmap='gray')
#plt.show()

    #----------------------------------------------------------------------#
    # ---  Calculate and plot the average face of the cohort
    #----------------------------------------------------------------------#

import os, numpy as np, PIL
from PIL import Image, ImageOps

# pictures path
directorio_imagenes = 'foto/fotos/'


# Access all jpg files in the directory
allfiles=os.listdir(directorio_imagenes)
imlist=[filename for filename in allfiles if  filename[-4:] in [".jpg",".JPG"]]

# We will store all images in following array
imagenes = []

# append all images on the imagenes array, each image is resized to a 256 x 256 size image and converted into a grayscale format.
for im in imlist:
    imagen = Image.open(directorio_imagenes+im).convert('L')
    imagen = imagen.resize((256, 256))
    imagen = ImageOps.grayscale(imagen)
    # Convertir la imagen a una matriz de NumPy
    imagenes.append(np.array(imagen))  

 
# Obtain the first image size
alto, ancho = imagenes[0].shape
suma_imagenes = np.zeros((alto, ancho), dtype=np.uint64)

#pixels sum from each image
for imagen in imagenes:
    suma_imagenes += imagen

# average calculation
avg_image = suma_imagenes // len(imagenes)


# We create a new image with the average array
average_image = Image.fromarray(avg_image.astype(np.uint8), 'L')  # converting it as an image
average_image.save(directorio_imagenes+"promedio_imagen.jpg")
#imagen_promedio.show()

#Load the average image into the plot.
ax['Promedio'].imshow(average_image, cmap='gray')

#show the plot
#plt.show()


    #----------------------------------------------------------------------#
    # --- How distant is your face from the average? How would you measure it
    #----------------------------------------------------------------------#

#To know how distant is my face from the average, we will use an euclidean distance, so 
# the first step is to substract my face image from the average image, and then calculate 
# the euclidean distance 

# images difference
difference = average_image-np.array(my_image)

# euclidean distance
euclidean_distance = np.linalg.norm(difference)

print(f'Distancia euclidiana entre las dos imágenes: {euclidean_distance}')

ax['Diferencia'].imshow(difference, cmap='gray')


plt.show()

#----------------------------------------------------------------------#
#----------------------------------------------------------------------#
## 3. Let’s create the unsupervised Python package
#----------------------------------------------------------------------#
#----------------------------------------------------------------------#

    #----------------------------------------------------------------------#
    # --- Implement SVD from scratch using Python and NumP
    #----------------------------------------------------------------------#

# the code with the class SVD is allocated on ../unsupervised/svd

from unsupervised.svd import svd


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


import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, load_iris
from unsupervised.pca import pca






# create a PCA object with 2 components
pca = pca.PCA(n_components=50)

# fit the data
pca.fit(my_image)

# transform the data using the PCA object
X_transformed = pca.transform(my_image)

Image.fromarray(X_transformed.astype(np.uint8), 'L').show()




    #----------------------------------------------------------------------#
    # --- Implement t-SNE from scratch using Python and NumPy
    #----------------------------------------------------------------------#


#----------------------------------------------------------------------#
#----------------------------------------------------------------------#
## 4. Apply SVD over the picture of your face, progressively increasing the number of singular values used. 
# Is there any point where you can say the image is appropriately reproduced? How would you quantify how 
# different your photo and the approximation are?
#----------------------------------------------------------------------#
#----------------------------------------------------------------------#

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


image = Image.open(r"foto/juan_Rojas.jpg") 
#Image resizing and greyscale convertion
my_image = image.resize((256, 256))
my_image = ImageOps.grayscale(my_image) 


n=255
# Crear subgráficas
fig, axes = plt.subplots(2, 5, figsize=(8, 15))

# Iterar sobre las subgráficas y trazar los datos correspondientes
for i, ax in enumerate(axes.flat):
    n_componentes = 5*(i+2)
    image_svd=svd.SVD(n_components=n_componentes)
    # Apply the svd transformation
    Image_reduced = image_svd.fit_transform(my_image)
    Reconst = image_svd.inverse_transform(Image_reduced)

    ax.imshow(Reconst, cmap='gray')  
    ax.set_title(n_componentes)
    ax.axis('off')  # Opcional: desactivar ejes para mejorar la presentación


# Ajustar el diseño de la figura
plt.tight_layout()
plt.show()

n_componentes = 25
image_svd=svd.SVD(n_components=n_componentes)
# Apply the svd transformation
Image_reduced = image_svd.fit_transform(my_image)
Reconst = image_svd.inverse_transform(Image_reduced)
Image_reconst = Image.fromarray(Reconst.astype(np.uint8), 'L')  # Convertir de nuevo a imagen



# images difference
difference = my_image-Reconst

# euclidean distance
euclidean_distance = np.linalg.norm(difference)

print(f'Distancia euclidiana entre las dos imágenes: {euclidean_distance}')

Image.fromarray(difference.astype(np.uint8), 'L').show()

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

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1,parser='auto')
print("mnist")

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
print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualize some predictions
num_samples = 5
selected_indices = np.random.choice(len(X_test), num_samples, replace=False)

for i, index in enumerate(selected_indices):
    plt.subplot(1, num_samples, i + 1)
    plt.imshow(X_test[index].reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {y_pred[index]} / True: {y_test[index]}")
    plt.axis('off')

plt.show()
