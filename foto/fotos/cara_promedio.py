import os, numpy as np, PIL
from PIL import Image, ImageOps

# Ruta al directorio que contiene las imágenes
directorio_imagenes = 'foto/fotos/'


# Access all PNG files in directory
#allfiles=os.listdir(os.getcwd())
allfiles=os.listdir(directorio_imagenes)
imlist=[filename for filename in allfiles if  filename[-4:] in [".jpg",".JPG"]]

# Assuming all images are the same size, get dimensions of first image
w,h=Image.open(directorio_imagenes+ imlist[0]).size
N=len(imlist)

# Create a numpy array of floats to store the average (assume RGB images)
arr=np.zeros((h,w,3),float)


# Lista para almacenar las imágenes
imagenes = []

# Build up average pixel intensities, casting each image as an array of floats
for im in imlist:
    imagen = Image.open(directorio_imagenes+im).convert('L')
    imagen = imagen.resize((256, 256))
    imagen = ImageOps.grayscale(imagen) 



    imagenes.append(np.array(imagen))  # Convertir la imagen a una matriz de NumPy
    #print(imagen.shape)

# Obtener el tamaño de una imagen para inicializar el arreglo de suma
alto, ancho = imagenes[0].shape
suma_imagenes = np.zeros((alto, ancho), dtype=np.uint64)

# Sumar los valores de píxeles de todas las imágenes
for imagen in imagenes:
    suma_imagenes += imagen

# Calcular el promedio dividiendo por el número de imágenes
promedio_imagen = suma_imagenes // len(imagenes)


# Crear una nueva imagen a partir del promedio
imagen_promedio = Image.fromarray(promedio_imagen.astype(np.uint8), 'L')  # Convertir de nuevo a imagen
imagen_promedio.save(directorio_imagenes+"promedio_imagen.jpg")
imagen_promedio.show()