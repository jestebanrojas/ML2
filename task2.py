#----------------------------------------------------------------------#
#----------------------------------------------------------------------#
## 2. Add a steady, well-centered picture of your face to a shared folder alongside your classmates
#----------------------------------------------------------------------#
#----------------------------------------------------------------------#


    #----------------------------------------------------------------------#
    # --- edit your picture to be 256x256 pixels, grayscale (single channel)
    #----------------------------------------------------------------------#

from PIL import Image, ImageOps 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os, numpy as np, PIL


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
