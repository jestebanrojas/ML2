#----------------------------------------------------------------------#
#----------------------------------------------------------------------#
## 4. Apply SVD over the picture of your face, progressively increasing the number of singular values used. 
# Is there any point where you can say the image is appropriately reproduced? How would you quantify how 
# different your photo and the approximation are?
#----------------------------------------------------------------------#
#----------------------------------------------------------------------#

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageOps 
import numpy as np
from unsupervised.dim_red import svd


image = Image.open(r"foto/juan_Rojas.jpg") 
#Image resizing and greyscale convertion
my_image = image.resize((256, 256))
my_image = ImageOps.grayscale(my_image) 


n=255
# subplots creation
fig, axes = plt.subplots(2, 5, figsize=(8, 15))

#  iterate over the subplots to plot each image
for i, ax in enumerate(axes.flat):
    n_componentes = 5*(i+2)
    image_svd=svd.SVD(n_components=n_componentes)
    # Apply the svd transformation
    Image_reduced = image_svd.fit_transform(my_image)
    Reconst = image_svd.inverse_transform(Image_reduced)

    ax.imshow(Reconst, cmap='gray')  
    ax.set_title(n_componentes)
    ax.axis('off') 


# image design adjustment
plt.tight_layout()
plt.show()

n_componentes = 25
image_svd=svd.SVD(n_components=n_componentes)
# Apply the svd transformation
Image_reduced = image_svd.fit_transform(my_image)
Reconst = image_svd.inverse_transform(Image_reduced)
Image_reconst = Image.fromarray(Reconst.astype(np.uint8), 'L')  # convert the array into an image



# images difference
difference = my_image-Reconst

# euclidean distance
euclidean_distance = np.linalg.norm(difference)

print(f'Distancia euclidiana entre las dos imágenes: {euclidean_distance}')

Image.fromarray(difference.astype(np.uint8), 'L').show()

# With 25 singular values we can appropriately reproduce the image.