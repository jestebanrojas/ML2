from PIL import Image, ImageOps 

image = Image.open(r"foto/juan_Rojas.jpg") 

new_image = image.resize((256, 256))
new_image = ImageOps.grayscale(new_image) 
new_image.save('foto/Juan_Rojas1.jpg')

