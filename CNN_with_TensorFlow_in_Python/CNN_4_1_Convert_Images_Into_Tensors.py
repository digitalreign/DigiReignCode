# Section CNN - Convert Images Into Tensors
# Author Jose Smith
# Start Date: 20210119
# End Date:

# Importing the relevant packages
print('=============Importing the relevant packages================================')
import numpy as np
from PIL import Image
import datetime

# programstart stores current time 
programstart = datetime.datetime.now() 

# Define the name of the image file we will use
filename = "CNN_with_TensorFlow_in_Python/Converting_images_to_tensors/Test_image.jpg"

# Load the image as a Python variable
# We can manipulate this variable with the Pillow package
# Useful image functions:
# Image.crop(box=None) - Returns a rectangular region from the image.
# Image.filter(filter) - Filters the image using the given filter (kernel)
# Image.getbox() - Calculates the bounding box of the non-zero regions in the image.
# Image.rotate(angle, ...) - Returns a rotated copy of the image.
image = Image.open(filename)

# Displays the image contained in the variable through your default photo veiwer
print('=============Show the Test Image============================================')
image.show()

# Resize the image while keeping the aspect ration constant
# The 'image' variable is updated with the new image
image.thumbnail((90,120))
print('=============Show the Thumbnail=============================================')
image.show()

# Convert the variable into a NumPy array (tensor) 
image_array = np.asarray(image)
# Check the dimensions of the array
# The convention for image arrays is (height x width x channels)
# 'channels' refferes to the colors 
#   - 1 for grayscale (only 1 value for each pixel)
#   - 3 for color images (red, green and blue values/channels)
print('=============Show the Tensor Array==========================================')
print(np.shape(image_array))

# programend stores current time 
programend = datetime.datetime.now()
roundedstart = programstart - datetime.timedelta(microseconds=programstart.microsecond)
roundedend = programend - datetime.timedelta(microseconds=programend.microsecond)
print("Program Started:-", roundedstart) 
print("Program Ended:-", roundedend) 