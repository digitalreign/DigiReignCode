# Section CNN - MNIST exercise
# Author Jose Smith
# Date: 20210106

# Importing the relevant packages
print('=============Importing the relevant packages===================================')
import tensorflow as tf
import tensorflow_datasets as tfds
# using datetime module 
import datetime; 
  
# programstart stores current time 
programstart = datetime.datetime.now() 

## Downloading and preprocessing the data

# Before continuing with our model and training, our first job is to preprocess the dataset
# This is a very important step in all of machine learning

# The MNIST dataset is, in general, highly processed already - after all its 28x28 grayscale images of clearly visible digits
# Thus, our preprocessing will be limited to scaling the pixel values, shuffling the data and creating a validation set

# NOTE: When finally deploying a model in practice, it might be a good idea to include the prerpocessing as initial layers
# In that way, the users could just plug the data (images) directly, instead of being required to resize/rescale it before

# Defining some constants/hyperparameters
BUFFER_SIZE = 70_000 # for reshuffling
BATCH_SIZE = 128
NUM_EPOCHS = 20

# Downloading the MNIST dataset
print('=============Downloading the MNIST dataset=====================================')

# When 'with_info' is set to True, tfds.load() returns two variables: 
# - the dataset (including the train and test sets) 
# - meta info regarding the dataset itself

mnist_dataset, mnist_info = tfds.load(name='mnist', with_info=True, as_supervised=True)

# Extracting the train and test datasets
print('=============Extracting the train and test datasets============================')
mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']

# Creating a function to scale our image data (it is recommended to scale the pixel values in the range [0,1] )
print('=============Creating a function to scale our image data=======================')
def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.

    return image, label

# Scaling the data
print('=============Scaling the data==================================================')
train_and_validation_data = mnist_train.map(scale)
test_data = mnist_test.map(scale)

# Defining the size of the validation set
print('=============Defining the size of the validation set===========================')
num_validation_samples = 0.1 * mnist_info.splits['train'].num_examples
num_validation_samples = tf.cast(num_validation_samples, tf.int64)

# Defining the size of the test set
print('=============Defining the size of the test det=================================')
num_test_samples = mnist_info.splits['test'].num_examples
num_test_samples = tf.cast(num_test_samples, tf.int64)

# Reshuffling the dataset
print('=============Reshuffling the dataset===========================================')
train_and_validation_data = train_and_validation_data.shuffle(BUFFER_SIZE)

# Splitting the dataset into training + validation
print('=============Splitting the dataset into training and validation================')
train_data = train_and_validation_data.skip(num_validation_samples)
validation_data = train_and_validation_data.take(num_validation_samples)

# Batching the data
# NOTE: For proper functioning of the model, we need to create one big batch for the validation and test sets
print('=============Batching the data=================================================')
train_data = train_data.batch(BATCH_SIZE)
validation_data = validation_data.batch(num_validation_samples) 
test_data = test_data.batch(num_test_samples)

## Creating the model and training it
print('===============================================================================')
print('=============Creating the model and training it================================')
print('===============================================================================')
# Now that we have preprocessed the dataset, we can define our CNN and train it

# Outlining the model/architecture of our CNN

# CONV -> MAXPOOL -> CONV -> MAXPOOL -> FLATTEN -> DENSE
print('=============Outlining the model of our CNN====================================')
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(50, 5, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)), 
    # (2,2) is the default pool size so we could have just used MaxPooling2D() with no explicit arguments
    tf.keras.layers.Conv2D(50, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)), 
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10) # You can apply softmax activation here, see below for comentary
])

# A brief summary of the model and parameters
print('=============A brief summary of the model and parameters=======================')
model.summary(line_length = 75)

''' Model: "sequential"
___________________________________________________________________________
Layer (type)                     Output Shape                  Param #     
===========================================================================
conv2d (Conv2D)                  (None, 24, 24, 50)            1300        
___________________________________________________________________________
max_pooling2d (MaxPooling2D)     (None, 12, 12, 50)            0           
___________________________________________________________________________
conv2d_1 (Conv2D)                (None, 10, 10, 50)            22550       
___________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)   (None, 5, 5, 50)              0           
___________________________________________________________________________
flatten (Flatten)                (None, 1250)                  0           
___________________________________________________________________________
dense (Dense)                    (None, 10)                    12510       
===========================================================================
Total params: 36,360
Trainable params: 36,360
Non-trainable params: 0
___________________________________________________________________________ '''




# programend stores current time 
programend = datetime.datetime.now()
print("Program Started:-", programstart) 
print("Program Ended:-", programend) 
