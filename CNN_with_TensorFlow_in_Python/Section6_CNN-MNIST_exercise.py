# Section CNN - MNIST exercise
# Author Jose Smith
# Date: 20210106

# Importing the relevant packages
print('=============Importing the relevant packages================================')
import tensorflow as tf
import tensorflow_datasets as tfds
  
# programstart stores current time 
import datetime; 
programstart = datetime.datetime.now() 

## Downloading and preprocessing the data

# Defining some constants/hyperparameters
BUFFER_SIZE = 70_000 # for reshuffling
BATCH_SIZE = 128
NUM_EPOCHS = 20

# Downloading the MNIST dataset
print('=============Downloading the MNIST dataset==================================')

# When 'with_info' is set to True, tfds.load() returns two variables: 
# - the dataset (including the train and test sets) 
# - meta info regarding the dataset itself

mnist_dataset, mnist_info = tfds.load(name='mnist', with_info=True, as_supervised=True)

# Extracting the train and test datasets
mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']

# Creating a function to scale our image data (it is recommended to scale the pixel values in the range [0,1] )
def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.

    return image, label

# Scaling the data
train_and_validation_data = mnist_train.map(scale)
test_data = mnist_test.map(scale)

# Defining the size of the validation set
num_validation_samples = 0.1 * mnist_info.splits['train'].num_examples
num_validation_samples = tf.cast(num_validation_samples, tf.int64)

# Defining the size of the test set
num_test_samples = mnist_info.splits['test'].num_examples
num_test_samples = tf.cast(num_test_samples, tf.int64)

# Reshuffling the dataset
train_and_validation_data = train_and_validation_data.shuffle(BUFFER_SIZE)

# Splitting the dataset into training + validation
train_data = train_and_validation_data.skip(num_validation_samples)
validation_data = train_and_validation_data.take(num_validation_samples)

# Batching the data
# NOTE: For proper functioning of the model, we need to create one big batch for the validation and test sets
train_data = train_data.batch(BATCH_SIZE)
validation_data = validation_data.batch(num_validation_samples) 
test_data = test_data.batch(num_test_samples)

## Creating the model and training it
print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
print('=============Creating the model and training it=============================')
print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
# Now that we have preprocessed the dataset, we can define our CNN and train it

# Outlining the model/architecture of our CNN

# CONV -> MAXPOOL -> CONV -> MAXPOOL -> FLATTEN -> DENSE
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
print('=============A brief summary of the model and parameters====================')
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

# Defining the loss function

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Compiling the model with Adam optimizer and the cathegorical crossentropy as a loss function
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

# Defining early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',    
    min_delta = 0,
    patience = 2,
    verbose = 0, 
    restore_best_weights = True
)

print('=============Train the network==============================================')
# Train the network
model.fit(
    train_data, 
    epochs = NUM_EPOCHS, 
    callbacks = [early_stopping], 
    validation_data = validation_data,
    verbose = 2
)

""" 422/422 - 6s - loss: 0.2780 - accuracy: 0.9202 - val_loss: 0.0891 - val_accuracy: 0.9738
Epoch 2/20
422/422 - 3s - loss: 0.0770 - accuracy: 0.9769 - val_loss: 0.0551 - val_accuracy: 0.9847
Epoch 3/20
422/422 - 3s - loss: 0.0549 - accuracy: 0.9828 - val_loss: 0.0547 - val_accuracy: 0.9852
Epoch 4/20
422/422 - 3s - loss: 0.0450 - accuracy: 0.9861 - val_loss: 0.0392 - val_accuracy: 0.9880
Epoch 5/20
422/422 - 2s - loss: 0.0374 - accuracy: 0.9882 - val_loss: 0.0262 - val_accuracy: 0.9917
Epoch 6/20
422/422 - 3s - loss: 0.0328 - accuracy: 0.9897 - val_loss: 0.0299 - val_accuracy: 0.9902
Epoch 7/20
422/422 - 3s - loss: 0.0285 - accuracy: 0.9912 - val_loss: 0.0242 - val_accuracy: 0.9925
Epoch 8/20
422/422 - 2s - loss: 0.0261 - accuracy: 0.9920 - val_loss: 0.0212 - val_accuracy: 0.9932
Epoch 9/20
422/422 - 2s - loss: 0.0221 - accuracy: 0.9927 - val_loss: 0.0166 - val_accuracy: 0.9952
Epoch 10/20
422/422 - 2s - loss: 0.0192 - accuracy: 0.9940 - val_loss: 0.0169 - val_accuracy: 0.9943
Epoch 11/20
422/422 - 3s - loss: 0.0176 - accuracy: 0.9945 - val_loss: 0.0188 - val_accuracy: 0.9960 """

## Testing our model
print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
print('=============Testing our model==============================================')
print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')

# Testing our model
test_loss, test_accuracy = model.evaluate(test_data)

"""1/1 [==============================] - 1s 622ms/step - loss: 0.0310 - accuracy: 0.9911"""

# Printing the test results
print('=============Printing the test results======================================')
print('Test loss: {0:.4f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))

""" Test loss: 0.0310. Test accuracy: 99.11% """

### Plotting images and the results

import matplotlib.pyplot as plt
import numpy as np

# Split the test_data into 2 arrays, containing the images and the corresponding labels
for images, labels in test_data.take(1):
    images_test = images.numpy()
    labels_test = labels.numpy()

# Reshape the images into 28x28 form, suitable for matplotlib (original dimensions: 28x28x1)
images_plot = np.reshape(images_test, (10000,28,28))

# The image to be displayed and tested
## Generating a random image, the 10000 is from the code above.
import random
i = random.randint(1,10000)


# Plot the image
plt.figure(figsize=(2, 2))
plt.axis('off')
plt.imshow(images_plot[i-1], cmap="gray", aspect='auto')
plt.show()

# Print the correct label for the image
print('=============Printing the correct label for the image=======================')
print("Label: {}".format(labels_test[i-1]))

""" Label: THIS WILL BE BASED ON WHATEVER YOUR IMAGE WAS """

# Obtain the model's predictions (logits)
predictions = model.predict(images_test[i-1:i])

# Convert those predictions into probabilities (recall that we incorporated the softmaxt activation into the loss function)
probabilities = tf.nn.softmax(predictions).numpy()

# Convert the probabilities into percentages
probabilities = probabilities*100

# Create a bar chart to plot the probabilities for each class
print('=============Create a bar chart to plot the probabilities for each class====')
plt.figure(figsize=(12,5))
plt.bar(x=[1,2,3,4,5,6,7,8,9,10], height=probabilities[0], tick_label=["0","1","2","3","4","5","6","7","8","9"])
plt.show()

# programend stores current time 
programend = datetime.datetime.now()
roundedstart = programstart - datetime.timedelta(microseconds=programstart.microsecond)
roundedend = programend - datetime.timedelta(microseconds=programend.microsecond)
print("Program Started:-", roundedstart) 
print("Program Ended:-", roundedend) 