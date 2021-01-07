# Section CNN - MNIST exercise with Tensorboard
# Author Jose Smith
# Start Date: 20210107
# End Date: 

# Importing the relevant packages
print('=============Importing the relevant packages================================')
import tensorflow as tf
import tensorflow_datasets as tfds
import datetime

# programstart stores current time 
programstart = datetime.datetime.now() 

## Downloading and preprocessing the data
print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
print('=============Downloading and preprocessing the data=========================')
print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
# Defining some constants/hyperparameters
BUFFER_SIZE = 70_000 # for reshuffling
BATCH_SIZE = 128
NUM_EPOCHS = 20

# Downloading the MNIST dataset
mnist_dataset, mnist_info = tfds.load(name='mnist', with_info=True, as_supervised=True)

mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']

# Creating a function to scale our data
def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.

    return image, label

# Scaling the data
train_and_validation_data = mnist_train.map(scale)
test_data = mnist_test.map(scale)

# Defining the size of validation set
num_validation_samples = 0.1 * mnist_info.splits['train'].num_examples
num_validation_samples = tf.cast(num_validation_samples, tf.int64)

# Defining size of test set
num_test_samples = mnist_info.splits['test'].num_examples
num_test_samples = tf.cast(num_test_samples, tf.int64)

# Reshuffling the dataset
train_and_validation_data = train_and_validation_data.shuffle(BUFFER_SIZE)

# Splitting the dataset into training + validation
train_data = train_and_validation_data.skip(num_validation_samples)
validation_data = train_and_validation_data.take(num_validation_samples)

# Batching the data
train_data = train_data.batch(BATCH_SIZE)
validation_data = validation_data.batch(num_validation_samples)
test_data = test_data.batch(num_test_samples)

## Creating the model and training it

# Outlining the model/architecture of our CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(50, 5, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(50, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)), 
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10)
])

# A brief summary of the model and parameters
print('=============A brief summary of the model and parameters====================')
model.summary(line_length = 75)

''' =============A brief summary of the model and parameters====================
Model: "sequential"
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

# Logging the training process data to use later in tensorboard
log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the network
print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
print('=============Creating the model and training it=============================')
print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
model.fit(
    train_data, 
    epochs = NUM_EPOCHS, 
    callbacks = [tensorboard_callback, early_stopping], 
    validation_data = validation_data,
    verbose = 2
)

''' 422/422 - 7s - loss: 0.2642 - accuracy: 0.9266 - val_loss: 0.0791 - val_accuracy: 0.9767
Epoch 2/20
422/422 - 3s - loss: 0.0702 - accuracy: 0.9789 - val_loss: 0.0473 - val_accuracy: 0.9872
Epoch 3/20
422/422 - 3s - loss: 0.0505 - accuracy: 0.9851 - val_loss: 0.0408 - val_accuracy: 0.9880
Epoch 4/20
422/422 - 3s - loss: 0.0417 - accuracy: 0.9872 - val_loss: 0.0326 - val_accuracy: 0.9905
Epoch 5/20
422/422 - 3s - loss: 0.0360 - accuracy: 0.9892 - val_loss: 0.0352 - val_accuracy: 0.9883
Epoch 6/20
422/422 - 3s - loss: 0.0318 - accuracy: 0.9901 - val_loss: 0.0296 - val_accuracy: 0.9897
Epoch 7/20
422/422 - 3s - loss: 0.0273 - accuracy: 0.9916 - val_loss: 0.0203 - val_accuracy: 0.9930
Epoch 8/20
422/422 - 3s - loss: 0.0235 - accuracy: 0.9928 - val_loss: 0.0215 - val_accuracy: 0.9933
Epoch 9/20
422/422 - 3s - loss: 0.0205 - accuracy: 0.9934 - val_loss: 0.0196 - val_accuracy: 0.9938
Epoch 10/20
422/422 - 3s - loss: 0.0191 - accuracy: 0.9940 - val_loss: 0.0179 - val_accuracy: 0.9938
Epoch 11/20
422/422 - 3s - loss: 0.0161 - accuracy: 0.9949 - val_loss: 0.0118 - val_accuracy: 0.9967
Epoch 12/20
422/422 - 3s - loss: 0.0139 - accuracy: 0.9956 - val_loss: 0.0103 - val_accuracy: 0.9972
Epoch 13/20
422/422 - 3s - loss: 0.0123 - accuracy: 0.9959 - val_loss: 0.0104 - val_accuracy: 0.9960
Epoch 14/20
422/422 - 3s - loss: 0.0116 - accuracy: 0.9961 - val_loss: 0.0126 - val_accuracy: 0.9972 '''

## Testing our model
print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
print('=============Testing our model==============================================')
print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')

# Testing our model
test_loss, test_accuracy = model.evaluate(test_data)
''' 1/1 [==============================] - 1s 612ms/step - loss: 0.0316 - accuracy: 0.9907 '''

# Printing the test results
print('Test loss: {0:.4f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))
''' Test loss: 0.0316. Test accuracy: 99.07% '''

# programend stores current time 
programend = datetime.datetime.now()
roundedstart = programstart - datetime.timedelta(microseconds=programstart.microsecond)
roundedend = programend - datetime.timedelta(microseconds=programend.microsecond)
print("Program Started:-", roundedstart) 
print("Program Ended:-", roundedend) 