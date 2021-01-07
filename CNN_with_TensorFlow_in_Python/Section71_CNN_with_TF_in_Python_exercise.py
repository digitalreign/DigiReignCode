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
model.fit(
    train_data, 
    epochs = NUM_EPOCHS, 
    callbacks = [tensorboard_callback, early_stopping], 
    validation_data = validation_data,
    verbose = 2
)





# programend stores current time 
programend = datetime.datetime.now()
roundedstart = programstart - datetime.timedelta(microseconds=programstart.microsecond)
roundedend = programend - datetime.timedelta(microseconds=programend.microsecond)
print("Program Started:-", roundedstart) 
print("Program Ended:-", roundedend) 