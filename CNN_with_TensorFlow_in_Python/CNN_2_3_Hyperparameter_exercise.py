# Section CNN - MNIST exercise with Hyperparameter Tuning
# Author Jose Smith
# Start Date: 20210112
# End Date: 20210112

# Importing the relevant packages
print('=============Importing the relevant packages================================')
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorboard.plugins.hparams import api as hp

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

## Defining hyperparameters
print('=============Defining Hyperparameters=========================++++++++++++++')

# Defining the hypermatarest we would test and their range
HP_FILTER_SIZE = hp.HParam('filter_size', hp.Discrete([3,5,7]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'adagrad', 'rmsprop', 'adadelta', 'sgd']))

METRIC_ACCURACY = 'accuracy'

# Logging setup info
with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[HP_FILTER_SIZE, HP_OPTIMIZER],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
    )

## Cerating functions for training our model and for logging purposes
# Wrapping our model and training in a function
def train_test_model(hparams):
    
    # Outlining the model/architecture of our CNN
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(50, hparams[HP_FILTER_SIZE], activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(50, hparams[HP_FILTER_SIZE], activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)), 
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10)
    ])
    
    # Defining the loss function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Compiling the model with parameter value for the optimizer
    model.compile(optimizer=hparams[HP_OPTIMIZER], loss=loss_fn, metrics=['accuracy'])
    
    # Defining early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        mode = 'auto',
        min_delta = 0,
        patience = 2,
        verbose = 0, 
        restore_best_weights = True
    )
    
    # Training the model
    model.fit(
        train_data, 
        epochs = NUM_EPOCHS,
        callbacks = [early_stopping],
        validation_data = validation_data,
        verbose = 2
    )
    
    _, accuracy = model.evaluate(test_data)
    
    return accuracy

# Creating a function to log the resuls
def run(log_dir, hparams):
    
    with tf.summary.create_file_writer(log_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model(hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

## Training the model with the different hyperparameters

# Performing a grid search on the hyperparameters we need to test
session_num = 0

for filter_size in HP_FILTER_SIZE.domain.values:
    for optimizer in HP_OPTIMIZER.domain.values:
    
        hparams = {
            HP_FILTER_SIZE: filter_size,
            HP_OPTIMIZER: optimizer
        }
        run_name = "run-%d" % session_num 
        print('--- Starting trial: %s' % run_name)
        print({h.name: hparams[h] for h in hparams})
        run('logs/hparam_tuning/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), hparams)

        session_num += 1

print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
print('=============Visualizing in Tensorboard=====================================')
print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')

# programend stores current time 
programend = datetime.datetime.now()
roundedstart = programstart - datetime.timedelta(microseconds=programstart.microsecond)
roundedend = programend - datetime.timedelta(microseconds=programend.microsecond)
print("Program Started:-", roundedstart) 
print("Program Ended:-", roundedend) 

# Loading the Tensorboard extension
# This code will stop the program and run tensorboard.
# TensorBoard 2.4.0 can be accessed at http://localhost:6006/ (Press CTRL+C to quit)

import os
osdir = os.getcwd()
os.system('python -m tensorboard.main --logdir {}\\logs\\hparam_tuning'.format(osdir))

# The code below is much cleaner but is a jupyter extension:
# %load_ext tensorboard
# %tensorboard --logdir "logs/fit"

# TO KILL TENSORBOARD IN WINDOWS: 
# command line> taskkill /im tensorboard.exe /f
# command line> del /q %TMP%\.tensorboard-info\*



