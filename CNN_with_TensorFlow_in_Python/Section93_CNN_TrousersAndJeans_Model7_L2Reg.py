# Section CNN - Trousers and Jeans - Model 7 - L2 Regularization
# Author Jose Smith
# Start Date: 20210128
# End Date: 

# Importing the relevant packages
print('=============Importing the relevant packages================================')
import io
import itertools

import numpy as np
import sklearn.metrics

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

import matplotlib.pyplot as plt

import datetime

# programstart stores current time 
programstart = datetime.datetime.now() 

# Loading the datasets
print('=============Loading the datasets===========================================')

# I have been running this code in the base git directory and in this case I am pulling images from the 9_Practical_Project sub directory where the dataset is being held.
data_train = np.load(r"CNN_with_TensorFlow_in_Python/9_Practical_Project/Dataset/Trousers & Jeans - All - Train.npz")
data_val = np.load(r"CNN_with_TensorFlow_in_Python/9_Practical_Project/Dataset/Trousers & Jeans - All - Validation.npz")
data_test = np.load(r"CNN_with_TensorFlow_in_Python/9_Practical_Project/Dataset/Trousers & Jeans - All - Test.npz")

# Extracting the arrays from the imported data
print('=============Extracting the arrays from the imported data===================')
images_train = data_train['images']
labels_train = data_train['labels']

images_val = data_val['images']
labels_val = data_val['labels']

images_test = data_test['images']
labels_test = data_test['labels']

# Scaling the pixel values of all images
print('=============Scaling the pixel values of all images=========================')
images_train = images_train/255.0
images_val = images_val/255.0
images_test = images_test/255.0

# Defining constants
### Updated to 20 Epochs ###
EPOCHS = 20
BATCH_SIZE = 64

# Defining the hyperparameters we would tune, and their values to be tested
### For this model I am hardcoding the parameters that were the most successful from the last model run. ###
### Added HP_LAMBDA_REG so that we can change the L2 variable. ###
#HP_FILTER_SIZE_1 = hp.HParam('filter_size_1', hp.Discrete([3,5,7]))
#HP_FILTER_NUM = hp.HParam('filters_number', hp.Discrete([32,64,96,128]))
#HP_FILTER_SIZE_2 = hp.HParam('filter_size_2', hp.Discrete([3,5]))
#HP_DENSE_SIZE = hp.HParam('dense_size', hp.Discrete([256,512,1024]))
HP_FILTER_SIZE_1 = hp.HParam('filter_size_1', hp.Discrete([5]))
HP_FILTER_NUM = hp.HParam('filters_number', hp.Discrete([32]))
HP_FILTER_SIZE_2 = hp.HParam('filter_size_2', hp.Discrete([3]))
HP_DENSE_SIZE = hp.HParam('dense_size', hp.Discrete([256]))
HP_LAMBDA_REG = hp.HParam('lambda', hp.Discrete([0.0, 1e-5, 3e-5, 7e-5, 9e-5, 1e-4, 3e-4, 5e-4, 7e-4, 9e-4, 1e-3, 3e-3, 5e-3, 7e-3, 9e-3, 1e-2, 3e-2, 5e-2, 7e-2, 9e-2, 0.1]))

METRIC_ACCURACY = 'accuracy'

# Logging setup info
with tf.summary.create_file_writer(r'logs/Model_7_L2Reg/hparam_tuning/').as_default():
    hp.hparams_config(
        hparams=[HP_FILTER_SIZE_1, HP_FILTER_NUM, HP_FILTER_SIZE_2, HP_DENSE_SIZE, HP_LAMBDA_REG],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
    )
# Wrapping our model and training in a function
print('=============Wrapping our model and training in a function==================')

def train_test_model(hparams, session_num):
    
    # Outlining the model/architecture of our CNN
### Added the L2 regularization ###
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(hparams[HP_FILTER_NUM], hparams[HP_FILTER_SIZE_1], activation='relu', input_shape=(120,90,3), kernel_regularizer=tf.keras.regularizers.l2(hparams[HP_LAMBDA_REG])),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(hparams[HP_FILTER_NUM], hparams[HP_FILTER_SIZE_2], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(hparams[HP_LAMBDA_REG])),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hparams[HP_DENSE_SIZE], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(hparams[HP_LAMBDA_REG])),
        tf.keras.layers.Dense(4)
    ])

    # Defining the loss function
    ### Added variable name to loss ###
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Compiling the model with parameter value for the optimizer
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy', tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name='sparse_categorical_crossentropy')])

    # Defining the logging directory
    log_dir = "logs\\Model_7_L2Reg\\fit\\" + "run-{}".format(session_num)
    
    
    def plot_confusion_matrix(cm, class_names):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.

        Args:
          cm (array, shape = [n, n]): a confusion matrix of integer classes
          class_names (array, shape = [n]): String names of the integer classes
        """
        figure = plt.figure(figsize=(12, 12))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Normalize the confusion matrix.
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=3)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return figure
    def plot_to_image(figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image
    
    
    # Defining a file writer for Confusion Matrix logging purposes
    file_writer_cm = tf.summary.create_file_writer(log_dir + '/cm')     
    
    
    def log_confusion_matrix(epoch, logs):
        # Use the model to predict the values from the validation dataset.
        test_pred_raw = model.predict(images_val)
        test_pred = np.argmax(test_pred_raw, axis=1)

        # Calculate the confusion matrix.
        cm = sklearn.metrics.confusion_matrix(labels_val, test_pred)
        # Log the confusion matrix as an image summary.
        figure = plot_confusion_matrix(cm, class_names=['Trousers Male', 'Jeans Male', 'Trousers Female', 'Jeans Female'])
        cm_image = plot_to_image(figure)

        # Log the confusion matrix as an image summary.
        with file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)
    
    
    
    # Define the Tensorboard and Confusion Matrix callbacks.
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=0)
    cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

    
    # Defining early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_sparse_categorical_crossentropy',
        mode = 'auto',
        min_delta = 0,
        patience = 2,
        verbose = 0, 
        restore_best_weights = True
    )
    
    # Training the model
    model.fit(
        images_train,
        labels_train,
        epochs = EPOCHS,
        batch_size = BATCH_SIZE,
        callbacks = [tensorboard_callback, cm_callback, early_stopping],
        validation_data = (images_val,labels_val),
        verbose = 2
    )
    
    _, accuracy, _ = model.evaluate(images_val,labels_val)
    
    # Saving the current model for future reference
    model.save(r"saved_models\Model_7_L2Reg\Run-{}".format(session_num))
    
    return accuracy

# Creating a function to log the results
print('=============Creating a function to log the results=========================')

def run(log_dir, hparams, session_num):
    
    with tf.summary.create_file_writer(log_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model(hparams, session_num)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

session_num = 1
### Added the L2 as a layer ###
for lambda_reg in HP_LAMBDA_REG.domain.values:
    for filter_size_1 in HP_FILTER_SIZE_1.domain.values:
        for filter_num in HP_FILTER_NUM.domain.values:
            for filter_size_2 in HP_FILTER_SIZE_2.domain.values:
                for dense_size in HP_DENSE_SIZE.domain.values:

                    hparams = {
                        HP_FILTER_SIZE_1: filter_size_1,
                        HP_FILTER_NUM: filter_num,
                        HP_FILTER_SIZE_2: filter_size_2,
                        HP_DENSE_SIZE: dense_size,
                        HP_LAMBDA_REG: lambda_reg
                    }

                    run_name = "run-%d" % session_num
                    print('--- Starting trial: %s' % run_name)
                    print({h.name: hparams[h] for h in hparams})
                    run('Logs/Model_7_L2Reg/hparam_tuning/' + run_name, hparams, session_num)

                    session_num += 1


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
# I have a windows cleanup script here to clear out tensorboard.
os.system('python -m tensorboard.main --logdir {}\\logs\\Model_7_L2Reg\\hparam_tuning'.format(osdir))
print("=============Cleaning the windows cache for tensorboard.====================")
os.system('taskkill /im tensorboard.exe /f')
os.system('del /q %TMP%\.tensorboard-info\*')
print("=============Windows cache cleared for tensorboard.=========================")
os.system('python -m tensorboard.main --logdir {}\\logs\\Model_7_L2Reg\\fit'.format(osdir))
print("=============Cleaning the windows cache for tensorboard.====================")
os.system('taskkill /im tensorboard.exe /f')
os.system('del /q %TMP%\.tensorboard-info\*')
print("=============Windows cache cleared for tensorboard.=========================")



# The code below is much cleaner but is a jupyter extension:
# %load_ext tensorboard
# %tensorboard --logdir "logs/fit"

# TO KILL TENSORBOARD IN WINDOWS: 
# command line> taskkill /im tensorboard.exe /f
# command line> del /q %TMP%\.tensorboard-info\*