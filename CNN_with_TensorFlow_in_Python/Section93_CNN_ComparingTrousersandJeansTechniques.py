# Section CNN - Comparing Trousers and Jeans Techniques
# Author Jose Smith
# Start Date: 20210121
# End Date: 

# Importing the relevant packages
print('=============Importing the relevant packages================================')
import numpy as np

import tensorflow as tf

import datetime

# programstart stores current time 
programstart = datetime.datetime.now() 

# Loading the dataset
# Only need the test set, as we will not train any netwroks in this notebook
data_test  = np.load(r"CNN_with_TensorFlow_in_Python/9_Practical_Project/Dataset/Trousers & Jeans - All - Test.npz")

# Extracting the images array
print('=============Extracting the images array====================================')
images_test = data_test['images']

# Extracting the label arrays
gender_test = data_test['genders']
type_test   = data_test['labels'] % 2

# The Type label is contained in the combined labels:
#    - 0 and 2 correspond to Trousers   (0)
#    - 1 and 3 correspond to Jeans      (1)
# 0 and 2 are both even, 1 and 3 are odd
# Therefore '% 2' works as it: 
#     returns 0, for input 0 and 2
# and returns 1, for input 1 and 3

# Scaling the pixel values
images_test = images_test/255.0

# Loading the necessary models
print('=============Loading the necessary models===================================')

# Model for 'Combined Labels' approach
model_all = tf.keras.models.load_model(r"CNN_with_TensorFlow_in_Python\9_Practical_Project\saved_models\Model_All")

# Models for 'Hierarchical classification' approach
model_gender = tf.keras.models.load_model(r"CNN_with_TensorFlow_in_Python\9_Practical_Project\saved_models\Model_Gender")
model_male = tf.keras.models.load_model(r"CNN_with_TensorFlow_in_Python\9_Practical_Project\saved_models\Model_Male")
model_female = tf.keras.models.load_model(r"CNN_with_TensorFlow_in_Python\9_Practical_Project\saved_models\Model_Female")

# Defining the scores for both approaches
score_all = 0
score_hierarchy = 0


for i in range(len(images_test)):
    
    
    # Testing the 'Combined Labels' approach
    
    # Obtaining the model's output for the image
    predict_all = model_all(images_test[i:i+1])
    # The predicted label is the index corresponding with the highest score
    label_all = np.argmax(predict_all)
    
    # Scoring the prediction
    if label_all // 2 == gender_test[i]: # Combined Label // 2 corresponds to the 'gender' label
        score_all = score_all + 1
    
    if label_all % 2 == type_test[i]:    # Combined Label % 2 corresponds to the 'type' label
        score_all = score_all + 1


    # Testing the 'Hierarchical Classification' approach
    
    # Running the Gender model first
    predict_gender = model_gender(images_test[i:i+1])
    label_gender = np.argmax(predict_gender)
    
    if label_gender == gender_test[i]:
        score_hierarchy = score_hierarchy + 1
    
    
    # Evaluating the Male model, if the gender prediction was male
    if label_gender == 0:
        
        predict_male = model_male(images_test[i:i+1])
        label_type = np.argmax(predict_male)
        
        if label_type == type_test[i]:
            score_hierarchy = score_hierarchy + 1
    
    # Evaluating the Female model, if the gender prediction was female
    if label_gender == 1:
        
        predict_female = model_female(images_test[i:i+1])
        label_type = np.argmax(predict_female)
        
        if label_type == type_test[i]:
            score_hierarchy = score_hierarchy + 1

# Printing the scores
print('=============Printing the scores============================================')

print("Combined Labels: \n{0} points \n \nHierarchical Classification: \n{1} points".format(score_all,score_hierarchy))


# programend stores current time 
programend = datetime.datetime.now()
roundedstart = programstart - datetime.timedelta(microseconds=programstart.microsecond)
roundedend = programend - datetime.timedelta(microseconds=programend.microsecond)
print("Program Started:-", roundedstart) 
print("Program Ended:-", roundedend) 