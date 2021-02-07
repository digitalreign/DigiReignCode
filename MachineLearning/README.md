# MachineLearning
Welcome to my quilt of knowledge pertaining to Machine Learning. I am going to be walking through basic ML concepts and move into neural networks. \
https://machinelearningmastery.com/start-here/

## Python
Built-in Fuctions \
https://docs.python.org/3/library/functions.html 
Zip Fuction \
https://mathdatasimplified.com/2021/02/05/zip-associate-elements-from-two-iterators-based-on-the-order/
Graphviz: \
https://graphviz.readthedocs.io/en/stable/manual.html
Storing to local:
https://mathdatasimplified.com/2021/02/07/environment-variables-save-private-information-in-your-local-machine 

## d3

## Keras

### Keras Optimizers:
Keras provides APIs for various implementations of Optimizers. You will find the following types of optimizers in Keras – \
SGD - SGD optimizer uses gradient descent along with momentum. In this type of optimizer, a subset of batches is used for gradient calculation. \
RMSprop - In the RMSProp optimizer, the aim is to ensure a constant movement of the average of square of gradients. And secondly, the division of gradient by average’s root is also performed.\
Adam -The adam optimizer uses adam algorithm in which the stochastic gradient descent method is leveraged for performing the optimization process. It is efficient to use and consumes very little memory. It is appropriate in cases where huge amount of data and parameters are available for usage. \

Keras Adam Optimizer is the most popular and widely used optimizer for neural network training. \
Adadelta - In Adadelta optimizer,  it uses an adaptive learning rate with stochastic gradient descent method. Adadelta is useful to counter two drawbacks: \

The continuous learning rate degradation during training. \
It also solves the problem of the global learning rate. \
Adagrad - Keras Adagrad optimizer has learning rates that use specific parameters. Based on the frequency of updates received by a parameter, the working takes place. \

Even the learning rate is adjusted according to the individual features. This means there are different learning rates for some weights. \
Adamax \
Nadam \
Ftrl \
https://machinelearningknowledge.ai/keras-optimizers-explained-with-examples-for-beginners/

### L2 Regularization
Keras provides a weight regularization API that allows you to add a penalty for weight size to the loss function. \
https://machinelearningmastery.com/how-to-reduce-overfitting-in-deep-learning-with-weight-regularization/

### Dropout Regularization
When created, the dropout rate can be specified to the layer as the probability of setting each input to the layer to zero. This is different from the definition of dropout rate from the papers, in which the rate refers to the probability of retaining an input. \
https://machinelearningmastery.com/how-to-reduce-overfitting-with-dropout-regularization-in-keras/

### Data Augmentation
Training deep learning neural network models on more data can result in more skillful models, and the augmentation techniques can create variations of the images that can improve the ability of the fit models to generalize what they have learned to new images. \
The Keras deep learning neural network library provides the capability to fit models using image data augmentation via the ImageDataGenerator class. \
https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/

### FastAI and StreamLit
Another Image classifier \
https://towardsdatascience.com/how-to-create-an-app-to-classify-dogs-using-fastai-and-streamlit-af3e75f0ee28 

## Tensorflow
### Layers
https://missinglink.ai/guides/tensorflow/tensorflow-conv2d-layers-practical-guide/ 

### Saving and Loading pb files into Tensorflow
https://leimao.github.io/blog/Save-Load-Inference-From-TF-Frozen-Graph/

### Deep Playground
Deep playground is an interactive visualization of neural networks, written in TypeScript using d3.js. \
https://github.com/tensorflow/playground

### Tensorboard
TensorBoard is a suite of web applications for inspecting and understanding your TensorFlow runs and graphs. \
https://github.com/tensorflow/tensorboard

## Convolutional Neural Networks

### Pooling Layers
A pooling layer is a new layer added after the convolutional layer. Specifically, after a nonlinearity (e.g. ReLU) has been applied to the feature maps output by a convolutional layer; for example the layers in a model may look as follows: \
https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/ 

### Overfitting and Early Stopping
When training a large network, there will be a point during training when the model will stop generalizing and start learning the statistical noise in the training dataset. \
https://machinelearningmastery.com/early-stopping-to-avoid-overtraining-neural-network-models/

## Existing Models
AlexNet \
VGG \ 
GoogLeNet \
ResNet
Slay the Spire \

## Extras
https://towardsdatascience.com/bringing-deep-neural-networks-to-slay-the-spire-a2971d5a5115 
Free AWS Machine Learning \
https://aws.amazon.com/free/machine-learning/ 
