## Architectural Basics

# Number of layers:
It has 8 convolution layers of which 6 of them uses 3x3 kernel, one uses 1x1 and one uses 4x4 kernel.
It has 1 maxpool layer.

# MaxPooling
The architecture uses 1 maxpool layer. The use of maxpool layer allows us to reduce the dimension of image and thus in turn reduces the 
number of layers and the computaion.

# 1x1 convolution
The architecture uses one 1x1 convolution. This has been used to retain the same receptive field.

# 3x3 convolutions
The architecture uses 6 3x3 convolutions. 3x3 convolutions are mostly used.

# Receptive Field

# Softmax
Softmax has been used at the end. It converts the activations into probabilities which shows the maximum likelihood of a class.

# Learning Rate
Learning Rate of 0.003 has been used initially. With the help of Learning Rate Scheduler, the learning rate can be changed with each
epoch to make the loss converge.

# Kernels
In this architecture, 3 types of kernels has been used namely 3x3, 1x1 and 4x4. The number of kernels in this architecture has been
kept low as we want to achieve a total of less than 15k parameters.

# Batch Normalization
Batch normalization has been used after each convolution. Batch normalization normalizes the output of a previous activation layer by subtracting the batch mean and dividing by the batch standard deviation.

# Image Normalization
Image Normalization is done as a part of the preprocessing. The image pixel values are divided by 255.

# Position of MaxPooling
Maxpooling is used when we can see the edges and gradients. The receptive fields may differ depending on the dimension of image.
In this case, we can use maxpooling at a receptive field of 5x5 since the dimension is small.

# Concept of Transition Layer

# Number of epochs and when to increase them
We generally begin with less number of epochs. If we see that loss is gradually decreasing and this trend goes till the end of 
last epoch, that means there is still a possibility of decreasing the loss and hence we can use more epochs to get a converging 
loss.

# Dropout
In this architecture, Dropout of 0.1 has been used after each convolution. This means we are discarding the 10% of the activations
at each layer randomly. By doing this we dont rely on a specific combinations of activation. This reduces the overfitting in the
network.

# When do we introduce DropOut, or when do we know we have some overfitting
After training the model, if we see a gap between training and validation accuracy i.e. Training accuracy is more than validation 
accuracy, that means the network is overfitting. To reduce this gap, we employ dropout.

# The distance of MaxPooling from Prediction
Maxpooling is used generally used atleast few receptive fields from the prediction. If we use just before the prediction, it may not give the correct prediction, because network is not able to see the whole image before making predictions.

# The distance of Batch Normalization from Prediction
Batch Normalization can be used after each convolution but should not be use just before the prediction layer.

# How do we know our network is not going well, comparatively, very early
The good models should be able to achieve a training accuracy of more than 90% with the few epochs itself and the loss should 
continuosly decrease with the epochs. If something like this doesn't happen, that means there is something wrong with the network.

# Batch Size, and effects of batch size
With the increase in the batch size, the validation accuracy increases till it reaches the saturation point and starts decreasing.

# LR schedule and concept behind it
LR schedule is used to decrease the learning rate with the increasing epochs. This is done to converge the loss to its minimum. If 
a static learning rate is used, loss may not converge and can overshoot its minimum on the other side.

# Adam vs SGD
Adam is fast but generalize poorly compared to SGD.
SGD is a variant of gradient descent. Instead of performing computations on the whole dataset — which is redundant and inefficient — SGD only computes on a small subset or random selection of data examples. SGD produces the same performance as regular gradient descent when the learning rate is low.


