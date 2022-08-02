from __future__ import absolute_import
from statistics import variance
from matplotlib import pyplot as plt
from preprocess import get_data
from convolution import conv2d

import os
import tensorflow as tf
import numpy as np
import random
import math

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class Model(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that 
        classifies images. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()

        self.batch_size = 200
        self.num_classes = 2
        self.optimizer=tf.keras.optimizers.Adam(learning_rate=.001)
        self.loss_list = [] # Append losses to this list in training so you can visualize loss vs time in main

        # TODO: Initialize all hyperparameters

        # TODO: Initialize all trainable parameters
        self.f1 = tf.Variable(tf.random.truncated_normal(shape=[5,5,3,16], stddev=.1, dtype=tf.float32))
        self.f2 =tf.Variable(tf.random.truncated_normal(shape=[5,5,16,20], stddev=.1, dtype=tf.float32))
        self.f3 =tf.Variable(tf.random.truncated_normal(shape=[3,3,20,20], stddev=.1, dtype=tf.float32))
		
        self.f1b =tf.Variable(tf.random.truncated_normal([16], stddev=.1, dtype=tf.float32))
        self.f2b = tf.Variable(tf.random.truncated_normal([20], stddev=.1, dtype=tf.float32))
        self.f3b = tf.Variable(tf.random.truncated_normal([20], stddev=.1, dtype=tf.float32))

        self.w1 = tf.Variable(tf.random.truncated_normal(shape=[320,16], stddev=.1, dtype=tf.float32)) 
        self.w2 = tf.Variable(tf.random.truncated_normal(shape=[16,8], stddev=.1, dtype=tf.float32))
        self.w3 = tf.Variable(tf.random.truncated_normal(shape=[8,2], stddev=.1, dtype=tf.float32))

        self.b1 = tf.Variable(tf.random.truncated_normal([16], stddev=.1, dtype=tf.float32))
        self.b2 = tf.Variable(tf.random.truncated_normal([8], stddev=.1, dtype=tf.float32))
        self.b3 = tf.Variable(tf.random.truncated_normal([2], stddev=.1, dtype=tf.float32))
    
    def flattening_layer(self, layer):
        input_size = layer.get_shape().as_list()
        new_size = input_size[-1] * input_size[-2] * input_size[-3]
        return tf.reshape(layer, [-1, new_size]
)

    def call(self, inputs, is_testing=True):
        """
        Runs a forward pass on an input batch of images.
        
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
        # shape of filter = (filter_height, filter_width, in_channels, out_channels)
        # shape of strides = (batch_stride, height_stride, width_stride, channels_stride)

        variance_epsilon=.00001
        stride_size=[1,2,2,1]
        axes=[0,1,2]
        dropout_rate=.3
        strides=[1,1,1,1]
        ksize=[1,3,3,1]
        padding='SAME'

        conv1 = tf.nn.conv2d(inputs,self.f1,stride_size,padding=padding)
        conv1_w_b = tf.nn.bias_add(conv1, self.f1b)
        batch_mean1, batch_var1 = tf.nn.moments(conv1_w_b,axes)
        batch_norm1 = tf.nn.batch_normalization(conv1_w_b, batch_mean1, batch_var1,None,None,variance_epsilon)
        conv1n = tf.nn.relu(batch_norm1)
        pooled_conv_1 = tf.nn.max_pool(conv1n, ksize, stride_size, padding=padding)
        conv_1 = pooled_conv_1
		
        #Layer 2
        conv2 = tf.nn.conv2d(conv_1, self.f2, strides, padding=padding)
        conv2_w_b = tf.nn.bias_add(conv2, self.f2b)
        batch_mean2, batch_var2 = tf.nn.moments(conv2_w_b,axes)
        batch_norm2 = tf.nn.batch_normalization(conv2_w_b, batch_mean2, batch_var2,None,None,variance_epsilon)
        conv2n = tf.nn.relu(batch_norm2)
        pooled_conv_2 = tf.nn.max_pool(conv2n, ksize, stride_size, padding=padding)
        conv_2 = pooled_conv_2
		
        #Layer 3
        if is_testing:
            conv3 = conv2d(conv_2, self.f3,strides,padding=padding)
        else: 
            conv3 = tf.nn.conv2d(conv_2, self.f3,strides,padding=padding)
		
        conv3_w_b = tf.nn.bias_add(conv3, self.f3b)
        batch_mean3, batch_var3 = tf.nn.moments(conv3_w_b,axes)
        batch_norm3 = tf.nn.batch_normalization(conv3_w_b, batch_mean3, batch_var3, None, None, variance_epsilon)
        conv3n = tf.nn.relu(batch_norm3)
        conv = self.flattening_layer(conv3n)

        layer1o = tf.nn.relu(tf.matmul(conv, self.w1) +self.b1)
        layer1oD = tf.nn.dropout(layer1o, dropout_rate)
       
        layer2o = tf.nn.leaky_relu(tf.matmul(layer1oD, self.w2) + self.b2)
        layer2oD = tf.nn.dropout(layer2o, dropout_rate)
        
        logits = tf.matmul(layer2oD, self.w3) + self.b3
        return logits
        

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.
        
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        loss = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
        return tf.reduce_mean(loss)
        

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT
        
        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs 
    and labels - ensure that they are shuffled in the same order using tf.gather or zipping.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: Optionally list of losses per batch to use for visualize_loss
    '''
    train_inputs_length=len(train_inputs)
    
    sack = tf.range(0, train_inputs_length)  
    shuffle = tf.random.shuffle(sack, seed=3)
   
    train_inputs = tf.gather(train_inputs, shuffle)
    train_labels = tf.gather(train_labels, shuffle)
    
    num_batches = train_inputs_length/model.batch_size
    num_batches= int(num_batches)
    
    for i in range(num_batches):
        input = tf.image.random_flip_left_right(train_inputs[i*(model.batch_size):(i+1)*model.batch_size])
        label = train_labels[i*(model.batch_size):(i+1)*model.batch_size]
		
        with tf.GradientTape() as tape:
            predictions = model.call(input)
            loss = model.loss(predictions, label)
            train_acc = model.accuracy(predictions, label)
            print("The accuracy  after {} training steps on the training data is: {}".format(i, train_acc))
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly 
    flip images or do any extra preprocessing.
    
    :param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """
    prob= model.call(test_inputs)
    test_acc= model.accuracy(prob, test_labels)

    return test_acc


def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()  


def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
    """
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (50, num_classes)
    :param image_labels: the labels from get_data(), shape (50, num_classes)
    :param first_label: the name of the first class, "cat"
    :param second_label: the name of the second class, "dog"

    NOTE: DO NOT EDIT

    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """
    # Helper function to plot images into 10 columns
    def plotter(image_indices, label): 
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle("{} Examples\nPL = Predicted Label\nAL = Actual Label".format(label))
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i+1)
            ax.imshow(image_inputs[ind], cmap="Greys")
            pl = first_label if predicted_labels[ind] == 0.0 else second_label
            al = first_label if np.argmax(
                image_labels[ind], axis=0) == 0 else second_label
            ax.set(title="PL: {}\nAL: {}".format(pl, al))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
        
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = image_inputs.shape[0]

    # Separate correct and incorrect images
    correct = []
    incorrect = []
    for i in range(num_images): 
        if predicted_labels[i] == np.argmax(image_labels[i], axis=0): 
            correct.append(i)
        else: 
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()


def main():
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and 
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs. 
    
    CS1470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=70%.
    
    CS2470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=75%.
    
    :return: None
    '''
    model = Model()
    train_inputs, train_labels = get_data('/Users/loganheft/Desktop/hw2-cnns-hefty99-master/data/train', 3, 5)
    test_inputs, test_labels = get_data('/Users/loganheft/Desktop/hw2-cnns-hefty99-master/data/test', 3, 5)
    for i in range(1):
        train(model, train_inputs, train_labels)
    print(test(model, test_inputs, test_labels))
    visualize_results(test_inputs[300:310], model.call(test_inputs[300:310]), test_labels[300:310], "dog", "cat") 
    return


if __name__ == '__main__':
    main()
