import numpy as np
import matplotlib.pyplot as plt
import struct
from array import array
from os.path  import join

# The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = np.array(array("B", file.read()))  # Convert labels to a NumPy array
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        # Convert image data to a NumPy array and reshape
        images = np.array(image_data).reshape(size, rows, cols)          
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)  
    
#
# Verify Reading Dataset via MnistDataloader class
#

#
# Set file paths based on added MNIST Datasets
#
input_path = '/Users/andrewzhou/microservices_suite'
training_images_filepath = join(input_path, 'train/train-images.idx3-ubyte')
training_labels_filepath = join(input_path, 'train/train-labels.idx1-ubyte')
test_images_filepath = join(input_path, 'tests/t10k-images.idx3-ubyte')
test_labels_filepath = join(input_path, 'tests/t10k-labels.idx1-ubyte')

#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);        
        index += 1

#
# Load MINST dataset
#
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

#
# Show some random training and test images 
import random
images_2_show = []
titles_2_show = []
for i in range(10):
    r = random.randint(1, 60000)
    images_2_show.append(x_train[r])
    titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))    

for i in range(5):
    r = random.randint(1, 10000)
    images_2_show.append(x_test[r])        
    titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))    

show_images(images_2_show, titles_2_show)
plt.show()

# x_train, y_train, x_test, y_test are directly read from the binary files
# x_train, x_test are numpy arrays of 28x28 image
# y_train, y_test, are lists of integers representing the corresponding labels (0-9)
# 28 * 28 = 784 pixel training images, each pixel has a value 0-255
# matrix X, m rows of training images, of 784 columns
# transpose X, m columns of training images, of 784 rows
# images classified using labels (0-9)

# 60000 examples images {training set}
print(x_train.shape) # Should output something like (60000, 28, 28)
print(y_train.shape) # Should output (60000,)
# 10000 examples images {testing set}
print(x_test.shape) # Should output something like (10000, 28, 28)
print(y_test.shape) # Should output (10000,)

batch_size = 32
max_iters = 4000
eval_interval = 300
learning_rate = 1e-2

# preprocess the data

def preprocess_images(images):
    # Flatten the images from (28, 28) to (784,) and convert to np array
    pass

# Preprocess the training and test images
X_train = preprocess_images(x_train)
X_test = preprocess_images(x_test)

# Convert labels to np arrays
def preprocess_labels(labels):
    pass

# Preprocess the training and test labels
Y_train = preprocess_labels(y_train)
Y_test = preprocess_labels(y_test)

# Split training data into training and development sets
n = int(0.9 * len(X_train))
train_data = X_train[:n]
dev_data = X_train[n:]
train_label = Y_train[:n]
dev_label = Y_train[n:]

def get_batch(split):
    # 'train', 'dev', error if neither
    """
    Generate a small batch of data of inputs X and targets Y using DataLoader.
    
    Parameters:
    - split (str): 'train' or 'val' to indicate which dataset to use.
    
    Returns:
    - X (torch.Tensor): A batch of input images.
    - Y (torch.Tensor): A batch of corresponding labels.
    """
    pass


def initialize_params():
    #Manually Define params
    W1 = np.rand(10, 784) - 0.5
    b1 = np.rand(10, 1) - 0.5
    W2 = np.rand(10, 10) - 0.5
    b2 = np.rand(10, 1) - 0.5

    # two layer neural network
    # input (zeroth 0) layer 784 nodes -> hidden (first 1) layer 10 nodes -> output (second 2) layer 10 nodes
    # forward propagation: take an image, run it through the network, compute what the output is going to be
    # 7840 weights correspond to the 7840 connnections in the neural network
    # layer 0 --> linear combination --> layer 1
    # A^[0] = X, the input layer 784*m matrix; Z^[1] = w^[1] * A^[0] + b^[1]
    # w^[1] weights 10*784 matrix, b^[1] bias 10*1 --> 10* m matrix
    # without an activation function, no hidden layer
    # each node is just a linear combination of the nodes before plus bias term
    # second layer is a linear combination of the nodes in the first layer
    # first layer is just a linear combination of the nodes in the input layer
    # meaning the second layer is just a linear combination of the nodes in the input layer, just a fancy linear regression
    # layer 1 --> activation function --> layer 2
    # apply an activation function: common ones tanh, sigmoid, ReLU, a layer of non-linear combination, add the complexity we need
    # A^[1] = g(Z^[1]) = ReLU(Z^[1]), ReLU: rectified linear unit: ReLU(x) = {x if x>0, 0 if x<= 0}
    # Z^[2] = w^[2] * A^[1] + b^[2]
    # layer 2 --> softmax activation function --> probabilities
    # A^[2] = softmax(Z^[2]), softmax(z) = e^z_i / sum_j=1_K e^z_j 

def forward(X):
    # manual forward prop
    Z1 = W1 * X + b1
    A1 = relu(Z1) # [10, batch_size]
    Z2 = W2 * A1 + b2 # [10, 10]
    A2 = softmax(Z2) # [10, batch_size]
    return Z1, A1, Z2, A2
    
def compute_loss(A2, Y):
    # cross_entropy(A2, Y)
    pass

# machine learning: we learn these weights and biases;
# algorithm that optimizes these weights and biases
# we optimize these weights and biases by running the algorithm over and over again
# back propagation: start with prediction, and find out how much the prediction deviated from the previous error
# how much the 2 layer was off by
# dz^[2] = A^[2] - Y
# dW^[2] = 1/m dZ^[2] A^[1]T
# db^[2] = 1/m sum dz^[2]
# how much the first layer was off by
# dz^[1] = w^[2]T dz^[2] * g' (Z^[1]), g' = derivative of g
# dW^[1] = 1/m dz^[2] X^T
# db^[1] = 1/m sum dz^[1]

def back_prop(self):
    # manually compute gradients
    Z1, A1, Z2, A2 = forward(X)
    m = Y.size(0)
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    # one_hot_Y = F.one_hot(Y, num_classes=A2.size(1)).float() # Convert one_hot_Y to float
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2 @ A1.T
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = self.W2.T @ dZ2.T * (Z1 > 0).float() # Derivative of ReLU is 1 where Z1 > 0
    dW1 = 1 / m * dZ2 * X.T
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

    # update params
    # w^[1] = w^[1] - a dw^[1]
    # b^[1] = b^[1] - a db^[1]
    # w^[2] = w^[2] - a dw^[2]
    # b^[2] = b^[2] - a db^[2]
    # a = {alpha} = learning rate

def update_params(self):
    #Bias b1: Shape: [num_hidden_units, 1], e.g., [10, 1].
    # Gradient db1: Shape: [num_hidden_units], e.g., [10].
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2

def get_predictions(self, A2):
    return np.argmax(A2, dim=1)
    
def get_accuracy(self, predictions, Y):
    correct = np.sum(predictions == Y).item()
    total = Y.size(0)
    return correct / total

def make_predictions(self, X):
    _, _, _, A2 = self.forward(X)
    return self.get_predictions(A2)

def test_predictions(self, index):
    current_image = X_test[index].unsqueeze(0).to(device)  # Add batch dimension
    prediction = self.make_predictions(current_image)
    label = Y_test[index]
    print("Prediction:", prediction.item())
    print("Label:", label.item())
    plt.imshow(current_image.squeeze().cpu(), cmap='gray')
    plt.show()

# call on the methods

# training loop
for i in range(max_iters):
    # sample a batch of data
    X, Y = get_batch(x_train)
    Z1, A1, Z2, A2 = forward(X)
    # compute loss
    loss = compute_loss(A2, Y)
    accuracy = get_accuracy(get_predictions(A2), Y)
    # Backward pass
    dW1, db1, dW2, db2 = back_prop()
    # Update parameters
    update_params(dW1, db1, dW2, db2, learning_rate)
    # Print progress
    if i % eval_interval == 0 or i == (max_iters - 1):
        pass
