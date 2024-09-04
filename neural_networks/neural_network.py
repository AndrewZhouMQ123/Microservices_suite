import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import struct
from array import array
from os.path  import join

# The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
# MNIST Data Loader Class
# directly read data from the binary files
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

# Set file paths based on added MNIST Datasets
input_path = '/Users/andrewzhou/microservices_suite'
training_images_filepath = join(input_path, 'train/train-images.idx3-ubyte')
training_labels_filepath = join(input_path, 'train/train-labels.idx1-ubyte')
test_images_filepath = join(input_path, 'tests/t10k-images.idx3-ubyte')
test_labels_filepath = join(input_path, 'tests/t10k-labels.idx1-ubyte')

# Load MINST dataset
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_iters = 4000
eval_interval = 300
learning_rate = 1e-2

# preprocess the data

def preprocess_images(images):
    # Flatten the images from (28, 28) to (784,) and convert to tensor
    return torch.tensor(images.reshape(images.shape[0], -1), dtype=torch.float32)

# Preprocess the training and test images
X_train = preprocess_images(x_train)
X_test = preprocess_images(x_test)

# Convert labels to tensors
def preprocess_labels(labels):
    return torch.tensor(labels, dtype=torch.long)

# Preprocess the training and test labels
Y_train = preprocess_labels(y_train)
Y_test = preprocess_labels(y_test)

# Split training data into training and development sets
n = int(0.9 * len(X_train))
train_data = X_train[:n]
dev_data = X_train[n:]
train_label = Y_train[:n]
dev_label = Y_train[n:]

# Create TensorDataset
train_dataset = TensorDataset(train_data, train_label)
dev_dataset = TensorDataset(dev_data, dev_label)
test_dataset = TensorDataset(X_test, Y_test)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def get_batch(split):
    if split == 'train':
        loader = train_loader
    elif split == 'val':
        loader = dev_loader
    else:
        raise ValueError("Split must be 'train' or 'val'")
    
    # Get the next batch from the DataLoader
    data_iter = iter(loader)
    X, Y = next(data_iter)
    
    return X, Y

class SimpleNeuralNetwork(nn.Module):
    def __init__(self):
        super(SimpleNeuralNetwork, self).__init__()
        self.input_size = 784
        self.hidden_size = 10
        self.num_classes = 10 # For digit classification (0-9)

        # Define layers with nn.Linear for simplicity
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, X):
        Z1 = self.fc1(X)
        A1 = F.relu(Z1)
        Z2 = self.fc2(A1)
        A2 = F.softmax(Z2, dim=1)
        return Z1, A1, Z2, A2
    
    def compute_loss(self, A2, Y):
        return F.cross_entropy(A2, Y)

    def get_predictions(self, A2):
        return torch.argmax(A2, dim=1)
    
    def get_accuracy(self, predictions, Y):
        correct = torch.sum(predictions == Y).item()
        total = Y.size(0)
        return correct / total
    
    def make_predictions(self, X):
        _, _, _, A2 = self.forward(X)
        return self.get_predictions(A2)

    def test_predictions(self, index):
        # Extract the image and label
        current_image = X_test[index].unsqueeze(0).to(device)  # Add batch dimension and move to device
        label = Y_test[index]
        
        # Make prediction
        prediction = self.make_predictions(current_image)
        
        # Print prediction and label
        print(f"Prediction: {prediction.item()}")
        print(f"Label: {label.item()}")
        
        # Convert the image to its original shape (28x28) and move to CPU
        current_image = current_image.squeeze().cpu().numpy()
        
        # Display the image with its prediction
        plt.imshow(current_image.reshape(28, 28), cmap='gray')  # Reshape to 28x28
        plt.title(f"Prediction: {prediction.item()}, Label: {label.item()}")
        plt.show()

# Create model instance
model = SimpleNeuralNetwork().to(device)
# Define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training loop
for i in range(max_iters):
    model.train()
    # sample a batch of data
    X, Y = get_batch('train')
    X, Y = X.to(device), Y.to(device)
    # Forward pass
    # nn.Module handles such that this implicitly calls forward method
    # This is the preferred and conventional way to invoke the forward pass in PyTorch.
    # It calls the forward method of the model automatically.
    Z1, A1, Z2, A2 = model(X) 
    # compute loss
    loss = model.compute_loss(A2, Y)
    accuracy = model.get_accuracy(model.get_predictions(A2), Y)
    # Zero out gradients
    optimizer.zero_grad()  
    # Backward pass
    loss.backward()  # Compute gradients  
    # Update parameters
    optimizer.step()  # Apply gradients to parameters
    # Print progress
    if i % eval_interval == 0 or i == (max_iters - 1):
        model.eval()
        with torch.no_grad():
            dev_loss = 0.0
            dev_accuracy = 0.0
            for X_dev, Y_dev in dev_loader:
                X_dev, Y_dev = X_dev.to(device), Y_dev.to(device)
                _, _, _, A2_dev = model(X_dev)
                dev_loss += model.compute_loss(A2_dev, Y_dev).item()
                dev_accuracy += model.get_accuracy(model.get_predictions(A2_dev), Y_dev)
            
            dev_loss /= len(dev_loader)
            dev_accuracy /= len(dev_loader)
            
            print(f"Step {i}: Train loss {loss.item():.4f}, accuracy {accuracy:.4f}, Dev loss {dev_loss:.4f}, Dev accuracy {dev_accuracy:.4f}")

# Select indices of samples you want to test
sample_indices = [0, 10, 25, 50, 100]  # Replace with any indices you want to visualize

for idx in sample_indices:
    model.test_predictions(idx)