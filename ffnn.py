import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import math
import random
import os
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser

# Define an unknown token for words not found in vocab
unk = '<UNK>'

# Define the Feedforward Neural Network (FFNN) class
class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        
        # Define model parameters and layers
        self.h = h
        self.output_dim = 5  # Fixed value as per assignment requirements (5-class output)
        
        # Define the layers for the FFNN
        self.W1 = nn.Linear(input_dim, h*2)  # First linear transformation with double hidden layer size
        self.W2 = nn.Linear(h*2, h)          # Second transformation down to hidden layer size
        self.W3 = nn.Linear(h, self.output_dim)  # Third transformation to the output layer size
        
        # Other layers and functions used within the network
        self.dropout = nn.Dropout(0.5)       # Dropout for regularization #adjusted to 0.5 for experiment was at 0.2
        self.activation = nn.ReLU()          # ReLU activation function
        self.softmax = nn.LogSoftmax(dim=1)  # LogSoftmax for output (to convert to log probabilities)
        self.loss = nn.NLLLoss()             # Negative Log-Likelihood Loss for classification

    def compute_Loss(self, predicted_vector, gold_label):
        # Computes the loss between the predicted vector and the gold label
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # This function handles the forward pass through the network
        
        # Ensure input has correct dimensions for processing
        if input_vector.dim() == 1:
            input_vector = input_vector.unsqueeze(0)
        
        # First hidden layer
        hidden = self.W1(input_vector)   # Apply linear transformation
        hidden = self.activation(hidden) # Apply ReLU activation
        hidden = self.dropout(hidden)    # Apply dropout
        
        # Second hidden layer
        hidden = self.W2(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        
        # Output layer
        output = self.W3(hidden)
        
        # Apply LogSoftmax to get probability distribution over output classes
        predicted_vector = self.softmax(output)
        
        return predicted_vector


# Function to create vocabulary from the dataset
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)  # Add each word in the document to the vocabulary set
    return vocab


# Function to generate word indices for vocabulary lookup
def make_indices(vocab):
    vocab_list = sorted(vocab)  # Sort vocabulary for consistent ordering
    vocab_list.append(unk)      # Add unknown token to vocabulary list
    
    word2index = {}  # Maps words to their indices
    index2word = {}  # Maps indices back to words
    
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    
    vocab.add(unk)  # Ensure <UNK> is part of the vocab set
    
    return vocab, word2index, index2word


# Function to vectorize the dataset based on the generated vocabulary
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index), dtype=torch.float32)  # Set dtype to float32 for MPS compatibility
        for word in document:
            index = word2index.get(word, word2index[unk])  # Get index of word, or <UNK> if not in vocab
            vector[index] += 1  # Increment count for word in vector
        vectorized_data.append((vector, y))  # Append vector and label to dataset
    return vectorized_data


# Function to load data from JSON files for training and validation
def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(), int(elt["stars"]-1)))  # Process training data
    for elt in validation:
        val.append((elt["text"].split(), int(elt["stars"]-1)))  # Process validation data

    return tra, val


# Main function to run the model
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", default="to fill", help="path to test data")
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument("--note", default="", help="Notes about code changes or hyperparameters")  # New argument for code notes
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01, help="learning rate") #to adjust learning rate
    parser.add_argument("-bs", "--batch_size", type=int, default=16, help="batch size") #to adjust batch size  
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay (L2 regularization)") #weight decay

    args = parser.parse_args()

    # Save hyperparameters and any additional notes
    hyperparameters = {
        'hidden_dim': args.hidden_dim,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'momentum': 0.9,  # Remove this if using Adam
        'batch_size': args.batch_size,
        'weight_decay': args.weight_decay,
        'note': args.note
}


    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    # Detect device (MPS for Apple Silicon GPU, CPU otherwise)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")  # Output to terminal which device is being used

    # Load data from specified file paths
    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data)
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    # Vectorize the training and validation data
    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)

    # Initialize model and move it to the appropriate device
    model = FFNN(input_dim=len(vocab), h=args.hidden_dim)
    model.to(device)  # Move model to device (CPU or MPS)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay) #updated for weight decay
    #optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


    results = []  # Initialize an empty list to store results for each epoch

    print("========== Training for {} epochs ==========".format(args.epochs))
    
    # Training loop
    for epoch in range(args.epochs):
        # Track training accuracy and time
        model.train()
        correct, total, start_time = 0, 0, time.time()
        
        # Shuffle data at the start of each epoch
        random.shuffle(train_data)
        minibatch_size = args.batch_size  # Use args.batch_size instead of hardcoded value
        N = len(train_data)
        
        for minibatch_index in tqdm(range(N // minibatch_size)):
            # Collect inputs and labels for the minibatch
            inputs = []
            labels = []
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                inputs.append(input_vector)
                labels.append(gold_label)

            # Stack inputs and labels into tensors and move them to the device
            inputs = torch.stack(inputs).to(device)  # Move inputs to device
            labels = torch.tensor(labels, dtype=torch.long).to(device)  # Move labels to device

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            predicted_vectors = model(inputs)

            # Compute loss
            loss = model.compute_Loss(predicted_vectors, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Compute accuracy
            predicted_labels = torch.argmax(predicted_vectors, dim=1)
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)
        
        # Calculate training accuracy and time taken for the epoch
        train_accuracy = correct / total
        training_time = time.time() - start_time

        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, train_accuracy))
        print("Training time for this epoch: {}".format(training_time))

        # Validation loop
        model.eval()
        val_correct, val_total, start_time = 0, 0, time.time()
        
        with torch.no_grad():  # Disable gradient calculation for validation
            for minibatch_index in tqdm(range(len(valid_data) // minibatch_size)):
                # Collect inputs and labels for the minibatch
                inputs = []
                labels = []
                for example_index in range(minibatch_size):
                    input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                    inputs.append(input_vector)
                    labels.append(gold_label)

                # Stack inputs and labels into tensors and move them to the device
                inputs = torch.stack(inputs).to(device)  # Move inputs to device
                labels = torch.tensor(labels, dtype=torch.long).to(device)  # Move labels to device

                # Forward pass
                predicted_vectors = model(inputs)

                # Compute loss (not used here, but kept for completeness)
                loss = model.compute_Loss(predicted_vectors, labels)

                # Compute accuracy
                predicted_labels = torch.argmax(predicted_vectors, dim=1)
                val_correct += (predicted_labels == labels).sum().item()
                val_total += labels.size(0)
        
        # Calculate validation accuracy and time taken for the epoch
        val_accuracy = val_correct / val_total
        validation_time = time.time() - start_time

        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, val_accuracy))
        print("Validation time for this epoch: {}".format(validation_time))

        # Save epoch results
        epoch_result = {
            'epoch': epoch + 1,
            'training_accuracy': train_accuracy,
            'training_time': training_time,
            'validation_accuracy': val_accuracy,
            'validation_time': validation_time
        }
        results.append(epoch_result)

    # Save all results, hyperparameters, and code notes to ffnnTestsOutput.json
    run_result = {
        'hyperparameters': hyperparameters,
        'results': results
    }

    if os.path.exists('ffnnTestsOutput.json'):
        # Load existing results if the file exists
        with open('ffnnTestsOutput.json', 'r') as f:
            all_results = json.load(f)
    else:
        all_results = []

    # Append current run results to all results and save back to file
    all_results.append(run_result)
    with open('ffnnTestsOutput.json', 'w') as f:
        json.dump(all_results, f, indent=4)
