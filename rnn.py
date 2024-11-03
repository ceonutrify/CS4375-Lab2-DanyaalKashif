import numpy as np
import torch
import os 
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import StepLR
import random
import torch.optim as optim
import time
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle

# Constants
unk = '<UNK>'

# Define the Recurrent Neural Network (RNN) class
class RNN(nn.Module):
    def __init__(self, input_dim, h, num_layers=1):
        super(RNN, self).__init__()
        self.h = h
        self.num_layers = num_layers
        self.output_dim = 5  # Fixed value as per assignment requirements (5-class output)

        # Define the RNN layer with tanh activation and specified input/output dimensions
        self.rnn = nn.RNN(input_dim, h, num_layers=self.num_layers, batch_first=True, nonlinearity='tanh')
        
        # Output layer that maps RNN output to the 5-class output space
        self.W = nn.Linear(h, self.output_dim)
        
        # Softmax layer and loss function for training
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        # Calculates the negative log-likelihood loss
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs, lengths):
        # Forward pass for RNN model
        # Inputs: Padded sequences of shape (batch_size, seq_len, input_dim)
        # Lengths: Original lengths of sequences before padding for each example in the batch
        
        # Pack the padded sequences for efficient processing in the RNN
        packed_inputs = pack_padded_sequence(inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # Pass through the RNN layer
        packed_output, hidden = self.rnn(packed_inputs)
        
        # Unpack the output back to padded form for consistency with input dimensions
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Creating a mask for valid time steps (to avoid padded regions)
        batch_size = inputs.size(0)
        max_seq_len = inputs.size(1)
        mask = torch.arange(max_seq_len).expand(batch_size, max_seq_len).to(inputs.device) < lengths.unsqueeze(1)
        
        # Masking the padded positions in the output and summing over valid time steps
        masked_output = output * mask.unsqueeze(2)
        sum_output = torch.sum(masked_output, dim=1)
        
        # Linear layer maps summed output to class space
        zi = self.W(sum_output)
        
        # LogSoftmax for class probabilities
        predicted_vector = self.softmax(zi)
        
        return predicted_vector

# Function to load data from JSON files for training and validation
def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(), int(elt["stars"] - 1)))
    for elt in validation:
        val.append((elt["text"].split(), int(elt["stars"] - 1)))
    return tra, val

# Function to preprocess data by converting text into embeddings
def preprocess_data(data, word_embedding):
    embedding_dim = next(iter(word_embedding.values())).shape[0]
    default_vector = np.zeros(embedding_dim, dtype=np.float32)

    processed_data = []
    for input_words, gold_label in data:
        # Clean text: remove punctuation and split into words
        input_words = " ".join(input_words)
        input_words = input_words.translate(str.maketrans("", "", string.punctuation)).split()

        # Convert words to embeddings, using zero vector for unknown words
        vectors = [word_embedding.get(word.lower(), default_vector) for word in input_words]
        vectors = np.array(vectors, dtype=np.float32)
        vectors = torch.from_numpy(vectors)  # Convert to PyTorch tensor

        processed_data.append((vectors, gold_label))
    return processed_data

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden dimension size")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="number of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--note", default="", help="Notes about code changes or hyperparameters")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01, help="learning rate")
    parser.add_argument("-bs", "--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay (L2 regularization)")
    parser.add_argument("--num_layers", type=int, default=1, help="number of RNN layers")
    args = parser.parse_args()

    # Saving hyperparameters to a dictionary for easy tracking and storage
    hyperparameters = {
        'hidden_dim': args.hidden_dim,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'weight_decay': args.weight_decay,
        'num_layers': args.num_layers,
        'note': args.note
    }

    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    # Check if MPS backend is available for Apple Silicon GPU acceleration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load and preprocess data
    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data)
    print("========== Loading word embeddings ==========")
    word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))
    print("========== Preprocessing data ==========")
    train_data = preprocess_data(train_data, word_embedding)
    valid_data = preprocess_data(valid_data, word_embedding)

    # Initialize model and move it to the appropriate device
    model = RNN(input_dim=50, h=args.hidden_dim, num_layers=args.num_layers)
    model.to(device)

    # Initialize optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)  # Step down LR every 5 epochs

    results = []  # To store results for each epoch

    print(f"========== Training for {args.epochs} epochs ==========")

    for epoch in range(args.epochs):
        model.train()
        correct = 0
        total = 0
        start_time = time.time()
        print(f"Training started for epoch {epoch + 1}")

        # Shuffle data at each epoch to ensure randomness
        random.shuffle(train_data)
        minibatch_size = args.batch_size
        N = len(train_data)
        num_batches = N // minibatch_size
        
        for batch_index in tqdm(range(num_batches)):
            optimizer.zero_grad()  # Zero gradients for each minibatch

            # Get current batch
            batch_data = train_data[batch_index * minibatch_size:(batch_index + 1) * minibatch_size]
            
            # Sort by sequence length for efficient RNN processing
            batch_data.sort(key=lambda x: len(x[0]), reverse=True)
            
            # Extract sequences and labels
            sequences = [item[0] for item in batch_data]
            labels = torch.tensor([item[1] for item in batch_data], dtype=torch.long)

            # Calculate sequence lengths
            lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)

            # Pad sequences to the max length in the batch
            padded_sequences = pad_sequence(sequences, batch_first=True)

            # Move inputs to device
            padded_sequences = padded_sequences.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            # Forward pass
            predicted_vectors = model(padded_sequences, lengths)

            # Compute loss and perform backpropagation
            loss = model.compute_Loss(predicted_vectors, labels)
            loss.backward()
            optimizer.step()

            # Calculate training accuracy
            predicted_labels = torch.argmax(predicted_vectors, dim=1)
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)

        train_accuracy = correct / total
        training_time = time.time() - start_time
        print(f"Training accuracy for epoch {epoch + 1}: {train_accuracy}")
        print(f"Training time for this epoch: {training_time}")

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        start_time = time.time()
        print(f"Validation started for epoch {epoch + 1}")

        with torch.no_grad():
            num_batches_val = len(valid_data) // minibatch_size
            for batch_index in tqdm(range(num_batches_val)):
                batch_data = valid_data[batch_index * minibatch_size:(batch_index + 1) * minibatch_size]
                batch_data.sort(key=lambda x: len(x[0]), reverse=True)

                sequences = [item[0] for item in batch_data]
                labels = torch.tensor([item[1] for item in batch_data], dtype=torch.long)

                lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
                padded_sequences = pad_sequence(sequences, batch_first=True)

                padded_sequences = padded_sequences.to(device)
                labels = labels.to(device)
                lengths = lengths.to(device)

                predicted_vectors = model(padded_sequences, lengths)
                predicted_labels = torch.argmax(predicted_vectors, dim=1)
                val_correct += (predicted_labels == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = val_correct / val_total
        validation_time = time.time() - start_time
        print(f"Validation accuracy for epoch {epoch + 1}: {val_accuracy}")
        print(f"Validation time for this epoch: {validation_time}")

        current_learning_rate = scheduler.get_last_lr()[0]
        scheduler.step()

        epoch_result = {
            'epoch': epoch + 1,
            'learning_rate': current_learning_rate,
            'training_accuracy': train_accuracy,
            'training_time': training_time,
            'validation_accuracy': val_accuracy,
            'validation_time': validation_time
        }
        results.append(epoch_result)

    # Save results to file
    run_result = {
        'hyperparameters': hyperparameters,
        'results': results
    }
    if os.path.exists('rnnTestsOutput.json'):
        with open('rnnTestsOutput.json', 'r') as f:
            all_results = json.load(f)
    else:
        all_results = []
    all_results.append(run_result)
    with open('rnnTestsOutput.json', 'w') as f:
        json.dump(all_results, f, indent=4)
