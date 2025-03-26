import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.nn.utils.rnn import pad_sequence  # for padded inputs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

# ----------------------------
# Data Loading
# ----------------------------
def loadData():
    X_train = np.load('data/X_train.npy', allow_pickle=True)
    y_train = np.load('data/y_train.npy', allow_pickle=True)
    X_test = np.load('data/X_test.npy', allow_pickle=True)
    y_test = np.load('data/y_test.npy', allow_pickle=True)

    # Convert each sequence into a Torch tensor.
    X_train = [torch.Tensor(x).to(device) for x in X_train]  # List of Tensors, each (seq_len, input_size)
    X_test = [torch.Tensor(x).to(device) for x in X_test]
    y_train = torch.Tensor(y_train).to(device)  # (num_samples, 1)
    y_test = torch.Tensor(y_test).to(device)   # (num_samples, 1)

    return X_train, X_test, y_train, y_test

# ----------------------------
# Vanilla RNN Layer with Shared Weights
# ----------------------------
class RNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNLayer, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        # Linear transform for input -> hidden (no bias) 
        self.W_xh = nn.Linear(input_size, hidden_size, bias=False)
        # Linear transform for hidden -> hidden (with bias)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=True)
        self.activation = torch.tanh

    def forward(self, x, hidden):
        # Compute the next hidden state: tanh(W_xh*x + W_hh*hidden)
        hidden = self.activation(self.W_xh(x) + self.W_hh(hidden))
        return hidden

# ----------------------------
# Vanilla RNN based Model (Weights Shared Across Time)
# ----------------------------
class SequenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SequenceModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = RNNLayer(input_size, hidden_size)  # same weights for every time step
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, seq_lengths):
        # input_seq: list of tensors of shape (seq_len, input_size)
        batch_size = len(input_seq)
        last_hidden = torch.zeros(batch_size, self.hidden_size).to(device)

        for b in range(batch_size):
            hidden = torch.zeros(self.hidden_size).to(device)
            seq_length = int(seq_lengths[b])
            for t in range(seq_length):
                hidden = self.rnn(input_seq[b][t], hidden)
            last_hidden[b] = hidden

        output = self.linear(last_hidden)
        return output

# ----------------------------
# Fixed-Length Non-Shared Weights Model (Truncated to min length)
# ----------------------------
class SequenceModelFixedLen(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_len):
        super(SequenceModelFixedLen, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len  # fixed (truncated) sequence length
        # Create a separate RNN layer for each time step (non-shared weights)
        self.rnn_layers = nn.ModuleList([RNNLayer(input_size, hidden_size) for _ in range(seq_len)])
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, seq_lengths):
        # input_seq: list of tensors, where each tensor is (seq_len, input_size)
        batch_size = len(input_seq)
        last_hidden = torch.zeros(batch_size, self.hidden_size).to(device)

        for b in range(batch_size):
            hidden = torch.zeros(self.hidden_size).to(device)
            # Use the minimum of self.seq_len and the actual sequence length.
            seq_length = min(self.seq_len, int(seq_lengths[b]))
            for t in range(seq_length):
                hidden = self.rnn_layers[t](input_seq[b][t], hidden)
            last_hidden[b] = hidden

        output = self.linear(last_hidden)
        return output

# ----------------------------
# Padded Sequences Model (Non-Shared Weights, with Masking)
# ----------------------------
class SequenceModelPadded(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_len):
        super(SequenceModelPadded, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len  # maximum sequence length after padding
        self.rnn_layers = nn.ModuleList([RNNLayer(input_size, hidden_size) for _ in range(seq_len)])
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, seq_lengths):
        """
        input_seq: tensor of shape (batch_size, max_seq_len, input_size)
        seq_lengths: list of actual sequence lengths per sample.
        """
        batch_size = input_seq.size(0)
        hidden = torch.zeros(batch_size, self.hidden_size).to(device)

        for t in range(self.seq_len):
            # Build a mask for which sequences have a valid time step t
            mask = torch.tensor([1 if t < l else 0 for l in seq_lengths],
                                dtype=torch.float32).to(device)  # shape: (batch_size,)
            mask = mask.unsqueeze(1)  # shape: (batch_size, 1)
            # Process the t-th time step for the whole batch
            current_input = input_seq[:, t, :]  # (batch_size, input_size)
            new_hidden = self.rnn_layers[t](current_input, hidden)
            # Only update hidden for valid (non-padded) steps
            hidden = mask * new_hidden + (1 - mask) * hidden

        output = self.linear(hidden)
        return output

# ----------------------------
# Hyperparameters and Device Setup
# ----------------------------
input_size = 10   # Dimension of each input vector
hidden_size = 64
output_size = 1
num_epochs = 10
learning_rate = 0.001
batch_size = 32

# ----------------------------
# Setup Data and Device
# ----------------------------
X_train, X_test, y_train, y_test = loadData()
# Compute individual sequence lengths from the training set
seq_lengths_train = [x.shape[0] for x in X_train]

# ----------------------------
# Training Loop (for list-of-tensors models)
# ----------------------------
def train(model, num_epochs, lr, batch_size, X_train, y_train, seq_lengths):
    print('\n')
    print(f'Training for model: {model.__class__.__name__}')
    # Mean Squared Error loss for a regression task.
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataset_size = len(X_train)
    for epoch in range(num_epochs):
        # Shuffle the indices
        permutation = list(range(dataset_size))
        random.shuffle(permutation)
        for i in range(0, dataset_size, batch_size):
            indices = permutation[i:i+batch_size]
            batch_X = [X_train[j] for j in indices]
            batch_y = y_train[indices]
            batch_lengths = [seq_lengths[j] for j in indices]

            optimizer.zero_grad()
            outputs = model(batch_X, batch_lengths)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        print("Epoch: {} Loss: {:.4f}".format(epoch, loss.item()))
    return model

# ----------------------------
# Training Loop for Padded-Input Model
# ----------------------------
def train_padded(model, num_epochs, lr, batch_size, X_train_padded, y_train, seq_lengths):
    # X_train_padded is a tensor of shape (num_samples, max_seq_len, input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataset_size = X_train_padded.size(0)
    for epoch in range(num_epochs):
        permutation = torch.randperm(dataset_size)
        for i in range(0, dataset_size, batch_size):
            indices = permutation[i:i+batch_size]
            batch_X = X_train_padded[indices]
            batch_y = y_train[indices]
            # Collect the actual lengths for this batch.
            batch_lengths = [seq_lengths[j] for j in indices.tolist()]
            optimizer.zero_grad()
            outputs = model(batch_X, batch_lengths)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        print("Epoch: {} Loss: {:.4f}".format(epoch, loss.item()))
    return model

# ----------------------------
# Model Initialization and Training
# ----------------------------

# (A) Vanilla RNN (weight shared)
model_vanilla = SequenceModel(input_size, hidden_size, output_size).to(device)
model_vanilla = train(model_vanilla, num_epochs, learning_rate, batch_size, X_train, y_train, seq_lengths_train)

# (B) Sequential NN with Truncated Sequences (non-shared weights)
# Truncate every sequence to the length of the shortest sequence.
min_seq_len = min(seq_lengths_train)
X_train_truncated = [x[:min_seq_len, :] for x in X_train]
X_test_truncated = [x[:min_seq_len, :] for x in X_test]
model_trunc = SequenceModelFixedLen(input_size, hidden_size, output_size, min_seq_len).to(device)
# Since all sequences are now of length min_seq_len we can pass that as length.
model_trunc = train(model_trunc, num_epochs, learning_rate, batch_size, 
                    X_train_truncated, y_train, [min_seq_len] * len(X_train_truncated))

# (C) Sequential NN with Padded Sequences (non-shared weights + mask)
# Pad all sequences to the maximum sequence length in the training set.
max_seq_len = max(seq_lengths_train)
X_train_padded = pad_sequence(X_train, batch_first=True)  # shape: (num_samples, max_seq_len, input_size)
X_test_padded = pad_sequence(X_test, batch_first=True)
model_padded = SequenceModelPadded(input_size, hidden_size, output_size, max_seq_len).to(device)
model_padded = train_padded(model_padded, num_epochs, learning_rate, batch_size, 
                            X_train_padded, y_train, seq_lengths_train)

# ----------------------------
# (Optional) Evaluation on Test Data
# ----------------------------
def evaluate(model, X, y, seq_lengths):
    model.eval()
    with torch.no_grad():
        outputs = model(X, seq_lengths)
        loss = nn.MSELoss()(outputs, y)
    print("Test Loss: {:.4f}".format(loss.item()))
    return loss.item()

# For evaluation on vanilla and truncated variants (list-of-tensors based):
seq_lengths_test = [x.shape[0] for x in X_test]
print("\nEvaluating Vanilla RNN on test set:")
evaluate(model_vanilla, X_test, y_test, seq_lengths_test)

print("\nEvaluating Truncated Model on test set:")
evaluate(model_trunc, X_test_truncated, y_test, [min_seq_len] * len(X_test_truncated))

# For padded model, evaluation must be done with padded inputs:
seq_lengths_test = [x.shape[0] for x in X_test]
print("\nEvaluating Padded Model on test set:")
evaluate(model_padded, X_test_padded, y_test, seq_lengths_test)