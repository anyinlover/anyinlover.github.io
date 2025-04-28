import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, text, seq_length, word2idx):
        self.seq_length = seq_length
        self.word2idx = word2idx
        
        # Convert text to indices
        self.data = []
        for sentence in text:
            indices = [word2idx[word] for word in sentence if word in word2idx]
            if len(indices) >= seq_length + 1:  # +1 for target
                for i in range(len(indices) - seq_length):
                    self.data.append((indices[i:i+seq_length], indices[i+1:i+seq_length+1]))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence, target = self.data[idx]
        return torch.tensor(sequence), torch.tensor(target)

class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(VanillaRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN parameters
        self.hidden_dim = hidden_dim
        self.W_xh = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.W_hh = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        # Initialize hidden state if None
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        embedded = self.embedding(x)
        
        # Process sequence
        outputs = []
        for t in range(seq_length):
            # Current input
            xt = embedded[:, t, :]
            
            # RNN step: h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b)
            hidden = self.tanh(self.W_xh(xt) + self.W_hh(hidden))
            outputs.append(hidden)
        
        # Stack outputs and apply final linear layer
        output = torch.stack(outputs, dim=1)
        output = self.fc(output)
        
        return output, hidden

def train_rnn(text, word2idx, embedding_dim=100, hidden_dim=128, seq_length=20,
              batch_size=32, num_epochs=10, learning_rate=0.001):
    # Create dataset and dataloader
    dataset = TextDataset(text, seq_length, word2idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model and training components
    vocab_size = len(word2idx)
    model = VanillaRNN(vocab_size, embedding_dim, hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        hidden = None
        
        for batch_idx, (sequences, targets) in enumerate(dataloader):
            # Forward pass
            output, hidden = model(sequences, hidden)
            hidden = hidden.detach()  # Detach hidden state from history
            
            # Reshape output and targets for loss calculation
            output = output.view(-1, vocab_size)
            targets = targets.view(-1)
            
            # Calculate loss
            loss = criterion(output, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Gradient clipping
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
    
    return model
