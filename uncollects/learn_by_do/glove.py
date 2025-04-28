import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict, Counter
import math

class GloveDataset(Dataset):
    def __init__(self, text, window_size=10, min_count=5):
        # Build co-occurrence matrix
        self.word_counts = Counter()
        for sentence in text:
            self.word_counts.update(sentence)
        
        # Filter vocabulary by minimum count
        self.vocab = {word for word, count in self.word_counts.items() if count >= min_count}
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        # Build co-occurrence matrix
        self.cooccur = defaultdict(float)
        for sentence in text:
            word_ids = [self.word2idx[word] for word in sentence if word in self.vocab]
            for center_pos in range(len(word_ids)):
                for context_pos in range(max(0, center_pos - window_size), 
                                      min(len(word_ids), center_pos + window_size + 1)):
                    if center_pos != context_pos:
                        distance = abs(context_pos - center_pos)
                        increment = 1.0 / distance
                        i_idx = word_ids[center_pos]
                        j_idx = word_ids[context_pos]
                        self.cooccur[(i_idx, j_idx)] += increment
        
        self.data = [(i, j, count) for (i, j), count in self.cooccur.items()]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        i, j, count = self.data[idx]
        return (torch.tensor(i), torch.tensor(j), 
                torch.tensor(math.log(count), dtype=torch.float32))

class GloveModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(GloveModel, self).__init__()
        self.wi = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.wj = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.bi = nn.Embedding(vocab_size, 1, sparse=True)
        self.bj = nn.Embedding(vocab_size, 1, sparse=True)
        
        # Initialize embeddings
        self.wi.weight.data.uniform_(-1, 1)
        self.wj.weight.data.uniform_(-1, 1)
        self.bi.weight.data.zero_()
        self.bj.weight.data.zero_()
    
    def forward(self, i_indices, j_indices):
        w_i = self.wi(i_indices)
        w_j = self.wj(j_indices)
        b_i = self.bi(i_indices).squeeze()
        b_j = self.bj(j_indices).squeeze()
        
        return torch.sum(w_i * w_j, dim=1) + b_i + b_j

def train_glove(text, embedding_dim=100, window_size=10, min_count=5,
                batch_size=512, num_epochs=25, learning_rate=0.05):
    
    # Create dataset and dataloader
    dataset = GloveDataset(text, window_size, min_count)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model and optimizer
    model = GloveModel(len(dataset.vocab), embedding_dim)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (i_indices, j_indices, log_cooccur) in enumerate(dataloader):
            # Forward pass
            log_pred = model(i_indices, j_indices)
            
            # GloVe weighting function
            weight_factor = torch.clamp(torch.pow(log_cooccur/100, 0.75), max=1.0)
            
            # Loss calculation
            loss = weight_factor * torch.pow(log_pred - log_cooccur, 2).mean()
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
    
    # Final embeddings are average of wi and wj
    final_embeddings = (model.wi.weight.data + model.wj.weight.data) / 2
    return final_embeddings, dataset.word2idx, dataset.idx2word
