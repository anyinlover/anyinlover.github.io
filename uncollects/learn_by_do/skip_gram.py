import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import random

class SkipGramDataset(Dataset):
    def __init__(self, text, window_size=2, n_negatives=5, min_count=5):
        # Create vocabulary and mappings
        words = text.lower().split()  # Convert to lowercase
        word_counts = Counter(words)
        
        # Filter out rare words and punctuation
        punctuation = set('.,!?;:()[]{}""\'\'``')
        self.word_counts = {word: count for word, count in word_counts.items()
                         if count >= min_count  # Remove rare words
                         and not any(p in word for p in punctuation)  # Remove words with punctuation
                         and len(word) > 1}  # Remove single characters
        
        self.vocab = list(self.word_counts.keys())
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for i, w in enumerate(self.vocab)}
        
        # Filter the input words
        self.valid_words = [word for word in words if word in self.word2idx]
        
        # Create training pairs
        self.data = []
        for i, word in enumerate(words):
            # Get positive context words within window
            window_start = max(0, i - window_size)
            window_end = min(len(words), i + window_size + 1)
            
            for j in range(window_start, window_end):
                if i != j:
                    self.data.append((self.word2idx[word], self.word2idx[words[j]], 1))
                    
                    # Add negative samples
                    neg_samples = self._get_negative_samples(word, n_negatives)
                    for neg_word in neg_samples:
                        self.data.append((self.word2idx[word], self.word2idx[neg_word], 0))
    
    def _get_negative_samples(self, target_word, n_samples):
        # Calculate probability distribution for negative sampling
        total_words = sum(self.word_counts.values())
        word_probs = {word: (count/total_words)**0.75 for word, count in self.word_counts.items()}
        
        # Normalize probabilities
        prob_sum = sum(word_probs.values())
        word_probs = {word: prob/prob_sum for word, prob in word_probs.items()}
        
        # Convert to list for random.choices
        words, probs = zip(*word_probs.items())
        
        # Sample negative words according to distribution
        neg_samples = []
        while len(neg_samples) < n_samples:
            neg_word = random.choices(words, weights=probs, k=1)[0]
            if neg_word != target_word:
                neg_samples.append(neg_word)
        return neg_samples
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        center, context, label = self.data[idx]
        return torch.tensor(center), torch.tensor(context), torch.tensor(label, dtype=torch.float32)

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        
        # Initialize embeddings
        self.center_embeddings.weight.data.uniform_(-0.5/embedding_dim, 0.5/embedding_dim)
        self.context_embeddings.weight.data.uniform_(-0.5/embedding_dim, 0.5/embedding_dim)
    
    def forward(self, center_words, context_words):
        center_embeds = self.center_embeddings(center_words)
        context_embeds = self.context_embeddings(context_words)
        
        # Compute dot product between center and context embeddings
        dot_product = torch.sum(center_embeds * context_embeds, dim=1)
        return torch.sigmoid(dot_product)

def train_skip_gram(text, embedding_dim=100, window_size=2, n_negatives=5, 
                   batch_size=256, num_epochs=5, learning_rate=0.001):
    # Create dataset and dataloader
    dataset = SkipGramDataset(text, window_size, n_negatives)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model and optimizer
    model = SkipGram(len(dataset.vocab), embedding_dim)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (center, context, label) in enumerate(dataloader):
            # Forward pass
            output = model(center, context)
            loss = criterion(output, label)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
    
    return model, dataset.word2idx, dataset.idx2word
