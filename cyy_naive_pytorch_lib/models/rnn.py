import torch
import torch.nn as nn


class SimpleRNN(nn.Module):
    def __init__(self, num_embeddings, num_classes, embedding_dim=250, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        assert torch.equal(output[-1, :, :], hidden.squeeze(0))
        return self.fc(hidden.squeeze(0))
