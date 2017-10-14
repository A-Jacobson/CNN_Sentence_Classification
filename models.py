import torch.nn.functional as F
from torch import nn
import torch


class MultiKernelConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiKernelConv, self).__init__()
        out_channels //= 3
        self.conv3x1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv4x1 = nn.Conv1d(in_channels, out_channels, kernel_size=4, padding=2)
        self.conv5x1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)

    def forward(self, x):
        f1 = F.relu(self.conv3x1(x))
        f2 = F.relu(self.conv4x1(x)[:, :, :-1])
        f3 = F.relu(self.conv5x1(x))
        return torch.cat([f1, f2, f3], dim=1)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, maxlen,
                 embeddings=None, freeze_embeddings=False):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embeddings is not None:
            self.embedding.weight = nn.Parameter(embeddings)
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False
        self.conv = MultiKernelConv(embedding_dim, 24)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(24 * maxlen//2, 32)
        self.dropout = nn.Dropout()
        self.output = nn.Linear(32, 2)

    def forward(self, x):
        x = self.embedding(x)  # N, word, channel (32, 104, 50)
        x = x.permute(0, 2, 1)  # swap to N, channel, word (32, 50, 104)
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        x = self.output(x)
        return x