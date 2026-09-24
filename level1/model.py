import torch
import torch.nn as nn
torch.manual_seed(2023)

class MLP(nn.Module):
    def __init__(self, 
    input_dim=784, 
    hidden_dim=256, 
    output_dim=10,
    dropout=0.1, 
    bias=True):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.act1 = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, output_dim, bias=bias)

    def forward(self, x):
        # x: (B, 1, 28, 28) -> (B, 784)
        x = torch.flatten(x, 1)
        x = self.input_layer(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x  # logits，不要 Softmax