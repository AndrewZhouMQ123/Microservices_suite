import torch # we use PyTorch
import torch.nn as nn
from torch.nn import functional as F

# version 4 self-attention

torch.manual_seed(1337)
B,T,C = 4,8,32
x = torch.randn(B,T,C)

# let's see a single Head perform self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)  # Define the value linear transformation

k = key(x) # (B, T, 16)
q = query(x) # (B, T, 16)
v = value(x)

# Compute raw attention scores
wei = q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) --> (B, T, T)

# Create lower triangular matrix for masking
tril = torch.tril(torch.ones(T, T))
# wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))

# Apply softmax to get attention weights
wei = F.softmax(wei, dim=-1)

# Compute the output of the self-attention head
out = wei @ v # (B, T, T) @ (B, T, 16) --> (B, T, 16)
# out = wei @ x

# Output shape
print(out.shape)  # Should output: torch.Size([4, 8, 16])