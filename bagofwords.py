import torch # we use PyTorch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(42)
# the mathematical trick in self-attention
# consider the following toy example:

B,T,C = 4,8,2 # batch, time, channels
x = torch.randn(B,T,C)
x.shape

# version 1
# we want x[b, t] = mean_{i<=t} x [b,i]
xbow = torch.zeros((B,T,C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1] # (t,C)
        xbow[b,t] = torch.mean(xprev, 0)

# more efficient using matrix multiplication
a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0, 10, (3,2)).float()
c = a @ b
print("a=")
print(a)
print("b=")
print(b)
print("c=")
print(c)

# version 2 matrix multiply
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x # (B, T, T) @ (B, T, C) --> (B, T, C)
print(torch.allclose(xbow, xbow2))

# version 3 softmax
tril = torch.tril(torch.ones(T,T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
print(torch.allclose(xbow, xbow3))