# deep neural network needs optimizing to perform well
# norm: layer normalization, layernorm
# across the batch dimension, 
# any individual neuron, had unit gaussian distribution: 0 mean, 1 std output
# normalizing every single row of the input
# implement layernorm

import torch


class LayerNorm1d:
    
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __call__(self, x):
        # calculate the forward pass
        xmean = x.mean(1, keepdim=True) # batch mean
        xvar = x.var(1, keepdim=True) # batch variance
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        return self.out
        
    def parameters(self):
        return [self.gamma, self.beta]
    
torch.manual_seed(1337)
module = LayerNorm1d(100)
x = torch.randn(32, 100) # batch size 32 of 100-dimensional vectors
x = module(x)
print(x.shape)