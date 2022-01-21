import numpy as np
from turtle import forward
import torch
import torch.nn as nn

class WordEmbedding(nn.Module):
    
    def __init__(
        self,
        embedding_size=300,
        hidden_size=512,
        pretrained_embedding=None,
        device=None
        ):

        super(WordEmbedding, self).__init__()

        self.pretrained_embedding = pretrained_embedding
        self.device = device
        self.linear = nn.Linear(int(embedding_size), int(hidden_size))
        self.batchnorm = nn.BatchNorm1d(num_features=int(hidden_size))
        self.activation = nn.Softmax(dim=2)

    
    def forward(self, tensor):
        if not isinstance(tensor, list):
            tensor = tensor.tolist()
        tensor = [[self.pretrained_embedding[int(x)] for x in row] for row in tensor]
        tensor = np.array(tensor)
        tensor = torch.Tensor(tensor)
        tensor = self.device.data_to_device(tensor)
        tensor = self.linear(tensor).transpose(1,2)
        tensor = self.batchnorm(tensor).transpose(1,2)
        tensor = self.activation(tensor)
        return tensor




                


