import torch
import torch.nn as nn

class SOM(nn.Module):
    def __init__(
        self,
        latt_size,
        latt_shape,
        n,
        sigma=1,
        lr=1e-1,
    ):
        '''Self-Organizing Map (SOM).
        
        Args:
            latt_size:  tuple with size of lattice.
            latt_shape: str with shape of lattice. Supports 'circle1d', 'line1d', 'square2d' and 'parallelogram2d'.
            n:          int with dimension of the data.
            sigma:      float with the sigma parameter of the neighborhood function.
            lr:         float with the learning rate.
        '''
        super(SOM, self).__init__()

        self.latt_size = latt_size
        self.latt_shape = latt_shape
        
        if self.latt_shape == 'line1d':
            self.num_neurons = latt_size[0]
            
        if self.latt_shape == 'circle1d':
            self.num_neurons = latt_size[0]
        
        if self.latt_shape == 'square2d':
            self.num_neurons = latt_size[0] * latt_size[1]
            
        if self.latt_shape == 'parallelogram2d':
            self.num_neurons = latt_size[0] * latt_size[1] * 2
            
        self.sigma = sigma
        self.lr = lr
        
        self.latt_dict = self.latt_gen()
        self.nhb_dists = self.latt_dists()
        
        self.W = torch.randn(self.num_neurons, n)
        
    def latt_gen(self):
        '''Generates a dictionary representing the nodes of the lattice.
        '''
        idx = {}

        if len(self.latt_size) == 1:
            for i in range(self.latt_size[0]): 
                idx.update({str((i)): torch.tensor((i))})
            
            return idx
        
        else:
            for i in range(self.latt_size[0]):
                for j in range(self.latt_size[1]):
                    idx.update({str((i, j)): torch.tensor((i, j))})
                    if self.latt_shape == 'parallelogram2d':
                        idx.update({str((i + 0.5, j + 0.5)): torch.tensor((i + 0.5, j + 0.5))})

            return idx
        
    def latt_dists(self):
        '''Calculates the distance matrix of the neuron lattice. 
        '''
        dists = []

        for i in self.latt_dict.values():
            row = []
            for j in self.latt_dict.values():
                d = torch.sqrt(torch.sum((i - j)**2)).item()
                row.append(d)
            dists.append(row)
            
        if self.latt_shape == 'circle1d':
            dists[0][-1] = 1.
            dists[-1][0] = 1.
        
        return torch.tensor(dists, dtype=torch.float32)
    
    def nhb_func(self, idx):
        '''Applies the neighborhood function.
        '''
        return torch.exp(-self.nhb_dists[idx] / (2 * self.sigma**2))

    def forward(self, x):
        dists = torch.sqrt(torch.sum((x - self.W)**2, dim=1)) 
        argmin = torch.argmin(dists)  

        latt_dists = self.nhb_func(argmin) 
        latt_dists = latt_dists.unsqueeze(1) 

        self.W += self.lr * latt_dists * (x - self.W)
