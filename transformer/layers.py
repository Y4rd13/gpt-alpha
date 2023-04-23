import numpy as np

class LinearLayer:
    def __init__(self, 
                 input_dim: int, # input dimension of the layer (number of neurons in the previous layer)
                 output_dim: int # output dimension of the layer (number of neurons in the current layer)
                 ):
        # initialize weight and bias randomly and using a normal distribution
        self.weight = np.random.rand(input_dim, output_dim)
        #self.bias = rand(1, output_dim)
        #self.scale = np.sqrt(output_dim)
        
    def forward(self, x):
        # Calculate dot product between the input (x) and the weight (w)
        output = np.matmul(x, self.weight.T)

        # Normalization of the output scores
        #scores = output / self.scale

        return output