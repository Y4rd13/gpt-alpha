import numpy as np

class LinearLayer:
    def __init__(self, 
                 input_dim: int, # input dimension of the layer (number of neurons in the previous layer)
                 output_dim: int # output dimension of the layer (number of neurons in the current layer)
                 ):
        # initialize weight randomly and using a normal distribution
        weight_size = (input_dim, output_dim)
        weight_total_elements = np.prod(weight_size)
        self.weight = np.random.rand(weight_total_elements).reshape(weight_size)
        
    def forward(self, x):
        # Calculate dot product between the input (x) and the weight (w)
        output = np.matmul(x, self.weight.T)

        return output
class ScaledDotProduct(LinearLayer):
    def  __init__(self, positional_embedding, len_input_text, d_model, output_dim, mask=None, *args, **kwargs) -> None:
        self.positional_embedding = positional_embedding
        self.input_dim = len_input_text
        self.output_dim = output_dim
        self.num_heads = d_model
        self.mask = mask

        super().__init__(self.input_dim, self.output_dim)

        # Initialize matrix for queries, keys, values, and output
        self.Wq = LinearLayer(self.input_dim, self.output_dim)
        self.Wk = LinearLayer(self.input_dim, self.output_dim)
        self.Wv = LinearLayer(self.input_dim, self.output_dim)
        self.Wo = LinearLayer(self.input_dim, self.output_dim)

    def forward(self):
        query = self.Wq.forward(self.positional_embedding)
        key = self.Wk.forward(self.positional_embedding)
        value = self.Wv.forward(self.positional_embedding)

        # Calculate attention scores
        output = np.matmul(query, key.T)
        # output /= output/np.sqrt(shape(output)[0], shape(output)[1])
        
        # Normalization of the output scores
        scores = output / np.sqrt(self.output_dim)

        if self.mask is not None: # que chucha?
            scores += -1e9 * self.mask

        attn_filter = self.softmax(x=scores)

        # Apply attention to values
        output = np.matmul(attn_filter, value.T) # value tiene que estar transpuesto?
        
        return output

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True) # keepdims to keep same shape and axis=-1 to sum over last axis
    
class MultiHeadAttention(ScaledDotProduct):
    def __init__(self, positional_embedding, len_input_text, d_model, output_dim, mask=None, *args, **kwargs) -> None:
        super().__init__(positional_embedding, len_input_text, d_model, output_dim, mask, *args, **kwargs)
        self.len_input_text = len_input_text
        self.d_model = d_model
    
    def forward(self):
        # Create multi-head attention object with Q, K, V, and output weights
        scaled_dot_prod = ScaledDotProduct(positional_embedding=self.positional_embedding, len_input_text=self.len_input_text, d_model=self.d_model, output_dim=self.d_model)

        # Apply multi-head attention
        filtered_value = [scaled_dot_prod.forward(), scaled_dot_prod.forward()] #* self.heads

        # Concatenate
        concat_value = np.concatenate(filtered_value, axis=0) # axis=0 to concatenate vertically and axis=1 to concatenate horizontally

        # Apply linear layer
        output = scaled_dot_prod.Wo.forward(concat_value.T)

        return output