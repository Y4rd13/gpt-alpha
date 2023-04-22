import numpy as np
from numpy.random import rand

class TransformerModel:
    def __init__(self, d_model: int, *args, **kwargs):
        self.d_model = d_model
    
    def execute(self, input_text: str):
        initial_emb = self.__get_rand_embedding(input_text)
        pos_enc = self.__get_positional_encoding(initial_embedding=initial_emb)
        positional_embedding = np.add(initial_emb, pos_enc)
        return positional_embedding
    
    def __get_positional_encoding(self):
        embedding_dim = self.d_model
        pos_enc = np.zeros((self.len_input_text, self.d_model))

        for pos in range(self.len_input_text):
            for i in range(embedding_dim):
                if not i % 2:
                    pos_enc[pos, i] = np.sin(pos / ((10000 ** (2 * i)) / self.d_model))
                else:
                    pos_enc[pos, i] = np.cos(pos / ((10000 ** (2 * i)) / self.d_model))
        return pos_enc

    def __get_rand_embedding(self, input_text: str):
        # random initial weights
        self.len_input_text = len(input_text.split())
        initial_embedding = rand(self.len_input_text, self.d_model)
        return initial_embedding
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True) # keepdims to keep same shape and axis=-1 to sum over last axis

class LinearLayer:
    def __init__(self, 
                 input_dim: int, # input dimension of the layer (number of neurons in the previous layer)
                 output_dim: int # output dimension of the layer (number of neurons in the current layer)
                 ):
        # initialize weight and bias randomly and using a normal distribution
        self.weight = rand(input_dim, output_dim)
        #self.bias = rand(1, output_dim)
        self.scale = np.sqrt(output_dim)
        
    def forward(self, x):
        # Calculate dot product between the input (x) and the weight (w)
        output = np.dot(x, self.weight)

        # Normalization of the output scores
        scores = output / self.scale
        
        return scores

class MultiHeadAttention(LinearLayer):
    def  __init__(self, len_input_text, d_model, *args, **kwargs) -> None:
        self.input_dim = len_input_text
        self.num_heads = d_model

        super().__init__(input_dim=self.input_dim, output_dim=self.input_dim) # no estoy seguro del output_dim

        # Initialize matrix for queries, keys, values, and output
        self.Wq = LinearLayer(input_dim=self.input_dim, output_dim=self.input_dim)
        self.Wk = LinearLayer(input_dim=self.input_dim, output_dim=self.input_dim)
        self.Wv = LinearLayer(input_dim=self.input_dim, output_dim=self.input_dim)
        self.Wo = LinearLayer(input_dim=self.input_dim, output_dim=self.input_dim)
    
    def forward(self, x):
        heads = np.array(np.split(x, self.num_heads, axis=1))
        query = self.Wq.forward(heads) # no estoy seguro de que sea heads o x
        key = self.Wk.forward(heads)
        value = self.Wv.forward(heads)
        return (query, key, value)


# class MultiHeadAttention:
#     def __init__(self, input_dim, num_heads, dropout_prob):
#         self.input_dim = input_dim
#         self.num_heads = num_heads
#         self.dropout_prob = dropout_prob #  bias?
        
#         # Initialize parameters
#         self.Wq = np.random.randn(input_dim, input_dim)
#         self.Wk = np.random.randn(input_dim, input_dim)
#         self.Wv = np.random.randn(input_dim, input_dim)
#         self.Wo = np.random.randn(input_dim, input_dim)
        
#     def forward(self, x):
#         # Split input into num_heads separate channels
#         heads = np.array(np.split(x, self.num_heads, axis=1))
        
#         # Calculate queries, keys, and values for each head
#         queries = np.matmul(heads, self.Wq)
#         keys = np.matmul(heads, self.Wk)
#         values = np.matmul(heads, self.Wv)
#         ------------------------------------------------------------
        
#         # Calculate attention scores
#         scores = np.matmul(queries, np.transpose(keys, (0, 2, 1)))
#         scores /= np.sqrt(self.input_dim)
#         attn = np.softmax(scores, axis=-1)
#         attn = np.random.binomial(1, 1 - self.dropout_prob, size=attn.shape) * attn
        
#         # Apply attention to values
#         attn_values = np.matmul(attn, values)
        
#         # Combine attention outputs from different heads and apply final linear transformation
#         concat = np.concatenate(attn_values, axis=1)
#         output = np.matmul(concat, self.Wo)
        
#         return output