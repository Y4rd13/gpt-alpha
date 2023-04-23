import numpy as np

class Transformer:
    pass

class Encoder:
    def __init__(self, d_model: int, *args, **kwargs):
        self.d_model = d_model
    
    def execute(self, input_text: str):
        initial_embedding = self.__get_rand_embedding(input_text)
        positional_encoding = self.__get_positional_encoding(initial_embedding=initial_embedding)
        positional_embedding = np.add(initial_embedding, positional_encoding)
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
        initial_embedding = np.random.rand(self.len_input_text, self.d_model)
        return initial_embedding

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

class MultiHeadAttention(LinearLayer):
    def  __init__(self, positional_embedding, len_input_text, d_model, output_dim, mask=None, *args, **kwargs) -> None:
        self.positional_embedding = positional_embedding
        self.input_dim = len_input_text
        self.output_dim = output_dim
        self.num_heads = d_model

        super().__init__(input_dim=self.input_dim, output_dim=self.output_dim) # no estoy seguro del output_dim

        # Initialize matrix for queries, keys, values, and output
        self.Wq = LinearLayer(input_dim=self.input_dim, output_dim=self.output_dim)
        self.Wk = LinearLayer(input_dim=self.input_dim, output_dim=self.output_dim)
        self.Wv = LinearLayer(input_dim=self.input_dim, output_dim=self.output_dim)
        self.Wo = LinearLayer(input_dim=self.input_dim, output_dim=self.output_dim)
    
    def forward(self):
        #heads = np.array(np.split(x, self.num_heads, axis=1))
        # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        filtered_value = self.scaled_dot_product_attention()

        # Concatenate
        concat_value = np.concatenate(filtered_value, axis=1)

        # Apply linear layer
        output = self.Wo.forward(concat_value)

        return output


    def scaled_dot_product_attention(self):
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
        output = np.matmul(attn_filter, value) # value tiene que estar transpuesto?
        
        return output

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True) # keepdims to keep same shape and axis=-1 to sum over last axis