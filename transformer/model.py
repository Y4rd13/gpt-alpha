import numpy as np
from layers import LinearLayer, MultiHeadAttention
class Transformer:
    pass
    
class Encoder(MultiHeadAttention):
    def __init__(self, d_model: int, heads: int, *args, **kwargs):
        self.d_model = d_model
        self.heads = heads
    
    def execute(self, input_text: str):
        # Get initial embedding and positional encoding
        initial_embedding = self.__get_rand_embedding(input_text)
        positional_encoding = self.__get_positional_encoding()
        positional_embedding = np.add(initial_embedding, positional_encoding)
        multi_head_attn = MultiHeadAttention(positional_embedding=positional_embedding, len_input_text=self.len_input_text, d_model=self.d_model, output_dim=self.d_model).forward()
        return multi_head_attn

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
    

if __name__ == '__main__':
    encoder = Encoder(d_model=4, heads=2)
    x = encoder.execute(input_text='hello world')
    print('Result:')
    print(x)
    print('Success!')