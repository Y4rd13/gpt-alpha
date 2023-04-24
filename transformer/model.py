import numpy as np
from layers import *
class Transformer:
    pass

class Encoder(MultiHeadAttention):
    def __init__(self, d_model: int, heads: int, *args, **kwargs):
        self.d_model = d_model
        self.heads = heads
    
    def call(self, input_text: str):
        len_input_text = len(input_text.split())
        self.positional_embedding = PositionalEmbedding(d_model=self.d_model).call(input_text)
        multi_head_attn = MultiHeadAttention(positional_embedding=self.positional_embedding, len_input_text=len_input_text, d_model=self.d_model, output_dim=self.d_model).forward()
        return multi_head_attn

if __name__ == '__main__':
    encoder = Encoder(d_model=4, heads=2)
    x = encoder.call(input_text='hello world')
    print('Result:')
    print(x)
    print('Success!')