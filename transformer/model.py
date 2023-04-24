
'''
TODO:
    - Define the input_dim and output_dim for the AddAndNorm layer and FeedForward layer.
'''
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
        self.multi_head_attn = MultiHeadAttention(positional_embedding=self.positional_embedding, len_input_text=len_input_text, d_model=self.d_model, output_dim=self.d_model).forward()
        self.add_norm = AddAndNorm(input_dim=self.d_model)
        #self.add_and_norm_output = self.add_norm.forward(x=None, pos_embeding=self.positional_embedding, multi_head_output=self.multi_head_attn, residual=self.positional_embedding)
        #self.feed_forward_output = FeedForward(input_dim=self.d_model, output_dim=self.d_model, activation='relu').forward(x=self.add_and_norm_output)

if __name__ == '__main__':
    encoder = Encoder(d_model=4, heads=2)
    x = encoder.call(input_text='hello world')
    print('Result:')
    print(x)
    print('Success!')