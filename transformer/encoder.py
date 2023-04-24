
'''
TODO:
    - Define the input_dim and output_dim for the AddAndNorm layer and FeedForward layer.
'''
import sys
from layers import *
from utils import handle_error
class Transformer:
    pass

class Encoder(MultiHeadAttention):
    def __init__(self, d_model: int, heads: int, *args, **kwargs):
        self.d_model = d_model
        self.heads = heads
        self.output_dim = d_model * heads
    
    def call(self, input_text: str):
        len_input_text = len(input_text.split())

        self.positional_embedding = PositionalEmbedding(d_model=self.d_model).call(input_text)

        self.multi_head_attn = MultiHeadAttention(positional_embedding=self.positional_embedding,
                                                  len_input_text=len_input_text,
                                                  d_model=self.d_model,
                                                  output_dim=self.output_dim,
                                                  heads=self.heads).forward()
        
        self.add_norm = AddAndNorm(input_dim=self.d_model)
        self.add_and_norm_output = self.add_norm.forward(pos_embeding=self.positional_embedding, multi_head_output=self.multi_head_attn, residual=self.positional_embedding)
        #self.feed_forward_output = FeedForward(input_dim=self.d_model, output_dim=self.d_model, activation='relu').forward(x=self.add_and_norm_output)

if __name__ == '__main__':
    heads = 2
    input_text = 'hello world'

    for i in range(1, 10):
        d_model = 2**i

        try:
            print(f'Iteration: {i}')
            encoder = Encoder(d_model, heads)
            encoder_result = encoder.call(input_text)
            print(f'Encoder result: {encoder_result}')

            with open('transformer/logs/logs.log', 'a') as f:
                f.write(f'Iteration: {i} . d_model: {d_model}\n')
                f.write(f'Encoder result: {encoder_result}\n')
                f.write('-' * 50 + '\n')

        except Exception as err:
            err = str(err)
            handle_error(err)
            with open('transformer/logs/error.log', 'a') as f:
                f.write(f'Iteration: {i} . d_model: {d_model}\n') 
                f.write(f'Error: {err}\n')

                if err == 'integer division result too large for a float':
                    error_raise_message = f'Maximum floating point number exceeded. Try to reduce the value of d_model.\nCurrent maximum floating point number: {sys.float_info.max}\n'
                    f.write(error_raise_message)
                    raise Exception(error_raise_message)
                f.write('-'*50 + '\n')