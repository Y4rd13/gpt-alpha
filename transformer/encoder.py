
import sys
import numpy as np

from preprocessing import Tokenizer
from layers import (
    PositionalEmbedding,
    MultiHeadAttention,
    LayerNormalization,
    FeedForward,
)

from utils import (
    plot_positional_embedding, 
    handle_error,
    )

class Encoder(MultiHeadAttention):
    def __init__(self, 
                 input_text: str, 
                 d_model: int, 
                 heads: int,
                 plot_posemb: bool = False
                 ):
        self.input_text = input_text
        self.input_sequence = self.input_text.lower().split()
        self.input_sequence_length = len(self.input_sequence)
        self.tokenizer = Tokenizer(lower=True, split=' ', char_level=False, oov_token=None, pad_token='<pad>')
        self.tokenizer.texts_to_sequences(self.input_text)

        self.d_model = d_model
        self.heads = heads
        self.output_dim = d_model * heads
        self.batch_size = 1

        self.plot_posemb = plot_posemb

    def call(self):
        # Convert input sequence to numpy array
        self.positional_embedding = PositionalEmbedding(d_model=self.d_model, input_sequence_length=self.input_sequence_length)
        input_sequence = np.array([self.tokenizer.word2idx[word] for word in self.input_sequence])
        input_sequence = input_sequence.reshape(self.batch_size, -1) # add batch dimension

        # Create mask for padding
        mask = np.zeros((self.batch_size, 1, self.input_sequence_length), dtype=bool)
        for i in range(self.input_sequence_length):
            if input_sequence[0, i] == self.tokenizer.word2idx['<pad>']:
                mask[0][0][i] = True

        # Get positional encoding
        self.positional_encoding = self.positional_embedding.call()
        
        if self.plot_posemb:
            plot_positional_embedding(self.positional_encoding, self.input_sequence_length, self.d_model)

        self.multi_head_attn = MultiHeadAttention(positional_encoding=self.positional_encoding,
                                                  input_sequence_length=self.input_sequence_length,
                                                  d_model=self.d_model,
                                                  batch_size=self.batch_size,
                                                  heads=self.heads).forward()
        
        self.layer_normalization = LayerNormalization(normalized_shape=self.d_model)
        self.layer_normalization_output = self.layer_normalization(positional_encoding=self.positional_encoding, multi_head_output=self.multi_head_attn, residual=self.positional_encoding)
        self.feed_forward = FeedForward(input_dim=self.d_model, output_dim=self.d_model, activation='relu')
        self.feed_forward_output = self.feed_forward(x=self.layer_normalization_output)
        return self.feed_forward_output

def test_encoder(input_text, heads, power, iter):
    # clean logs
    open('transformer/logs/logs.log', 'w').close()
    open('transformer/logs/error.log', 'w').close()

    for i in range(1, iter):
        d_model = power**i

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