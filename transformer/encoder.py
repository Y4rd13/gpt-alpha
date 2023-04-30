
import sys
import numpy as np
import logging

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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transformer/logs/logs.log', mode='w'),
        logging.StreamHandler()
    ]
)

error_logger = logging.getLogger('error')
error_logger.setLevel(logging.ERROR)
error_handler = logging.FileHandler('transformer/logs/error.log', mode='w')
error_logger.addHandler(error_handler)


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

    def __call__(self):
        try:
            logging.info(f'Encoder started: (batch_size: {self.batch_size}, d_model: {self.d_model}, heads: {self.heads}, input_sequence_length: {self.input_sequence_length}, output_dim: {self.output_dim})')
            output = self.forward()
            logging.info(f'Encoder finished with output shape: {output.shape}')
            return output
        except Exception as err:
            handle_error(err)
            error_logger.error(f'Error: {err} . d_model: {self.d_model}\n')

            if err == 'integer division result too large for a float':
                error_raise_message = f'Maximum floating point number exceeded. Try to reduce the value of d_model.\nCurrent maximum floating point number: {sys.float_info.max}\n'
                error_logger.error(error_raise_message)
                raise Exception(error_raise_message)
            raise Exception(err)

    def forward(self):
        # Convert input sequence to numpy array
        self.positional_embedding = PositionalEmbedding(d_model=self.d_model, input_sequence_length=self.input_sequence_length)
        input_sequence = np.array([self.tokenizer.word2idx[word] for word in self.input_sequence])
        input_sequence = input_sequence.reshape(self.batch_size, -1) # add batch dimension

        # Create mask for padding
        mask = self.create_padding_mask(input_sequence)

        # Get positional encoding
        self.positional_encoding = self.positional_embedding.call()
        
        if self.plot_posemb:
            plot_positional_embedding(self.positional_encoding, self.input_sequence_length, self.d_model)

        self.multi_head_attn = MultiHeadAttention(positional_encoding=self.positional_encoding,
                                                  input_sequence_length=self.input_sequence_length,
                                                  d_model=self.d_model,
                                                  batch_size=self.batch_size,
                                                  heads=self.heads)
        
        self.layer_normalization = LayerNormalization(normalized_shape=self.d_model)
        multi_head_output = self.multi_head_attn()
        layer_normalization_output = self.layer_normalization(positional_encoding=self.positional_encoding, multi_head_output=multi_head_output, residual=self.positional_encoding)
        feed_forward_output = self.calculate_feed_forward_output(layer_normalization_output)
        
        return feed_forward_output

    def create_padding_mask(self, input_sequence):
        mask = np.zeros((self.batch_size, 1, self.input_sequence_length), dtype=bool)
        for i in range(self.input_sequence_length):
            if input_sequence[0, i] == self.tokenizer.word2idx['<pad>']:
                mask[0][0][i] = True
        return mask

    def calculate_feed_forward_output(self, input_tensor):
        feed_forward = FeedForward(input_dim=self.d_model, output_dim=self.d_model, activation='relu')
        return feed_forward(x=input_tensor)