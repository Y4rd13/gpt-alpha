import numpy as np
from preprocessing import Tokenizer
from layers import (
    PositionalEmbedding,
    MultiHeadAttention,
    LayerNormalization,
    FeedForward,
    Dropout,
)

from utils import (
    pad_sequences,
    plot_positional_embedding, 
)

class Encoder(MultiHeadAttention):
    def __init__(self, 
                 input_text: str, 
                 d_model: int, 
                 heads: int,
                 plot_posemb: bool = False
                 ):
        self.d_model = d_model
        self.heads = heads
        self.output_dim = d_model * heads
        self.batch_size = 1
        self.drop_rate = .2
        self.maxlen = 20
        self.pad_token = '<pad>'
        self.plot_posemb = plot_posemb


        self.input_text = input_text
        #self.input_sequence = self.input_text.lower().split()

        # Tokenize input sequence
        self.tokenizer = Tokenizer(lower=True, split=' ', char_level=False, oov_token=None, pad_token=self.pad_token)
        self.tokenizer.texts_to_sequences(self.input_text)
        input_sequence = np.array([self.tokenizer.word2idx[word] for word in self.input_text.lower().split()])

        # Apply padding to input sequence
        padded_input_sequence = pad_sequences([input_sequence], padding='post', maxlen=20, value=self.tokenizer.word2idx[self.pad_token])

        # Generate positional encoding for padded sequence
        self.input_sequence_length = padded_input_sequence.shape[1]
        self.positional_embedding_layer = PositionalEmbedding(d_model=self.d_model, input_sequence_length=self.input_sequence_length)

        #input_sequence = input_sequence.reshape(self.batch_size, -1) # add batch dimension

        ## Create mask for padding
        #mask = self.create_padding_mask(input_sequence)

        ## Get positional encoding
        self.positional_encoding = self.positional_embedding_layer()
        print(f'positional_encoding: {self.positional_encoding.shape}, type: {type(self.positional_encoding)}')

        if self.plot_posemb:
            plot_positional_embedding(self.positional_encoding, self.input_sequence_length, self.d_model)

    def __call__(self):
        output = self.forward()
        return output

    def forward(self):
        # Multi-Head Attention layer 
        multi_head_attn = MultiHeadAttention(positional_encoding=self.positional_encoding,
                                                  input_sequence_length=self.input_sequence_length,
                                                  d_model=self.d_model,
                                                  batch_size=self.batch_size,
                                                  heads=self.heads)
        multi_head_output = multi_head_attn()

        # Dropout layer
        dropout_layer = Dropout(dropout_rate=self.drop_rate) 
        multi_head_output = dropout_layer(multi_head_output)
        print(f'multi_head_output: {multi_head_output.shape}')

        # Add & Norm: Add residual connection to multi-head attention output and normalize it with layer normalization
        layer_normalization = LayerNormalization(normalized_shape=self.d_model)
        layer_normalization_output = layer_normalization(normalize=multi_head_output,
                                                         residual=self.positional_encoding.reshape(self.batch_size, self.input_sequence_length, self.d_model)
                                                         )
        print(f'layer_normalization_output: {layer_normalization_output.shape}')

        # Feed Forward layer
        feed_forward = FeedForward(input_dim=self.d_model, output_dim=self.d_model, activation='relu')
        feed_forward_output = feed_forward(x=layer_normalization_output)
        print(f'feed_forward_output: {feed_forward_output.shape}')

        ## Dropout layer
        dropout_layer = Dropout(dropout_rate=self.drop_rate) 
        feed_forward_output = dropout_layer(feed_forward_output)

        # Add & Norm: Add residual connection to feed forward output and normalize it with layer normalization
        ## Normalize output
        layer_normalization = LayerNormalization(normalized_shape=self.d_model)
        feed_forward_output_norm = layer_normalization(normalize=feed_forward_output, residual=layer_normalization_output)
        
        K_output, V_output = (feed_forward_output_norm, feed_forward_output_norm)
        
        return K_output, V_output

    # def create_padding_mask(self, input_sequence):
    #     mask = np.zeros((self.batch_size, 1, self.input_sequence_length), dtype=bool)
    #     for i in range(self.input_sequence_length):
    #         if input_sequence[0, i] == self.tokenizer.word2idx['<pad>']:
    #             mask[0][0][i] = True
    #     return mask