#
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
)

class Decoder:
    def __init__(self, encoder_stack):
        self.encoder_stack = encoder_stack

        # Define input tokeen sequence 
        self.bos_token = '<|bos|>'
        self.maxlen = 20
        self.d_model = 16
        self.heads = 4
        self.drop_rate = 0.2
        self.batch_size = 1
        self.pad_token = '<pad>'

        # Tokenize input sequence
        self.tokenizer = self.encoder_stack.tokenizer
        self.tokenizer.texts_to_sequences(self.bos_token)
        input_sequence = np.array([self.tokenizer.word2idx[word] for word in self.bos_token.lower().split()])    
    
        # Apply padding to input sequence
        padded_input_sequence = pad_sequences([input_sequence], padding='post', maxlen=self.maxlen, value=self.tokenizer.word2idx[self.pad_token])

        # Generate positional encoding for padded sequence
        self.input_sequence_length = padded_input_sequence.shape[1]
        self.positional_embedding_layer = PositionalEmbedding(d_model=self.d_model, input_sequence_length=self.input_sequence_length)

        ## Get positional encoding
        self.positional_encoding = self.positional_embedding_layer().reshape(self.batch_size, self.input_sequence_length, self.d_model)

        ## Create mask for padding
        self.mask = self.create_padding_mask(padded_input_sequence)

    def __call__(self):
        output = self.forward()
        return output

    def forward(self):
        # Multi-Head Attention layer 
        multi_head_attn = MultiHeadAttention(positional_encoding=self.positional_encoding.reshape(self.batch_size, self.input_sequence_length, self.d_model),
                                             input_sequence_length=self.input_sequence_length,
                                             d_model=self.d_model,
                                             batch_size=self.batch_size,
                                             heads=self.heads,
                                             mask=self.mask)
        multi_head_output = multi_head_attn()

        # Dropout layer
        dropout_layer = Dropout(dropout_rate=self.drop_rate) 
        multi_head_output = dropout_layer(multi_head_output)

        # Add & Norm: Add residual connection to multi-head attention output and normalize it with layer normalization
        layer_normalization = LayerNormalization(normalized_shape=self.d_model)
        layer_normalization_output = layer_normalization(normalize=multi_head_output,
                                                         residual=self.positional_encoding.reshape(self.batch_size, self.input_sequence_length, self.d_model)
                                                         )
        
        # Multi-Head Attention layer
        #multi_head_attn = MultiHeadAttention() # CONTINUE FROM HERE!
        
        import pdb ; pdb.set_trace()

        # Feed Forward layer
        feed_forward = FeedForward(input_dim=self.d_model, output_dim=self.d_model, activation='relu')
        feed_forward_output = feed_forward(x=layer_normalization_output)

        ## Dropout layer
        dropout_layer = Dropout(dropout_rate=self.drop_rate) 
        feed_forward_output = dropout_layer(feed_forward_output)

        # Add & Norm: Add residual connection to feed forward output and normalize it with layer normalization
        ## Normalize output
        layer_normalization = LayerNormalization(normalized_shape=self.d_model)
        feed_forward_output_norm = layer_normalization(normalize=feed_forward_output, residual=layer_normalization_output)
        
        K_output, V_output = (feed_forward_output_norm, feed_forward_output_norm)
        
        return K_output, V_output

    def create_padding_mask(self, input_sequence):
        mask = np.zeros((self.batch_size, 1, self.input_sequence_length), dtype=bool)
        for i in range(self.input_sequence_length):
            if input_sequence[0, i] == self.tokenizer.word2idx[self.pad_token]:
                mask[0][0][i] = True
        return mask
    
if __name__ == '__main__':
    decoder = Decoder()
    decoder()
    