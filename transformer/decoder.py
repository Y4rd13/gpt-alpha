import numpy as np

from layers import (
    MultiHeadAttention,
    LayerNormalization,
    FeedForward,
    Linear,
    Dropout,
)

from activations import Activation

class Decoder:
    def __init__(self, 
                 d_model: int, 
                 input_sequence_length: int, 
                 num_layers: int,
                 heads: int,
                 feedforward_hidden_dim: int,
                 dropout_rate: float):
        self.d_model = d_model
        self.input_sequence_length = input_sequence_length
        self.num_layers = num_layers
        self.heads = heads
        self.feedforward_hidden_dim = feedforward_hidden_dim
        self.dropout_rate = dropout_rate

        self.decoder_layers = [self.build_decoder_layer() for _ in range(num_layers)]

    def build_decoder_layer(self):
        # Multi-head self-attention layer
        multihead_self_attention = MultiHeadAttention(
            positional_encoding=None,
            input_sequence_length=self.input_sequence_length,
            d_model=self.d_model,
            heads=self.heads,
            batch_size=None,  # Not required as the positional encoding is not set
            mask=None
        )

        # Layer normalization
        layer_norm1 = LayerNormalization(normalized_shape=self.d_model)

        # Multi-head encoder-decoder attention layer
        multihead_enc_dec_attention = MultiHeadAttention(
            positional_encoding=None,
            input_sequence_length=self.input_sequence_length,
            d_model=self.d_model,
            heads=self.heads,
            batch_size=None,  # Not required as the positional encoding is not set
            mask=None
        )

        # Layer normalization
        layer_norm2 = LayerNormalization(normalized_shape=self.d_model)

        # Feedforward layer
        feedforward = FeedForward(
            input_dim=self.d_model,
            output_dim=self.feedforward_hidden_dim,
            activation='relu'
        )

        # Layer normalization
        layer_norm3 = LayerNormalization(normalized_shape=self.d_model)

        # Dropout layer
        dropout = Dropout(dropout_rate=self.dropout_rate)

        return {
            'multihead_self_attention': multihead_self_attention,
            'layer_norm1': layer_norm1,
            'multihead_enc_dec_attention': multihead_enc_dec_attention,
            'layer_norm2': layer_norm2,
            'feedforward': feedforward,
            'layer_norm3': layer_norm3,
            'dropout': dropout
        }

    def __call__(self, decoder_input, encoder_output):
        x = decoder_input

        for layer in self.decoder_layers:
            # Multi-head self-attention
            multihead_self_attention_output = layer['multihead_self_attention'](x)

            # Layer normalization
            layer_norm1_output = layer['layer_norm1'](multihead_self_attention_output, x)

            # Multi-head encoder-decoder attention
            multihead_enc_dec_attention_output = layer['multihead_enc_dec_attention'](layer_norm1_output)

            # Layer normalization
            layer_norm2_output = layer['layer_norm2'](multihead_enc_dec_attention_output, layer_norm1_output)

            # Feedforward
            feedforward_output = layer['feedforward'](layer_norm2_output)

            # Layer normalization
            x = layer['layer_norm3'](feedforward_output, layer_norm2_output)

            # Dropout
            x = layer['dropout'](x)

        return x

if __name__ == "__main__":
    # Define some example parameters
    d_model = 512
    input_sequence_length = 128
    num_layers = 6
    heads = 8
    feedforward_hidden_dim = 2048
    dropout_rate = 0.1

    # Create a random input for the decoder
    decoder_input = np.random.rand(input_sequence_length, d_model)

    # Create a random input for the encoder output
    encoder_output = np.random.rand(input_sequence_length, d_model)

    # Instantiate the Decoder
    decoder = Decoder(d_model, input_sequence_length, num_layers, heads, feedforward_hidden_dim, dropout_rate)

    # Call the Decoder
    decoder_output = decoder(decoder_input, encoder_output)

    # Print the output shape
    print("Decoder output shape:", decoder_output.shape)
