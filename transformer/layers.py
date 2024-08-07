'''
https://nn.labml.ai/normalization/layer_norm/index.html
'''
import numpy as np
from activations import Activation
from typing import List, Optional

class Layer:
    def __init__(self, name: str = None, dtype=None, trainable=True, *args, **kwargs):
        self.name = name
        self.dtype = dtype
        self.trainable = trainable

        self._added_weight = []

    def __call__(self, inputs, *args, **kwargs):
        raise NotImplementedError

    def add_weight(self, 
                   name: str, 
                   shape: tuple, 
                   dtype: Optional[str] = None, 
                   initializer: Optional[str] = None, 
                   trainable: bool = True,
                   getter: Optional[str] = None, 
                   *args, **kwargs) -> np.ndarray:
        # A lightweight alternative to tf.Variable.
        # It simply adds a weight variable to the layer.
        if trainable:
            self._added_weight.append(name)
        weight = np.random.randn(*shape)
        setattr(self, name, weight)
        return weight

    @property
    def weights(self) -> List[np.ndarray]:
        # Returns the weights of the layer.
        return self.trainable_weights + self.non_trainable_weights

    @property
    def trainable_weights(self) -> List[np.ndarray]:
        # Returns the trainable weights of the layer.
        return [getattr(self, weight) for weight in self._added_weight]

    @property
    def non_trainable_weights(self) -> List[np.ndarray]:
        # Returns the non trainable weights of the layer.
        return []

    @staticmethod
    def get_activation(activation):
        activations = {cls.__name__.lower(): cls for cls in Activation.__subclasses__()}
        if activation.lower() in activations:
            return activations[activation.lower()]()
        else:
            raise ValueError(f"Activation function '{activation}' not found")

    def get_all_weights(self) -> List[np.ndarray]:
        # Returns the current weights of the layer.
        return self.weights

    def set_all_weights(self, weights: List[np.ndarray]) -> None:
        # Sets the weights of the layer.
        assert len(weights) == len(self.weights), "Expected {} arrays of weights, got {}".format(len(self.weights), len(weights))
        for i, weight in enumerate(weights):
            setattr(self, self._added_weight[i], weight)

class PositionalEmbedding(Layer):
    def __init__(self, d_model: int, input_sequence_length: int):
        super().__init__()
        self.d_model = d_model
        self.input_sequence_length = input_sequence_length
        self.n = 10000  # Constant for the sinusoidal functions: max number of words in a sentence used in Attention is All You Need paper.

    def __call__(self) -> np.ndarray:
        # Get initial embedding and positional encoding
        initial_embedding = np.random.randn(self.input_sequence_length, self.d_model)  # get random initial embedding
        positional_encoding = self.__get_positional_encoding()  # get positional encoding
        positional_embedding = np.add(initial_embedding, positional_encoding)  # add positional encoding to initial embedding
        return positional_embedding

    def __get_positional_encoding(self):
        embedding_dim = self.d_model
        pos_enc = np.zeros((self.input_sequence_length, self.d_model))

        for pos in range(self.input_sequence_length):
            for i in range(embedding_dim):
                if not i % 2:
                    pos_enc[pos, i] = np.sin(pos / ((self.n ** (2 * i)) / self.d_model))
                else:
                    pos_enc[pos, i] = np.cos(pos / ((self.n ** (2 * i)) / self.d_model))
        return pos_enc

class Linear(Layer):
    def __init__(self, input_dim: int, output_dim: int, input_size: Optional[int] = None):
        super().__init__()

        # initialize weight randomly and using a normal distribution
        weight_size = (input_dim or input_size, output_dim)
        self.weight = self.add_weight(name='weight', shape=weight_size)

    def __call__(self, inputs):
        # Calculate dot product between the input (x) and the weight (w)
        output = np.matmul(inputs, self.weight.T)
        return output
class Dropout(Layer):
    def __init__(self, dropout_rate: float):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.dropout_mask = None

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        # Compute the mask for dropout
        self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=inputs.shape)
        # Scale the input by the dropout mask
        output = inputs * self.dropout_mask
        return output

class Attention(Layer): # Also called ScaledDotProduct
    def  __init__(self, 
                  positional_encoding: np.ndarray,
                  input_sequence_length: int, 
                  heads: int, 
                  output_dim: int, 
                  mask: Optional[np.ndarray] = None, 
                  activation: str = 'softmax'):
        
        self.positional_encoding = positional_encoding
        self.input_dim = input_sequence_length
        self.output_dim = output_dim
        self.heads = heads
        self.mask = mask
        self.d_k = self.output_dim // self.heads

        super().__init__()
        #super().__init__(self.input_dim, self.output_dim)

        # Initialize matrix for queries, keys, values, and output
        self.Wq = Linear(self.input_dim, self.output_dim)
        self.Wk = Linear(self.input_dim, self.output_dim)
        self.Wv = Linear(self.input_dim, self.output_dim)
        self.Wo = Linear(self.output_dim, self.heads * self.d_k)

        # Check if specified activation function is available
        self.activation_fn = Layer.get_activation(activation)

    def __call__(self) -> np.ndarray:
        query = self.Wq(self.positional_encoding)
        key = self.Wk(self.positional_encoding)
        value = self.Wv(self.positional_encoding)

        # Calculate attention scores
        output = np.matmul(query, key.T)
        
        # Normalization of the output scores
        scores = output / np.sqrt(self.d_k) # self.output_dim/self.num_heads? 

        if self.mask is not None:
            scores += -1e9 * self.mask

        attn_filter = self.activation_fn(scores, axis=-1, keepdims=True)

        # Apply attention to values
        output = np.matmul(attn_filter, value[:, :self.d_k])
        
        return output


class MultiHeadAttention(Layer):
    def __init__(self,
                 positional_encoding: np.ndarray,
                 input_sequence_length: int, 
                 d_model: int, 
                 heads: int,
                 batch_size: int, 
                 mask: Optional[np.ndarray] = None) -> None:
        
        super().__init__()
        #super().__init__(positional_encoding, input_sequence_length, d_model, d_model // heads, mask)

        self.positional_encoding = positional_encoding
        self.input_sequence_length = input_sequence_length
        self.d_model = d_model
        self.output_dim = d_model
        self.heads = heads
        self.batch_size = batch_size
        self.mask = mask
        assert self.d_model % self.heads == 0, "Number of heads must be a multiple of the model dimension"

        # Create multi-head attention object with Q, K, V, and output weights
        self.attention = Attention(positional_encoding=self.positional_encoding,
                                   input_sequence_length=self.input_sequence_length, 
                                   heads=self.heads, 
                                   output_dim=self.output_dim,
                                   mask=self.mask,
                                   activation='softmax')
    
    def __call__(self) -> np.ndarray:
        # Apply multi-head attention
        filtered_value = np.array([self.attention() for _ in range(self.heads)])

        # Concatenate
        # axis=0 to concatenate vertically, axis=1 to concatenate horizontally, axis=-1 to concatenate over the last axis
        self.d_k = self.d_model // self.heads
        concat_value = np.concatenate(filtered_value, axis=-1).reshape(self.batch_size, self.input_sequence_length, self.d_model)
        #concat_value = np.concatenate(filtered_value, axis=-1)

        # Apply linear layer
        output = self.attention.Wo(concat_value.reshape(self.batch_size, self.input_sequence_length, self.d_model))
        
        return output
    
class LayerNormalization(Layer): # Also called AddAndNorm or Residual
    def __init__(self, normalized_shape: int, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon # 1e-8 to avoid division by zero
        self.normalized_shape = (normalized_shape,)
        # Initialize gamma and beta weights for scaling and shifting the normalized value
        # Gamma is usefull for scaling and beta for shifting
        self.gamma = self.add_weight(name='gamma', shape=self.normalized_shape, initializer='ones')
        self.beta = self.add_weight(name='beta', shape=self.normalized_shape, initializer='zeros')

    def __call__(self, normalize: np.ndarray, residual: np.ndarray) -> np.ndarray:
        # Adds the positional embedding and the multi head attention output.
        x = normalize + residual

        # Calculate mean and variance of input x
        mean = np.mean(x, axis=1, keepdims=True)
        var = np.var(x, axis=1, keepdims=True)
        
        # Normalize
        x_norm = (x - mean) / np.sqrt(var + self.epsilon)

        # Scale and shift
        output = self.gamma * x_norm + self.beta

        # Add residual connection
        output += residual #encoder.positional_embedding

        return output
    
class FeedForward(Layer):
    def __init__(self, input_dim: int, output_dim: int, activation: str = 'relu'):
        super().__init__()
        # super().__init__(input_dim, output_dim)
        self.linear_layer_1 = Linear(input_dim, output_dim)
        self.linear_layer_2 = Linear(output_dim, output_dim)
        self.activation_fn = Layer.get_activation(activation)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Apply linear layer
        linear_layer_1 = self.linear_layer_1(x)

        # Apply activation function
        activation_layer_1 = self.activation_fn(linear_layer_1)

        # Apply linear layer
        linear_layer_2 = self.linear_layer_2(activation_layer_1)

        return linear_layer_2
