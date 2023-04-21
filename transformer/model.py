import numpy as np
from numpy.random import rand

class TransformerModel:
    def __init__(self, d_model: int, *args, **kwargs):
        self.d_model = d_model
    
    def execute(self, input_text: str):
        initial_emb = self.__get_rand_embedding(input_text)
        pos_enc = self.__get_positional_encoding(initial_embedding=initial_emb)
        positional_embedding = np.add(initial_emb, pos_enc)
        return positional_embedding
        
    
    def __get_positional_encoding(self, initial_embedding):
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
        initial_embedding = rand(self.len_input_text, self.d_model)
        return initial_embedding