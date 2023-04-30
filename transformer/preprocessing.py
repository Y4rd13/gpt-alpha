class Tokenizer:
    def __init__(self, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ', char_level=False, oov_token=None, pad_token='<pad>'):
        self.word2idx = {}
        self.index2word = {}
        self.filters = filters
        self.lower = lower
        self.split = split
        self.char_level = char_level
        self.oov_token = oov_token
        self.pad_token = pad_token
    
    def texts_to_sequences(self, input_text: str):
        # Preprocessing
        if self.lower:
            input_text = input_text.lower()
        if self.char_level:
            input_text = ' '.join(list(input_text))
        
        # Tokenization
        words = input_text.split(self.split)
        
        # Filters
        if self.filters:
            words = [word for word in words if word not in self.filters]
        
        # Out-of-vocabulary token
        if self.oov_token is not None:
            words = [self.oov_token if word not in self.word2idx else word for word in words]
        
        # Word to index mapping
        for i, word in enumerate(words):
            self.word2idx[word] = i
            self.index2word[i] = word
        
        # Add padding token
        self.word2idx[self.pad_token] = len(self.word2idx)

