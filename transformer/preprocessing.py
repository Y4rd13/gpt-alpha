class Tokenizer:
    def __init__(
        self,
        filters: str = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        lower: bool = True,
        split: str = ' ',
        char_level: bool = False,
        oov_token: str = None,
        pad_token: str = '<pad>',
        ):
        """
        Tokenizer for text data.

        Parameters
        ----------
        filters: str
            Characters to filter out from input text.
        lower: bool
            Whether to lowercase the input text.
        split: str
            String to split input text on.
        char_level: bool
            Whether to tokenize at the character-level or word-level.
        oov_token: str
            Token to replace out-of-vocabulary words with.
        pad_token: str
            Token to pad sequences with.
        """
        self.word2idx = {}
        self.index2word = {}
        self.filters = filters
        self.lower = lower
        self.split = split
        self.char_level = char_level
        self.oov_token = oov_token
        self.pad_token = pad_token
    
    def texts_to_sequences(self, input_text: str) -> None:
        """
        Convert input text into sequences of integers.

        Parameters
        ----------
        input_text: str
            Text to tokenize.

        Returns
        -------
        None
        """
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

