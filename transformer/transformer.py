from decoder import Decoder
from encoder import Encoder
from debugger import log_object

class Transformer:
    def __init__(self,
                 input_text: str,
                 heads: int, 
                 d_model: int, 
                 batch_size: int = 1,
                 drop_rate: float = .2,
                 maxlen: int = 20,
                 pad_token: str = '<pad>',
                 plot_pe: bool = False):

        self.input_text = input_text
        self.d_model = d_model
        self.heads = heads
        self.output_dim = d_model * heads
        self.batch_size = batch_size
        self.drop_rate = drop_rate
        self.maxlen = maxlen
        self.pad_token = pad_token
        self.plot_pe = plot_pe

    def __call__(self) -> None:
        self.forward()
    
    def forward(self) -> None:
        encoder_stack = Encoder(input_text=self.input_text,
                          heads=self.heads,
                          d_model=self.d_model,
                          batch_size=self.batch_size,
                          drop_rate=self.drop_rate,
                          pad_token=self.pad_token,
                          maxlen=self.maxlen,
                          plot_pe=self.plot_pe)
        
        encoder_output = encoder_stack()

        log_object(obj=encoder_stack, 
                   output=encoder_output, 
                   exclude_attrs=['positional_encoding',
                                  'positional_embedding_layer',
                                  'tokenizer',
                                  'input_text'])

        decoder_stack = Decoder(encoder_stack)
        decoder_output = decoder_stack()