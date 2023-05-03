import argparse
from debugger import (
    handle_error,
    log_object,
    log_error,
)
from encoder import Encoder

def main():
    parser = argparse.ArgumentParser(description='Encoder for Transformer model')
    parser.add_argument('-F', '--input_file', type=str, default='transformer/input_text.txt', help='Path to the input text file')
    parser.add_argument('-H', '--heads', type=int, default=2, help='Number of attention heads')
    parser.add_argument('-D', '--d_model', type=int, default=4, help='Size of the model')
    parser.add_argument('--plot_posemb', default=False, action='store_true', help='Plot the positional embedding')

    args = parser.parse_args()

    input_file = args.input_file
    heads = args.heads
    d_model = args.d_model
    plot_posemb = args.plot_posemb

    with open(input_file, 'r') as f:
        input_text = f.read()

    try:
        encoder = Encoder(input_text, d_model, heads, plot_posemb)
        encoder_result = encoder()
        log_object(obj=encoder, output=encoder_result, exclude_attrs=['positional_encoding',
                                                                      'positional_embedding_layer',
                                                                      'tokenizer',
                                                                      'input_text'])
    except Exception as err:
        handle_error(err)
        log_error(err)

if __name__ == '__main__':
    main()