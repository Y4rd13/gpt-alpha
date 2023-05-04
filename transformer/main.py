import argparse
from debugger import (
    handle_error,
    log_error,
)
from transformer import Transformer

def main():
    parser = argparse.ArgumentParser(description='Encoder for Transformer model')
    parser.add_argument('-F', '--input_file', type=str, default='transformer/input_text.txt', help='Path to the input text file')
    parser.add_argument('--heads', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--d_model', type=int, default=4, help='Size of the model')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--drop_rate', type=float, default=.2, help='Dropout rate')
    parser.add_argument('--maxlen', type=int, default=20, help='Maximum length of the input sequence')
    parser.add_argument('--pad_token', type=str, default='<pad>', help='Padding token')
    parser.add_argument('--plot_pe', default=False, action='store_true', help='Plot the positional embedding')

    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        input_text = f.read()

    try:
        transformer_model = Transformer(input_text=input_text,
                          heads=args.heads,
                          d_model=args.d_model,
                          batch_size=args.batch_size,
                          drop_rate=args.drop_rate,
                          pad_token=args.pad_token,
                          maxlen=args.maxlen,
                          plot_pe=args.plot_pe)
        transformer_output = transformer_model()
    except Exception as err:
        handle_error(err)
        log_error(err)

if __name__ == '__main__':
    main()