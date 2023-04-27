import argparse
from encoder import Encoder, test_logger

def main():
    parser = argparse.ArgumentParser(description='Encoder for Transformer model')
    parser.add_argument('-F', '--input_file', type=str, default='transformer/input_text.txt', help='Path to the input text file')
    parser.add_argument('-H', '--heads', type=int, default=2, help='Number of attention heads')
    parser.add_argument('-D', '--d_model', type=int, default=4, help='Size of the model')
    parser.add_argument('-test', '--testing', default=False, action='store_true', help='Activate testing mode')
    parser.add_argument('--plot_posemb', default=False, action='store_true', help='Plot the positional embedding')

    args = parser.parse_args()

    input_file = args.input_file
    heads = args.heads
    d_model = args.d_model
    testing = args.testing
    plot_posemb = args.plot_posemb

    with open(input_file, 'r') as f:
        input_text = f.read()

    if testing:
        test_logger()
    else:
        print(f'd_model: {d_model}, heads: {heads}')
        encoder = Encoder(d_model, heads, plot_posemb)
        encoder_result = encoder.call(input_text)
        print(f'Encoder result: {encoder_result}')

if __name__ == '__main__':
    main()