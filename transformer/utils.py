import matplotlib.pyplot as plt
import pickle
from traceback import print_exc
from logging import basicConfig, DEBUG, debug

def handle_error(error):
    basicConfig(format='%(levelname)s: %(message)s', level=DEBUG)
    print()
    debug(f'An exception has ocurred: {str(error)}')
    print()
    print_exc()

def plot_positional_embedding(positional_embedding, length, d_model):
        plt.pcolormesh(positional_embedding, cmap='RdBu')
        plt.xlabel('Depth: (Output dimension)')
        plt.ylabel('Position: (Input dimension)')
        plt.colorbar()
        plt.show()

        with open(f"transformer/logs/posenc-{length}-{d_model}.pickle", "wb") as fp:
            pickle.dump(positional_embedding, fp)

def test_logger(heads, *args, **kwargs):
    for i in range(1, 10):
        d_model = 2**i

        try:
            print(f'Iteration: {i}')
            encoder = Encoder(d_model, heads)
            encoder_result = encoder.call(input_text)
            print(f'Encoder result: {encoder_result}')

            with open('transformer/logs/logs.log', 'a') as f:
                f.write(f'Iteration: {i} . d_model: {d_model}\n')
                f.write(f'Encoder result: {encoder_result}\n')
                f.write('-' * 50 + '\n')

        except Exception as err:
            err = str(err)
            handle_error(err)
            with open('transformer/logs/error.log', 'a') as f:
                f.write(f'Iteration: {i} . d_model: {d_model}\n') 
                f.write(f'Error: {err}\n')

                if err == 'integer division result too large for a float':
                    error_raise_message = f'Maximum floating point number exceeded. Try to reduce the value of d_model.\nCurrent maximum floating point number: {sys.float_info.max}\n'
                    f.write(error_raise_message)
                    raise Exception(error_raise_message)
                f.write('-'*50 + '\n')