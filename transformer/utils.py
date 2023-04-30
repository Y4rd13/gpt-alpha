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

def plot_positional_embedding(positional_encoding, length, d_model):
        plt.pcolormesh(positional_encoding, cmap='RdBu')
        plt.xlabel('Depth: (Output dimension)')
        plt.ylabel('Position: (Input dimension)')
        plt.colorbar()
        plt.show()

        with open(f"transformer/logs/plot_posenc.pickle", "wb") as fp:
            pickle.dump(positional_encoding, fp)