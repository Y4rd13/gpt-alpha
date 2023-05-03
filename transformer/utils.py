import matplotlib.pyplot as plt
import pickle
from traceback import print_exc
from logging import basicConfig, DEBUG, debug
import numpy as np
from typing import List, Union

def pad_sequence(
    sequences: List[List[Union[int, float, str]]],
    maxlen: int = None,
    dtype: str = "int32",
    padding: str = "pre",
    truncating: str = "pre",
    value: Union[float, str, int] = 0.0,
    ) -> np.ndarray:
    """
    Transform a list (of length num_samples) of sequences (lists of integers) into a 2D Numpy array of shape (num_samples, num_timesteps).
    num_timesteps is either the maxlen argument if provided or the length of the longest sequence in the list.

    Sequences that are shorter than num_timesteps are padded with value until they are num_timesteps long.
    Sequences longer than num_timesteps are truncated to fit the desired length.

    The position where padding or truncation occurs is determined by the padding and truncating arguments, respectively.
    Pre-padding or removing values from the beginning of the sequence is the default.

    Parameters
    ----------
    sequences: List[List[Union[int, float, str]]]
        List of sequences to be transformed into a numpy array.
    maxlen: int
        Maximum length of the sequences.
    dtype: str
        Data type of the numpy array.
    padding: str
        Where to apply padding (pre or post).
    truncating: str
        Where to apply truncation (pre or post).
    value: Union[float, str, int]
        Value to use for padding.

    Returns
    -------
    x: numpy.ndarray
        2D numpy array of shape (num_samples, num_timesteps).
    """
    # Obtener la longitud de la secuencia más larga en la lista de secuencias.
    if maxlen is None:
        maxlen = max(len(s) for s in sequences)

    # Crear el arreglo numpy de salida.
    x = np.full((len(sequences), maxlen), value, dtype=dtype)

    # Rellenar el arreglo con las secuencias acolchadas.
    for i, seq in enumerate(sequences):
        if truncating == "pre":
            trunc = seq[-maxlen:]
        else:
            trunc = seq[:maxlen]
        if padding == "pre":
            x[i, -len(trunc):] = trunc
        else:
            x[i, :len(trunc)] = trunc

    return x



def handle_error(error: Exception) -> None:
    '''
    Handle an exception by printing it and the traceback to the console.
    
    Parameters
    ----------
    error: Exception
        The exception to handle.
        
    Returns
    -------
    None
    '''
    basicConfig(format='%(levelname)s: %(message)s', level=DEBUG)
    print()
    debug(f'An exception has ocurred: {str(error)}')
    print()
    print_exc()


def plot_positional_embedding(positional_encoding: np.ndarray) -> None:
    '''
    Plot the positional embedding.

    Parameters
    ----------
    positional_encoding: np.ndarray
        The positional encoding.
    
    Returns
    -------
    None
    '''
    plt.pcolormesh(positional_encoding, cmap='RdBu')
    plt.xlabel('Depth: (Output dimension)')
    plt.ylabel('Position: (Input dimension)')
    plt.colorbar()
    plt.show()

    with open(f"transformer/logs/plot_posenc.pickle", "wb") as fp:
        pickle.dump(positional_encoding, fp)