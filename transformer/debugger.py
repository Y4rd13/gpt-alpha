
import sys
import logging
from traceback import print_exc
from logging import basicConfig, DEBUG, debug
from typing import List
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transformer/logs/logs.log', mode='w'),
        logging.StreamHandler()
    ]
)

error_logger = logging.getLogger('error')
error_logger.setLevel(logging.ERROR)
error_handler = logging.FileHandler('transformer/logs/error.log', mode='w')
error_logger.addHandler(error_handler)


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

def log_object(
        obj: object,
        output: np.ndarray = None,
        exclude_attrs: List[str] = list()
        ) -> None:
    class_name = obj.__class__.__name__
    logging.info(f'>> Logs started for {class_name} object:\n')
    logging.info(f'> Attributes')
    # Get the attributes of the object and exclude those specified in the list
    filtered_vars = {k: v for k, v in vars(obj).items() if k not in exclude_attrs}
    for k, v in filtered_vars.items():
        logging.info(f'+ {k}: {v}')
    
    if output is not None:
        logging.info(f'> Output shape:')
        if class_name == 'Encoder':
            encoder_output_shapes = output[0].shape, output[1].shape
            logging.info(f'+ {encoder_output_shapes}')
    logging.info(f'------------------------------\n')

def log_error(err: Exception) -> None:
    error_logger.error(f'Error: {err}\n')

    if err == 'integer division result too large for a float':
        error_raise_message = f'Maximum floating point number exceeded. Try to reduce the value of d_model.\nCurrent maximum floating point number: {sys.float_info.max}\n'
        error_logger.error(error_raise_message)
        raise Exception(error_raise_message)