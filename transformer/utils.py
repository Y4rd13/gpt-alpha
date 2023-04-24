from traceback import print_exc
from logging import basicConfig, DEBUG, debug

def handle_error(error):
    basicConfig(format='%(levelname)s: %(message)s', level=DEBUG)
    print()
    debug(f'An exception has ocurred: {str(error)}')
    print()
    print_exc()