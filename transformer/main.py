from layers import TransformerModel

def main():
    params = {
    'd_model': 4
    }
    transformer = TransformerModel(params['d_model'])
    transformer.execute(input_text='hello world')

if __name__ == '__main__':
    main()
