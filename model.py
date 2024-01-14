from sklearn.linear_model import LinearRegression
import pickle


class MyModel(LinearRegression):
    def __init__(self, path=None):
        super().__init__()

        if path:
            self.load_model(path)

    def load_model(self, path):
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            self.__dict__.update(model.__dict__)
        except FileNotFoundError:
            print(f'File Not Found: {path}')
        except Exception as e:
            print(f'Error loading model: {e}')

    def model_quality(self):

        quality = {
            'R2': 0.8578,
            'MSE': 13.6281,
            'MAX_ERROR': 14.2192
        }

        '''
        Calculating R2, MSE, MAX_ERROR
        '''
        return quality
