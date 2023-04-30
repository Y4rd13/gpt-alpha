import numpy as np

class Activation:
    def __init__(self):
        self.name = self.__class__.__name__

    def __call__(self, x, *args, **kwargs):
        return self.forward(x, *args, **kwargs)
    
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, x):
        raise NotImplementedError
    
class ReLU(Activation):
    def forward(self, x):
        return np.maximum(0, x)
    
    def backward(self, x):
        return np.where(x > 0, 1, 0)
    
class Sigmoid(Activation):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x):
        return self.forward(x) * (1 - self.forward(x))
    
class Tanh(Activation):
    def forward(self, x):
        return np.tanh(x)
    
    def backward(self, x):
        return 1 - np.square(self.forward(x))

class Softmax(Activation):
    def __init__(self):
        super().__init__()
        self.name = "Softmax"

    def forward(self, x, axis=-1, keepdims=True):
        return self.softmax(x, axis=axis, keepdims=keepdims)
    
    def backward(self, x):
        return self.softmax(x) * (1 - self.softmax(x))

    def softmax(self, x, axis=-1, keepdims=True):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=keepdims))
        return e_x / e_x.sum(axis=axis, keepdims=keepdims)
    
class GELU(Activation):
    def forward(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    
    def backward(self, x):
        return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))) + 0.5 * x * (1 - np.square(np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))) * np.sqrt(2 / np.pi) * (1 + 0.134145 * np.power(x, 2))
    
class LeakyReLU(Activation):
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    
    def forward(self, x):
        return np.where(x > 0, x, x * self.alpha)
    
    def backward(self, x):
        return np.where(x > 0, 1, self.alpha)
    
class ELU(Activation):
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def forward(self, x):
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
    
    def backward(self, x):
        return np.where(x > 0, 1, self.alpha * np.exp(x))
    
class SELU(Activation):
    def __init__(self, alpha: float = 1.6732632423543772848170429916717, scale: float = 1.0507009873554804934193349852946):
        self.alpha = alpha
        self.scale = scale
    
    def forward(self, x):
        return self.scale * np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
    
    def backward(self, x):
        return self.scale * np.where(x > 0, 1, self.alpha * np.exp(x))
    
class Softplus(Activation):
    def forward(self, x):
        return np.log(1 + np.exp(x))
    
    def backward(self, x):
        return 1 / (1 + np.exp(-x))
    
class Softsign(Activation):
    def forward(self, x):
        return x / (1 + np.abs(x))
    
    def backward(self, x):
        return 1 / np.square(1 + np.abs(x))
    
class Mish(Activation):
    def forward(self, x):
        return x * np.tanh(np.log(1 + np.exp(x)))
    
    def backward(self, x):
        return np.tanh(np.log(1 + np.exp(x))) + x * (1 - np.square(np.tanh(np.log(1 + np.exp(x))))) * (1 / (1 + np.exp(-x)))
    
class Swish(Activation):
    def forward(self, x):
        return x / (1 + np.exp(-x))
    
    def backward(self, x):
        return (np.exp(-x) * (x + 1) + 1) / np.square(1 + np.exp(-x))
    
class HardSigmoid(Activation):
    def forward(self, x):
        return np.clip((x + 3) / 6, 0, 1)
    
    def backward(self, x):
        return np.where((x > -3) & (x < 3), 1 / 6, 0)
    
class HardSwish(Activation):
    def forward(self, x):
        return x * np.clip((x + 3) / 6, 0, 1)
    
    def backward(self, x):
        return np.where((x > -3) & (x < 3), (x + 3) / 3, 0)

