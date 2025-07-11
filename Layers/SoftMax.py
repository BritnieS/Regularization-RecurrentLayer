import numpy as np

class SoftMax:
    def __init__(self):
        self.trainable = False

    def forward(self, input_tensor):
        # Numerical stability: subtract max
        exp = np.exp(input_tensor - np.max(input_tensor, axis=1, keepdims=True))
        self.output = exp / np.sum(exp, axis=1, keepdims=True)
        return self.output

    def backward(self, error_tensor):
        # Full Jacobian-vector product for softmax
        batch_size, n_classes = self.output.shape
        grad_input = np.empty_like(error_tensor)
        for i in range(batch_size):
            y = self.output[i].reshape(-1, 1)
            jacobian = np.diagflat(y) - np.dot(y, y.T)
            grad_input[i] = np.dot(jacobian, error_tensor[i])
        return grad_input