import numpy as np

class Pooling:
    def __init__(self, stride_shape, pooling_shape):
        if isinstance(stride_shape, int):
            self.stride_shape = (stride_shape, stride_shape)
        else:
            self.stride_shape = tuple(stride_shape)
        if isinstance(pooling_shape, int):
            self.pooling_shape = (pooling_shape, pooling_shape)
        else:
            self.pooling_shape = tuple(pooling_shape)
        self.trainable = False

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch, channels, in_y, in_x = input_tensor.shape
        pool_y, pool_x = self.pooling_shape
        stride_y, stride_x = self.stride_shape
        out_y = 1 + (in_y - pool_y) // stride_y
        out_x = 1 + (in_x - pool_x) // stride_x
        output = np.zeros((batch, channels, out_y, out_x))
        # For backward pass, save positions of maxima
        self.max_indices = {}
        for b in range(batch):
            for c in range(channels):
                for y in range(out_y):
                    for x in range(out_x):
                        y_start = y * stride_y
                        x_start = x * stride_x
                        window = input_tensor[b, c, y_start:y_start+pool_y, x_start:x_start+pool_x]
                        max_pos = np.unravel_index(np.argmax(window), window.shape)
                        output[b, c, y, x] = window[max_pos]
                        self.max_indices[(b, c, y, x)] = (y_start + max_pos[0], x_start + max_pos[1])
        return output

    def backward(self, error_tensor):
        input_tensor = self.input_tensor
        batch, channels, in_y, in_x = input_tensor.shape
        grad_input = np.zeros_like(input_tensor)
        out_y, out_x = error_tensor.shape[2], error_tensor.shape[3]
        for b in range(batch):
            for c in range(channels):
                for y in range(out_y):
                    for x in range(out_x):
                        max_y, max_x = self.max_indices[(b, c, y, x)]
                        grad_input[b, c, max_y, max_x] += error_tensor[b, c, y, x]
        return grad_input