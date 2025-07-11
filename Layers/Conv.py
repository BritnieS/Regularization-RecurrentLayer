import numpy as np
from scipy.signal import correlate, correlate2d

# [0] → batch, [1] → channels, [2] → width,... padded.shape[3] → width

def compute_padding(input_dim, kernel_dim, stride):
    out_dim = int(np.ceil(float(input_dim) / float(stride))) # np.ceil rounds up to ensure output size is an integer
    pad_needed = max((out_dim - 1) * stride + kernel_dim - input_dim, 0)
    pad_before = pad_needed // 2  # Splits padding equally before and after the input along the dimension
    pad_after = pad_needed - pad_before
    return pad_before, pad_after

class Conv:
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        # How much the filter moves (step size), Shape of the filter, How many filters you apply (output channels)
        self.trainable = True
        # Normalize stride and convolution shapes by converting to tuples
        if isinstance(stride_shape, int):
            self.stride_shape = (stride_shape,)
        else:
            self.stride_shape = tuple(stride_shape)
        # Save the filter shape and number of filters as class variables
        self.convolution_shape = tuple(convolution_shape)
        self.num_kernels = num_kernels

        # Initialize weights and bias uniformly in [0,1)
        # Output channels (number of filters), Input channels, Filter width, Filter height - 1
        if len(self.convolution_shape) == 2:  # 1D
            weight_shape = (num_kernels, self.convolution_shape[0], self.convolution_shape[1])
        else:  # 2D
            weight_shape = (num_kernels, self.convolution_shape[0], self.convolution_shape[1], self.convolution_shape[2])
        self.weights = np.random.uniform(0, 1, weight_shape)
        #Initializes one bias per output channel (per filter), random values between 0 and 1
        self.bias = np.random.uniform(0, 1, (num_kernels,))
        # store the gradients (derivatives) of weights and bias after the backward pass, initialized 0 for now
        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = np.zeros_like(self.bias)
        # Placeholders for optimizer objects that will update weights and bias during training
        self._optimizer = None
        self._optimizer_bias = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt): # Saves a copy of the optimizer for both weights and bias to avoid shared states
        import copy
        self._optimizer = copy.deepcopy(opt)
        self._optimizer_bias = copy.deepcopy(opt)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    # re initalizing - initializers
    def initialize(self, weights_initializer, bias_initializer):
        weight_shape = self.weights.shape
        # number of input/ output connections per neuron, Product of all dimensions except input channels -o/p
        fan_in = np.prod(self.convolution_shape)
        fan_out = self.num_kernels * np.prod(self.convolution_shape[1:]) # ---
        self.weights = weights_initializer.initialize(weight_shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize((self.num_kernels,), fan_in, fan_out)

    def _pad_input(self, input_tensor):
        # "Same" padding: output spatial dims == input spatial dims for stride=1
        # an internal helper to apply "same" padding to the input tensor
        if len(self.convolution_shape) == 2:  # 1D
            batch, channels, width = input_tensor.shape # Number of input samples, Number of input channels, Length of each signal (1D)
            _, kernel_w = self.convolution_shape # Width of the filter, ignores input channels
            stride = self.stride_shape[0] # using the first value - single/tuple
            pad_w = compute_padding(width, kernel_w, stride)
            # No padding on batch dimension or on channels, Padding only on width (before and after, based on pad_w), mode='constant' adds zeros.
            return np.pad(input_tensor, ((0, 0), (0, 0), pad_w), mode='constant')
        else:  # 2D
            batch, channels, h, w = input_tensor.shape
            _, kernel_h, kernel_w = self.convolution_shape
            stride_y, stride_x = self.stride_shape
            # Calculates padding needed for height and width using your helper function
            pad_h = compute_padding(h, kernel_h, stride_y)
            pad_w = compute_padding(w, kernel_w, stride_x)
            return np.pad(input_tensor, ((0, 0), (0, 0), pad_h, pad_w), mode='constant')

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        padded = self._pad_input(input_tensor) # helper fn
        if len(self.convolution_shape) == 2:  # 1D
            batch, in_channels, in_w = input_tensor.shape
            num_kernels, kernel_c, kernel_w = self.weights.shape # kernel_c: Channels per filter
            stride = self.stride_shape[0]
            out_w = int(np.ceil(float(in_w) / float(stride)))
            output = np.zeros((batch, self.num_kernels, out_w)) # Prepares empty output tensor with correct shape
            for b in range(batch):
                for k in range(self.num_kernels):
                    out = np.zeros(padded.shape[2] - kernel_w + 1) # Prepares temporary array to store result of correlation for this filter -
                    for c in range(kernel_c):
                        # Applies correlation between padded input and filter weights. Adds the result across channels
                        out += correlate(padded[b, c], self.weights[k, c], mode='valid') # ----------
                    # Apply stride, Trims the output to expected size, Add bias for that filter
                    output[b, k, :] = out[::stride][:out_w] + self.bias[k]
            return output
        else:  # 2D
            batch, in_channels, in_h, in_w = input_tensor.shape
            num_kernels, kernel_c, kernel_h, kernel_w = self.weights.shape
            stride_y, stride_x = self.stride_shape
            out_h = int(np.ceil(float(in_h) / float(stride_y)))
            out_w = int(np.ceil(float(in_w) / float(stride_x)))
            output = np.zeros((batch, self.num_kernels, out_h, out_w))
            for b in range(batch):
                for k in range(self.num_kernels):
                    out = np.zeros((padded.shape[2] - kernel_h + 1, padded.shape[3] - kernel_w + 1))
                    for c in range(kernel_c):
                        out += correlate2d(padded[b, c], self.weights[k, c], mode='valid')
                    output[b, k, :, :] = out[::stride_y, ::stride_x][:out_h, :out_w] + self.bias[k]
            return output

    def backward(self, error_tensor):
        if len(self.convolution_shape) == 2:  # 1D
            return self._backward1d(error_tensor)
        else:
            return self._backward2d(error_tensor)

    def _backward1d(self, error_tensor):
        input_tensor = self.input_tensor
        batch, in_channels, in_w = input_tensor.shape
        num_kernels, kernel_c, kernel_w = self.weights.shape
        stride = self.stride_shape[0]
        padded = self._pad_input(input_tensor)  #
        # Prepares empty arrays to accumulate gradients
        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = np.zeros_like(self.bias)
        grad_input = np.zeros_like(input_tensor)
        # Bias gradient - sum of all error contributions for each output channel across the batch and spatial positions
        self._gradient_bias = np.sum(error_tensor, axis=(0, 2))
        # Weights gradient
        for b in range(batch):
            for k in range(num_kernels):
                for c in range(kernel_c):
                    for i in range(error_tensor.shape[2]):
                        start = i * stride
                        region = padded[b, c, start:start + kernel_w]
                        self._gradient_weights[k, c] += error_tensor[b, k, i] * region
        # Input gradient
        for b in range(batch):
            for c in range(in_channels):
                for k in range(num_kernels):
                    # Upsample error tensor by stride
                    upsampled = np.zeros((error_tensor.shape[2] - 1) * stride + 1)
                    upsampled[::stride] = error_tensor[b, k]
                    grad = correlate(upsampled, self.weights[k, c][::-1], mode='full')
                    # Crop grad to in_w
                    start = (grad.shape[0] - in_w) // 2
                    grad_input[b, c, :] += grad[start:start + in_w]
        # Optimizer update
        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self._gradient_weights)
            self.bias = self._optimizer_bias.calculate_update(self.bias, self._gradient_bias)
        return grad_input

    def _backward2d(self, error_tensor):
        input_tensor = self.input_tensor
        batch, in_channels, in_h, in_w = input_tensor.shape
        num_kernels, kernel_c, kernel_h, kernel_w = self.weights.shape
        stride_y, stride_x = self.stride_shape
        padded = self._pad_input(input_tensor)
        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = np.zeros_like(self.bias)
        grad_input = np.zeros_like(input_tensor)
        # Bias gradient - Sums the error tensor across batch, height, and width for each output channel
        self._gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))
        # Weights gradient
        for b in range(batch):
            for k in range(num_kernels):
                for c in range(kernel_c):
                    for y in range(error_tensor.shape[2]):   #error_tensor.shape[0]	Batch size, Number of filters, o/p H, W
                        for x in range(error_tensor.shape[3]):
                            # Calculates where the filter starts on the padded input for this output position
                            y_start = y * stride_y
                            x_start = x * stride_x
                            # Selects the region of the padded input that contributed to this output
                            region = padded[b, c, y_start:y_start + kernel_h, x_start:x_start + kernel_w]
                            self._gradient_weights[k, c] += error_tensor[b, k, y, x] * region
        # Input - gradient
        for b in range(batch):
            for c in range(in_channels):
                for k in range(num_kernels):
                    # Upsample error tensor by stride (insert zeros)
                    # Creates a larger 2D array filled with zeros, The size is calculated to reverse the stride effect from the forward pass
                    upsampled = np.zeros((
                        (error_tensor.shape[2] - 1) * stride_y + 1,
                        (error_tensor.shape[3] - 1) * stride_x + 1
                    ))
                    upsampled[::stride_y, ::stride_x] = error_tensor[b, k] #
                    # [::-1, ::-1] flips the kernel both vertically and horizontally
                    # mode='full' produces the full gradient map, which might be larger than the original input
                    grad = correlate2d(upsampled, self.weights[k, c][::-1, ::-1], mode='full')
                    # Crop grad to in_h, in_w - Extracts the central region of grad to match the original input size
                    start_y = (grad.shape[0] - in_h) // 2
                    start_x = (grad.shape[1] - in_w) // 2
                    grad_input[b, c, :, :] += grad[start_y:start_y + in_h, start_x:start_x + in_w]
        # Optimizer update
        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self._gradient_weights)
            self.bias = self._optimizer_bias.calculate_update(self.bias, self._gradient_bias)
        return grad_input