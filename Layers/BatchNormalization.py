import numpy as np
from Layers import Base, Helpers
import copy


class BatchNormalization(Base.BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.trainable = True
        self.channels = channels
        self.initialize()
        self._optimizer = None
        self._optimizer_weights = None
        self._optimizer_bias = None
        self.moving_mean = None
        self.moving_var = None
        self.decay = 1

    def initialize(self, weights_initializer=None, bias_initializer=None):
        self.gamma = np.ones(self.channels)
        self.beta = np.zeros(self.channels)

    def forward(self, input_tensor):
        X = input_tensor
        conv = False
        if X.ndim == 4:
            conv = True
            X = self.reformat(X)
        self.X = X
        if self.testing_phase:
            if self.moving_mean is None or self.moving_var is None:
                raise Exception ("train model first")
            self.mean = self.moving_mean
            self.var = self.moving_var
        else:
            self.mean = np.mean(X, axis=0)
            self.var = np.var(X, axis=0)
            if self.moving_mean is None:
                self.moving_mean = copy.deepcopy(self.mean)
                self.moving_var = copy.deepcopy(self.var)
            else:
                self.moving_mean = self.moving_mean * self.decay + self.mean * (1 - self.decay)
                self.moving_var = self.moving_var * self.decay + self.var * (1 - self.decay)
        self.X_hat = (X - self.mean) / np.sqrt(self.var + 1e-11)
        out = self.gamma * self.X_hat + self.beta
        if conv:
            out = self.reformat(out)
        return out

    def backward(self, error_tensor):
        # Check if we're dealing with convolutional data (4D tensor)
        is_conv_layer = error_tensor.ndim == 4
        
        # Reshape conv data to 2D for processing
        if is_conv_layer:
            error_tensor = self.reformat(error_tensor)
        
        # Calculate gradients for gamma and beta parameters
        gamma_gradient = np.sum(error_tensor * self.X_hat, axis=0)
        beta_gradient = np.sum(error_tensor, axis=0)
        
        # Calculate gradient with respect to input
        input_gradient = Helpers.compute_bn_gradients(error_tensor, self.X, self.gamma, self.mean, self.var)
        
        # Update parameters if optimizer is available
        if self._optimizer_weights is not None:
            self.gamma = self._optimizer_weights.calculate_update(self.gamma, gamma_gradient)
        if self._optimizer_bias is not None:
            self.beta = self._optimizer_bias.calculate_update(self.beta, beta_gradient)
        
        # Reshape back to conv format if needed
        if is_conv_layer:
            input_gradient = self.reformat(input_gradient)
        
        # Store gradients for external access
        self.gradient_weights = gamma_gradient
        self.gradient_bias = beta_gradient
        
        return input_gradient

    def reformat(self, tensor):
        # Convert 4D conv tensor to 2D for batch norm processing
        if tensor.ndim == 4:
            # Save original shape for later
            self.reformat_shape = tensor.shape
            batch_size, channels, height, width = tensor.shape
            
            # Flatten spatial dimensions and move channels to last axis
            # (B, C, H, W) -> (B*H*W, C)
            tensor = tensor.transpose(0, 2, 3, 1)  # (B, H, W, C)
            tensor = tensor.reshape(-1, channels)   # (B*H*W, C)
            return tensor
        
        # Convert back from 2D to 4D conv format
        else:
            batch_size, channels, height, width = self.reformat_shape
            
            # Reshape back to spatial format
            # (B*H*W, C) -> (B, C, H, W)
            tensor = tensor.reshape(batch_size, height, width, channels)  # (B, H, W, C)
            tensor = tensor.transpose(0, 3, 1, 2)  # (B, C, H, W)
            return tensor

    @property
    def weights(self):
        return self.gamma

    @weights.setter
    def weights(self, gamma):
        self.gamma = gamma

    @property
    def bias(self):
        return self.beta

    @bias.setter
    def bias(self, beta):
        self.beta = beta

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer_weights = copy.deepcopy(optimizer)
        self._optimizer_bias = copy.deepcopy(optimizer)




