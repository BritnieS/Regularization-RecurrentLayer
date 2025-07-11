import numpy as np

# 🔹 Base Optimizer with Regularizer Support
class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer


# 🔹 Stochastic Gradient Descent (SGD)
class Sgd(Optimizer):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        # Apply regularizer directly to shrink weights first
        if self.regularizer:
            weight_tensor = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        return weight_tensor - self.learning_rate * gradient_tensor


# 🔹 SGD with Momentum
class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = None  # Velocity

    def calculate_update(self, weight_tensor, gradient_tensor):
        # Shrink weights first if regularizer is set
        if self.regularizer:
            weight_tensor = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        if self.v is None:
            self.v = np.zeros_like(weight_tensor)
        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        return weight_tensor + self.v


# 🔹 Adam Optimizer
class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu      # beta1
        self.rho = rho    # beta2
        self.epsilon = 1e-8
        self.t = 0
        self.m = None     # First moment
        self.v = None     # Second moment

    def calculate_update(self, weight_tensor, gradient_tensor):
        # Shrink weights first
        if self.regularizer:
            weight_tensor = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)

        if self.m is None:
            self.m = np.zeros_like(weight_tensor)
        if self.v is None:
            self.v = np.zeros_like(weight_tensor)

        self.t += 1
        self.m = self.mu * self.m + (1 - self.mu) * gradient_tensor
        self.v = self.rho * self.v + (1 - self.rho) * (gradient_tensor ** 2)

        m_hat = self.m / (1 - self.mu ** self.t)
        v_hat = self.v / (1 - self.rho ** self.t)

        return weight_tensor - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
