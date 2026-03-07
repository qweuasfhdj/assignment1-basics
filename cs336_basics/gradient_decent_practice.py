import numpy as np
import matplotlib.pyplot as plt
import torch

def linear_regression(x : np.ndarray, y: np.ndarray, need_intercept: bool) -> np.ndarray:
    one_column = np.ones(x.shape[0])
    x_b = np.hstack((one_column, x))
    theta = np.linalg.inv(x_b.T @ x_b) @ x_b.T @ y
    return theta

class LinearRegressionGradientDecentPractice:
    def __init__(self, lr: float = 1e-3, max_iter: int = 1000, tolerance: float = 1e-6, batch_size: int = 32):
        self.lr = lr
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.batch_size = batch_size
        self.loss_history = []

    def fit(self, x: np.ndarray, y : np.ndarray):
        n_samples, n_features = x.shape
        self.theta = np.zeros(n_features)
        self.intercept = 0.0
        for iter in range(self.max_iter):
            y_pred = x @ self.theta + self.intercept
            error = y - y_pred
            loss = np.mean((error) ** 2)
            self.loss_history.append(loss)

            grad_coeff = -2.0 / n_samples * x.T @ error
            grad_intercept = -2.0 * np.mean(error)

            self.theta -= self.lr * grad_coeff
            self.intercept -= self.lr * grad_intercept

            if iter > 0 and loss < self.tolerance:
                return True, self.theta

        return False, self.theta

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.theta + self.intercept


class LinearRegressionGradientDecentPracticeMiniBatch:
    def __init__(self, lr: float = 1e-3, max_iter: int = 1000, tolerance: float = 1e-6, batch_size: int = 32):
        self.lr = lr
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.batch_size = batch_size
        self.loss_history = []

    def fit(self, x: np.ndarray, y: np.ndarray):
        n_samples, n_features = x.shape
        self.theta = np.zeros(n_features)
        self.intercept = 0.0
        for iter in range(self.max_iter):
            indexes = np.random.permutation(n_samples)
            x_shuffled = x[indexes]
            y_shuffled = y[indexes]
            epoch_loss = 0.0
            n_batches = n_samples // self.batch_size

            for batch in range(n_batches):
                start = batch * self.batch_size
                end = min(start + self.batch_size, n_samples)
                batch_n = end - start

                x_batch = x_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                y_pred = x_batch @ self.theta + self.intercept
                error = y_batch - y_pred
                loss = np.mean((error) ** 2)
                self.loss_history.append(loss)

                grad_coeff = -2.0 / batch_n * x_batch.T @ error
                grad_intercept = -2.0 * np.mean(error)

                self.theta -= self.lr * grad_coeff
                self.intercept -= self.lr * grad_intercept
                epoch_loss += loss

            epoch_loss /= n_batches
            self.loss_history.append(epoch_loss)
            if iter > 0 and np.abs(self.loss_history[-2] - epoch_loss) < self.tolerance:
                return True, self.theta

        return False, self.theta

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.theta + self.intercept

def log_softmax(input: torch.Tensor) -> torch.Tensor:
    # e^x_i / (sum_i(e^x_i))
    max_val = torch.max(input, dim=-1, keepdim=True).values
    input_shifted = input - max_val
    return input_shifted / torch.sum(input_shifted, dim=-1, keepdim=True)


def cross_entropy_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # - sum(y_i log(softmax(yi))
    log_prob = log_softmax(input)
    selected = torch.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return -selected.mean()



if __name__ == '__main__':
    # 生成数据
    np.random.seed(42)
    X = 2 * np.random.rand(1000, 1)
    y = 4 + 3 * X.squeeze() + np.random.randn(1000)

    gd = LinearRegressionGradientDecentPracticeMiniBatch(lr=0.01, max_iter=1000)
    gd.fit(X, y)
    print(f"\n结果: y = {gd.theta[0]:.2f}x + {gd.intercept:.2f}")



