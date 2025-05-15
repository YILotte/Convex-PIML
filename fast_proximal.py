import numpy as np

class prox_method():
    import numpy as np
    def __init__(self, A, b, mu, init_iteration, max_iteration, tol, flag):
        self.flag = flag
        if self.flag == 1:
            self.A = A
            self.b = b
            self.m, self.n = self.A.shape
            self.mu = mu
            self.init_iteration = init_iteration
            self.max_iteration = max_iteration
            self.tol = tol
            self.cov = np.dot(self.A.T, self.A)
            self.ATb = np.dot(self.A.T, self.b)
            self.step_size = 1.0 / np.linalg.norm(self.cov, 2)
            self.result_path = []
        elif self.flag == 2:
            self.A1 = A[0]
            self.A2 = A[1]
            self.b1 = b[0]
            self.b2 = b[1]
            self.m, self.n = self.A1.shape
            self.mu = mu[0]
            self.a = mu[1]
            self.init_iteration = init_iteration
            self.max_iteration = max_iteration
            self.tol = tol
            self.cov1 = np.dot(self.A1.T, self.A1)
            self.cov2 = np.dot(self.A2.T, self.A2)
            self.ATb1 = np.dot(self.A1.T, self.b1)
            self.ATb2 = np.dot(self.A2.T, self.b2)
            self.step_size = 1.0 / (self.a * np.linalg.norm(self.cov1, 2) + np.linalg.norm(self.cov2, 2))
            self.result_path = []

    # define LASSO's object function
    def loss(self, w):
        w = w.reshape(-1)
        if self.flag == 1:
            return 0.5 * np.sum(np.square(np.dot(self.A, w) - self.b)) + self.mu * np.sum(np.abs(w))
        elif self.flag == 2:
            return 0.5 * self.a * np.sum(np.square(np.dot(self.A1, w) - self.b1)) + 0.5 * np.sum(
                np.square(np.dot(self.A2, w) - self.b2)) + self.mu * np.sum(np.abs(w))

    # define the proximal function
    def prox(self, u, t):
        if u >= t:
            return 1.0 * (u - t)
        elif u <= -t:
            return 1.0 * (u + t)
        else:
            return 0.0

    def train(self, method='FISTA'):
        self.prox = np.vectorize(self.prox)
        # initial weights
        self.x = np.random.normal(size=(self.n))
        self.x_ = self.x[:]


        def update(x, x_, k, mu):
            y = x + 1.0 * (k - 2) / (k + 1) * (x - x_)
            x_ = x[:]
            if self.flag == 1:
                grad = (np.dot(self.cov, y) - self.ATb)
            elif self.flag == 2:
                grad = self.a * (np.dot(self.cov1, y) - self.ATb1) + (np.dot(self.cov2, y) - self.ATb2)
            tmp = y - self.step_size * grad
            x = self.prox(tmp, mu * self.step_size)
            return x, x_

        for hot_mu in [1e3, 1e2, 1e1, 1e-1, 1e-2, 1e-3]:
            for k in range(self.init_iteration):
                self.x, self.x_ = update(self.x, self.x_, k, hot_mu)
                self.result_path.append(self.loss(self.x))

        self.iters = 1
        self.err_rate = 1.0
        while (self.err_rate > self.tol and self.iters < self.max_iteration):
            self.result_path.append(self.loss(self.x))
            # print('Iteration=%d, Total loss=%f' % (self.iters,self.loss(self.x)))
            self.x, self.x_ = update(self.x, self.x_, self.iters, self.mu)
            self.err_rate = np.abs(self.loss(self.x) - self.loss(self.x_)) / self.loss(self.x_)
            self.iters += 1
        if self.flag == 1:
            print('Iteration=%d, Measurement loss=%f, Sparsity loss=%f, Total loss=%f' % (
                self.iters, np.mean((self.b - self.A @ self.x) ** 2), np.sum(abs(self.x)),
                0.5 / len(self.b) * np.sum((self.b - self.A @ self.x) ** 2) + self.mu * np.sum(abs(self.x))))
        elif self.flag == 2:
            print('Iteration=%d, Measurement loss=%f, Sparsity loss=%f, Total loss=%f' % (
                    self.iters,
                    self.a * np.mean((self.b1 - self.A1 @ self.x) ** 2) + np.mean((self.b2 - self.A2 @ self.x) ** 2),
                    np.sum(abs(self.x)),
                    0.5 * self.a / len(self.b1) * np.sum((self.b1 - self.A1 @ self.x) ** 2) + 0.5 / len(
                        self.b2) * np.sum((self.b2 - self.A2 @ self.x) ** 2) + self.mu * np.sum(abs(self.x))))
