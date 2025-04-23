'''
This file is about proximal method included basic and fast solvers.
'''
import numpy as np


# Auther: Zhang David <pkuzc@pku.edu.cn>

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

    def train(self, method='BASIC'):
        '''
        Parameters
        ----------
        method: string, 'BASIC'(default) or 'FISTA' or 'Nesterov'
                Specifies the method to train the model.
        '''
        import time
        start_time = time.time()
        # print(method + ' is Solving...')
        self.prox = np.vectorize(self.prox)
        # initial weights
        self.x = np.random.normal(size=(self.n))
        self.x_ = self.x[:]

        if method == 'FISTA':
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
        elif method == 'Nesterov':
            self.v = self.x[:]

            def update(x, v, k, mu):
                theta = 2.0 / (k + 1)
                y = (1.0 - theta) * x + theta * v
                grad = np.dot(self.cov, y) - self.ATb
                tmp = v - self.step_size / theta * grad
                v = self.prox(tmp, mu * self.step_size / theta)
                x = (1.0 - theta) * x + theta * v
                return x, v

            for hot_mu in [1e3, 1e2, 1e1, 1e-1, 1e-2, 1e-3]:
                for k in range(self.init_iteration):
                    self.x, self.v = update(self.x, self.v, k, hot_mu)
                    self.result_path.append(self.loss(self.x))

            self.iters = 1
            self.err_rate = 1.0
            while (self.err_rate > self.tol and self.iters < self.max_iteration):
                self.x_ = self.x[:]
                self.result_path.append(self.loss(self.x))
                self.x, self.v = update(self.x, self.v, self.iters, mu)
                self.err_rate = np.abs(self.loss(self.x) - self.loss(self.x_)) / self.loss(self.x_)
                self.iters += 1

        else:
            def update(x, mu):
                grad = np.dot(self.cov, x) - self.ATb
                tmp = x - self.step_size * grad
                x = self.prox(tmp, mu * self.step_size)
                return x

            for hot_mu in [self.mu, self.mu, self.mu, self.mu, self.mu, self.mu]:
                for k in range(self.init_iteration):
                    self.x = update(self.x, hot_mu)
                    self.result_path.append(self.loss(self.x))

            self.iters = 1
            self.x_ = self.x[:]
            self.err_rate = 1.0
            while (self.err_rate > self.tol and self.iters < self.max_iteration):
                self.result_path.append(self.loss(self.x))
                self.x_ = self.x[:]
                self.x = update(self.x, self.mu)
                self.err_rate = np.abs(self.loss(self.x) - self.loss(self.x_)) / self.loss(self.x_)
                self.iters += 1

        self.run_time = time.time() - start_time
        # print('End!')

    def plot(self, method='BASIC'):
        from bokeh.plotting import figure, output_file, show
        x = range(len(self.result_path))
        y = self.result_path
        output_file("./fast_proximal" + method + ".html")
        p = figure(title="Proximal Method_" + method, x_axis_label='iteration', y_axis_label='loss')
        p.line(x, y, legend_label="Prox", line_width=2)
        show(p)


if __name__ == '__main__':
    import numpy as np
    from bokeh.plotting import figure, output_file, show

    # for reproducibility
    np.random.seed(1337)

    n = 1024
    m = 512
    mu = 1e-3
    init_iteration = int(1e2)
    max_iteration = int(1e3)
    tol = 1e-9

    # Generating test matrices
    A = np.random.normal(size=(m, n))
    u = np.random.normal(size=(n)) * np.random.binomial(1, 0.1, (n))
    b = np.dot(A, u).reshape(-1)

    result_time = []
    result_mse = []
    output_file("./proximal.html")
    p = figure(title="Proximal Method", x_axis_label='iteration', y_axis_label='loss')

    for method, color in zip(["BASIC", "FISTA", "Nesterov"], ["orange", "red", "blue"]):
        model = prox_method(A, b, mu, init_iteration, max_iteration, tol)
        model.train(method)
        result_time.append(model.run_time)
        result_mse.append(np.mean(np.square(model.x - u)))
        x = range(len(model.result_path))
        y = model.result_path
        p.line(x, y, legend=method, line_width=2, line_color=color)

    show(p)

#
# class prox_method():
#     import numpy as np
#     def __init__(self, A, b, Psi, B,beta, a1,a2, mu, init_iteration, max_iteration, tol):
#         self.A = A
#         self.b = b.reshape([len(b),])
#         self.Psi = Psi
#         self.B = B
#         self.beta = beta.reshape([len(beta),])
#         self.a1 = a1
#         self.a2 = a2
#         self.m, self.n = self.A.shape
#         self.mu = mu
#         self.init_iteration = init_iteration
#         self.max_iteration = max_iteration
#         self.tol = tol
#         self.cov = self.a2*np.dot(self.A.T, self.A)+self.a1*np.dot(self.Psi.T, self.Psi)
#         self.ATb = self.a2*np.dot(self.A.T, self.b)+self.a1*np.dot(self.Psi.T, np.dot(self.B, self.beta))
#         self.step_size = 1.0 / np.linalg.norm(self.cov, 2)
#         self.result_path = []
#
#     # define LASSO's object function
#     def loss(self, w):
#         w = w.reshape(-1)
#         return 0.5*self.a2 * np.sum(np.square(np.dot(self.A, w) - self.b)) + 0.5*self.a1 * np.sum(np.square(np.dot(self.Psi, w) - np.dot(self.B, self.beta)))+self.mu * np.sum(np.abs(w))
#
#     # define the proximal function
#     def prox(self, u, t):
#         if u >= t:
#             return 1.0 * (u - t)
#         elif u <= -t:
#             return 1.0 * (u + t)
#         else:
#             return 0.0
#
#     def train(self, method='BASIC'):
#         '''
#         Parameters
#         ----------
#         method: string, 'BASIC'(default) or 'FISTA' or 'Nesterov'
#                 Specifies the method to train the model.
#         '''
#         import time
#         start_time = time.time()
#         # print(method + ' is Solving...')
#         self.prox = np.vectorize(self.prox)
#         # initial weights
#         self.x = np.random.normal(size=(self.n))
#         self.x_ = self.x[:]
#
#         if method == 'FISTA':
#             def update(x, x_, k, mu):
#                 y = x + 1.0 * (k - 2) / (k + 1) * (x - x_)
#                 x_ = x[:]
#                 grad = np.dot(self.cov, y) - self.ATb
#                 tmp = y - self.step_size * grad
#                 x = self.prox(tmp, mu * self.step_size)
#                 return x, x_
#
#             for hot_mu in [1e3, 1e2, 1e1, 1e-1, 1e-2, 1e-3]:
#                 for k in range(self.init_iteration):
#                     self.x, self.x_ = update(self.x, self.x_, k, hot_mu)
#                     self.result_path.append(self.loss(self.x))
#
#             self.iters = 1
#             self.err_rate = 1.0
#             while (self.err_rate > self.tol and self.iters < self.max_iteration):
#                 self.result_path.append(self.loss(self.x))
#                 self.x, self.x_ = update(self.x, self.x_, self.iters, self.mu)
#                 self.err_rate = np.abs(self.loss(self.x) - self.loss(self.x_)) / self.loss(self.x_)
#                 self.iters += 1
#
#         elif method == 'Nesterov':
#             self.v = self.x[:]
#
#             def update(x, v, k, mu):
#                 theta = 2.0 / (k + 1)
#                 y = (1.0 - theta) * x + theta * v
#                 grad = np.dot(self.cov, y) - self.ATb
#                 tmp = v - self.step_size / theta * grad
#                 v = self.prox(tmp, mu * self.step_size / theta)
#                 x = (1.0 - theta) * x + theta * v
#                 return x, v
#
#             for hot_mu in [1e3, 1e2, 1e1, 1e-1, 1e-2, 1e-3]:
#                 for k in range(self.init_iteration):
#                     self.x, self.v = update(self.x, self.v, k, hot_mu)
#                     self.result_path.append(self.loss(self.x))
#
#             self.iters = 1
#             self.err_rate = 1.0
#             while (self.err_rate > self.tol and self.iters < self.max_iteration):
#                 self.x_ = self.x[:]
#                 self.result_path.append(self.loss(self.x))
#                 self.x, self.v = update(self.x, self.v, self.iters, mu)
#                 self.err_rate = np.abs(self.loss(self.x) - self.loss(self.x_)) / self.loss(self.x_)
#                 self.iters += 1
#
#         else:
#             def update(x, mu):
#                 grad = np.dot(self.cov, x) - self.ATb
#                 tmp = x - self.step_size * grad
#                 x = self.prox(tmp, mu * self.step_size)
#                 return x
#
#             for hot_mu in [1e3, 1e2, 1e1, 1e-1, 1e-2, 1e-3]:
#                 for k in range(self.init_iteration):
#                     self.x = update(self.x, hot_mu)
#                     self.result_path.append(self.loss(self.x))
#
#             self.iters = 1
#             self.x_ = self.x[:]
#             self.err_rate = 1.0
#             while (self.err_rate > self.tol and self.iters < self.max_iteration):
#                 self.result_path.append(self.loss(self.x))
#                 self.x_ = self.x[:]
#                 self.x = update(self.x, self.mu)
#                 self.err_rate = np.abs(self.loss(self.x) - self.loss(self.x_)) / self.loss(self.x_)
#                 self.iters += 1
#
#         self.run_time = time.time() - start_time
#         # print('End!')
#
#     def plot(self, method='BASIC'):
#         from bokeh.plotting import figure, output_file, show
#         x = range(len(self.result_path))
#         y = self.result_path
#         output_file("./fast_proximal" + method + ".html")
#         p = figure(title="Proximal Method_" + method, x_axis_label='iteration', y_axis_label='loss')
#         p.line(x, y, legend_label="Prox", line_width=2)
#         show(p)


# class prox_method():
#     import numpy as np
#     def __init__(self, A, Psi, b, beta, f_grads, B_ic, B_ic_d, a1, mu, a4, a5, ic, j, c, init_iteration, max_iteration,
#                  tol):
#         self.A = A
#         self.B_ic = B_ic
#         self.B_ic_d = B_ic_d
#         self.b = b.reshape([len(b), ])
#         self.Psi = Psi
#         self.beta = beta.reshape([len(beta), ])
#         self.a1 = a1
#         self.a4 = a4
#         self.a5 = a5
#         self.c = c
#         self.j = j
#         self.ic = ic
#         self.m, self.n = self.A.shape[0], 1
#         self.mu = mu * sum(abs(Psi))
#         self.f_grads = f_grads
#         self.init_iteration = init_iteration
#         self.max_iteration = max_iteration
#         self.tol = tol
#         self.cov = self.a1 * np.dot(self.A[:, self.j].T, self.A[:, self.j]) + self.c + self.a4 * np.dot(
#             self.B_ic[:, self.j].T, self.B_ic[:, self.j]) + self.a5 * np.dot(self.B_ic_d[:, self.j].T,
#                                                                              self.B_ic_d[:, self.j])
#         self.ATb = self.c * self.beta[self.j]
#         self.step_size = 1.0 / self.cov
#         self.result_path = []
#
#     # define LASSO's object function
#     def loss(self, w):
#         w = w.reshape(-1)
#         be = self.beta.copy()
#         be[self.j] = w
#         ind = np.setdiff1d(range(0, len(self.beta)), self.j)
#         Ad = self.A[:, ind]
#         betad = self.beta[ind]
#         Bd_ic = self.B_ic[:, ind]
#         Bd_ict = self.B_ic_d[:, ind]
#
#         return self.a1 * (np.sum((np.dot(Ad, betad) - self.b) * self.A[:, self.j])) * w + 0.5 * self.a1 * np.sum(
#             np.square(self.A[:, self.j] * w)) + self.mu * abs(w) + (
#                 w - self.beta[self.j]) * self.f_grads + 0.5 * self.c * (np.square(w - self.beta[self.j])) + self.a4 * (
#             np.sum((np.dot(Bd_ic, betad)) * self.B_ic[:, self.j])) * w + 0.5 * self.a4 * np.sum(
#             np.square(self.B_ic[:, self.j] * w)) + self.a5 * (
#             np.sum((np.dot(Bd_ict, betad) - self.ic) * self.B_ic_d[:, self.j])) * w + 0.5 * self.a5 * np.sum(
#             np.square(self.B_ic_d[:, self.j] * w))
#
#     # define the proximal function
#     def prox(self, u, t):
#         if u >= t:
#             return 1.0 * (u - t)
#         elif u <= -t:
#             return 1.0 * (u + t)
#         else:
#             return 0.0
#
#     def train(self, method='BASIC'):
#         '''
#         Parameters
#         ----------
#         method: string, 'BASIC'(default) or 'FISTA' or 'Nesterov'
#                 Specifies the method to train the model.
#         '''
#         import time
#         start_time = time.time()
#         # print(method + ' is Solving...')
#         self.prox = np.vectorize(self.prox)
#         # initial weights
#         self.x = np.random.normal(size=(self.n))
#         self.x_ = self.x[:]
#
#         if method == 'FISTA':
#             def update(x, x_, k, mu):
#                 y = x + 1.0 * (k - 2) / (k + 1) * (x - x_)
#                 x_ = x[:]
#                 be = self.beta.copy()
#                 be[self.j] = y
#                 ind = np.setdiff1d(range(0, len(self.beta)), self.j)
#                 Ad = self.A[:, ind]
#                 betad = self.beta[ind]
#                 Bd_ic = self.B_ic[:, ind]
#                 Bd_ict = self.B_ic_d[:, ind]
#
#                 grad = self.a1 * np.sum((np.dot(Ad, betad) - self.b) * self.A[:, self.j]) + np.dot(self.cov,
#                                                                                                    y) - self.ATb + self.f_grads + self.a4 * np.sum(
#                     (np.dot(Bd_ic, betad)) * self.B_ic[:, self.j]) + self.a5 * np.sum(
#                     (np.dot(Bd_ict, betad) - self.ic) * self.B_ic_d[:, self.j])
#
#                 tmp = y - self.step_size * grad
#                 x = self.prox(tmp, mu * self.step_size)
#                 return x, x_
#
#             for hot_mu in [1e3, 1e2, 1e1, 1e-1, 1e-2, 1e-3]:
#                 for k in range(self.init_iteration):
#                     self.x, self.x_ = update(self.x, self.x_, k, hot_mu)
#                     self.result_path.append(self.loss(self.x))
#
#                     self.iters = 1
#                     self.err_rate = 1.0
#
#                     while (self.err_rate > self.tol and self.iters < self.max_iteration):
#                         self.result_path.append(self.loss(self.x))
#                         self.x, self.x_ = update(self.x, self.x_, self.iters, self.mu)
#                         self.err_rate = np.abs(self.loss(self.x) - self.loss(self.x_)) / self.loss(self.x_)
#                         self.iters += 1
#
#
#         elif method == 'Nesterov':
#             self.v = self.x[:]
#
#             def update(x, v, k, mu):
#                 theta = 2.0 / (k + 1)
#                 y = (1.0 - theta) * x + theta * v
#                 grad = np.dot(self.cov, y) - self.ATb
#                 tmp = v - self.step_size / theta * grad
#                 v = self.prox(tmp, mu * self.step_size / theta)
#                 x = (1.0 - theta) * x + theta * v
#                 return x, v
#
#             for hot_mu in [1e3, 1e2, 1e1, 1e-1, 1e-2, 1e-3]:
#                 for k in range(self.init_iteration):
#                     self.x, self.v = update(self.x, self.v, k, hot_mu)
#                     self.result_path.append(self.loss(self.x))
#
#             self.iters = 1
#             self.err_rate = 1.0
#             while (self.err_rate > self.tol and self.iters < self.max_iteration):
#                 self.x_ = self.x[:]
#                 self.result_path.append(self.loss(self.x))
#                 self.x, self.v = update(self.x, self.v, self.iters, self.mu)
#                 self.err_rate = np.abs(self.loss(self.x) - self.loss(self.x_)) / self.loss(self.x_)
#                 self.iters += 1
#
#         else:
#             def update(x, mu):
#                 grad = np.dot(self.cov, x) - self.ATb
#                 tmp = x - self.step_size * grad
#                 x = self.prox(tmp, mu * self.step_size)
#                 return x
#
#             for hot_mu in [1e3, 1e2, 1e1, 1e-1, 1e-2, 1e-3]:
#                 for k in range(self.init_iteration):
#                     self.x = update(self.x, hot_mu)
#                     self.result_path.append(self.loss(self.x))
#
#             self.iters = 1
#             self.x_ = self.x[:]
#             self.err_rate = 1.0
#             while (self.err_rate > self.tol and self.iters < self.max_iteration):
#                 self.result_path.append(self.loss(self.x))
#                 self.x_ = self.x[:]
#                 self.x = update(self.x, self.mu)
#                 self.err_rate = np.abs(self.loss(self.x) - self.loss(self.x_)) / self.loss(self.x_)
#                 self.iters += 1
#
#         self.run_time = time.time() - start_time
#         # print('End!')
#
#     def plot(self, method='BASIC'):
#         from bokeh.plotting import figure, output_file, show
#         x = range(len(self.result_path))
#         y = self.result_path
#         output_file("./fast_proximal" + method + ".html")
#         p = figure(title="Proximal Method_" + method, x_axis_label='iteration', y_axis_label='loss')
#         p.line(x, y, legend_label="Prox", line_width=2)
#         show(p)

# if __name__ == '__main__':
#     import numpy as np
#     from bokeh.plotting import figure, output_file, show
#
#     # for reproducibility
#     np.random.seed(1337)
#
#     n = 1024
#     m = 512
#     mu = 1e-3
#     init_iteration = int(1e2)
#     max_iteration = int(1e3)
#     tol = 1e-9
#
#     # Generating test matrices
#     A = np.random.normal(size=(m, n))
#     u = np.random.normal(size=(n)) * np.random.binomial(1, 0.1, (n))
#     b = np.dot(A, u).reshape(-1)
#
#     result_time = []
#     result_mse = []
#     output_file("./proximal.html")
#     p = figure(title="Proximal Method", x_axis_label='iteration', y_axis_label='loss')
#
#     for method, color in zip(["BASIC", "FISTA", "Nesterov"], ["orange", "red", "blue"]):
#         model = prox_method(A, b, mu, init_iteration, max_iteration, tol)
#         model.train(method)
#         result_time.append(model.run_time)
#         result_mse.append(np.mean(np.square(model.x - u)))
#         x = range(len(model.result_path))
#         y = model.result_path
#         p.line(x, y, legend=method, line_width=2, line_color=color)
#
#     show(p)
