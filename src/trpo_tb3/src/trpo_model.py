#!/usr/bin/python3

import torch
from torch.nn import Module, Sequential, Linear, Tanh, Parameter
from torch.distributions import Categorical, MultivariateNormal

import numpy as np


class PolicyNetwork(Module):
    def __init__(self, state_dim, action_dim, discrete) -> None:
        super().__init__()
        self.net = Sequential(
            Linear(state_dim, 50),
            Tanh(),
            Linear(50, 50),
            Tanh(),
            Linear(50, 50),
            Tanh(),
            Linear(50, action_dim),
        )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete

        if not self.discrete:
            self.log_std = Parameter(torch.zeros(action_dim))

    def forward(self, states):
        if self.discrete:
            probs = torch.softmax(self.net(states), dim=-1)
            distb = Categorical(probs)
        else:
            mean = self.net(states)

            std = torch.exp(self.log_std)
            cov_mtx = torch.eye(self.action_dim) * (std**2)

            distb = MultivariateNormal(mean, cov_mtx)

        return distb


class ValueNetwork(Module):
    def __init__(self, state_dim) -> None:
        super().__init__()
        self.net = Sequential(
            Linear(state_dim, 50),
            Tanh(),
            Linear(50, 50),
            Tanh(),
            Linear(50, 50),
            Tanh(),
            Linear(50, 1),
        )

    def forward(self, states):
        return self.net(states)


class UTILS:
    def __init__(self) -> None:
        pass

    def get_flat_grads(self, f, net):
        flat_grads = torch.cat(
            [
                grad.view(-1)
                for grad in torch.autograd.grad(f, net.parameters(), create_graph=True)
            ]
        )

        return flat_grads

    def get_flat_params(self, net):
        return torch.cat([param.view(-1) for param in net.parameters()])

    def set_params(self, net, new_flat_params):
        start_idx = 0
        for param in net.parameters():
            end_idx = start_idx + np.prod(list(param.shape))
            param.data = torch.reshape(new_flat_params[start_idx:end_idx], param.shape)

            start_idx = end_idx

    def conjugate_gradient(self, Av_func, b, max_iter=10, residual_tol=1e-10):
        x = torch.zeros_like(b)
        r = b - Av_func(x)
        p = r
        rsold = r.norm() ** 2

        for _ in range(max_iter):
            Ap = Av_func(p)
            alpha = rsold / torch.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = r.norm() ** 2
            if torch.sqrt(rsnew) < residual_tol:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew

        return x

    def rescale_and_linesearch(
        self, g, s, Hs, max_kl, L, kld, old_params, pi, max_iter=10, success_ratio=0.1
    ):
        self.set_params(pi, old_params)
        L_old = L().detach()

        beta = torch.sqrt((2 * max_kl) / torch.dot(s, Hs))

        for _ in range(max_iter):
            new_params = old_params + beta * s

            self.set_params(pi, new_params)
            kld_new = kld().detach()

            L_new = L().detach()

            actual_improv = L_new - L_old
            approx_improv = torch.dot(g, beta * s)
            ratio = actual_improv / approx_improv

            if ratio > success_ratio and actual_improv > 0 and kld_new < max_kl:
                return new_params

            beta *= 0.5

        print("The line search was failed!")
        return old_params
