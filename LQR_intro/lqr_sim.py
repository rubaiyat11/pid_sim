#!/usr/bin/env python3
import numpy as np
from scipy.linalg import solve_continuous_are
from collections import deque


dt = 0.01

x = np.array([0.0, 0.0, 0.0, 0.0])
target = np.array([10.0, 5.0, 0.0, 0.0])

A = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
])

m = 2.0
B= np.array([
    [0, 0],
    [0, 0],
    [1/m, 0],
    [0, 1/m]
])

Q = np.diag([10, 10, 2, 2])
R = np.diag([0.5, 0.5])

P = solve_continuous_are(A, B, Q, R)

K = np.linalg.inv(R) @ B.T @ P    

for i in range(1000):

    error = x - target

    u = -K @ error
    u = np.clip(u, -2.0, 2.0)

    x_dot = A @ x + B @ u

    x += x_dot * dt

    print(x[:2])