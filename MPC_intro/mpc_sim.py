#!/usr/bin/env python3
import numpy as np


dt = 0.1

x = np.array([0.0, 0.0, 0.0, 0.0])
target = np.array([10.0, 5.0, 0.0, 0.0])

A = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

B = np.array([
    [0, 0],
    [0, 0],
    [dt, 0],
    [0, dt]
])

Q = np.diag([10, 10, 1, 1])
R = np.diag([0.1, 0.1])

N = 10

u_candidates = [
    np.array([ax, ay])
    for ax in [-1, 0, 1]
    for ay in [-1, 0, 1]
]

def simulate(x, u_sequence):
    x_sim = x.copy()
    total_cost = 0

    for u in u_sequence:
        error = x_sim - target
        cost = error.T @ Q @ error + u.T @ R @ u
        total_cost += cost

        x_sim = A @ x_sim + B @ u

    return total_cost
    

for step in range(100):
    best_cost = float("inf")
    best_u = None

    for u0 in u_candidates:
        for u1 in u_candidates:
            for u2 in u_candidates:
                for u3 in u_candidates:
                    u_sequence = [u0, u1, u2, u3]

                    cost = simulate(x, u_sequence)

                    if(cost < best_cost):
                        best_cost = cost
                        best_u = u0

    x = A @ x + B @ best_u

    print(x[:2])