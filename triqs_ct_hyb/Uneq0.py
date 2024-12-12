import datetime
import pickle

import numpy as np
from triqs.gf import GfImFreq
from triqs.operators import n
from triqs.utility.mpi import bcast, is_master_node
from triqs_cthyb import Solver

if is_master_node():
    with open("./input.pickle", "rb") as f:
        data = pickle.load(f)
    U_matrix = data["U_matrix"]
    beta = data["beta"]
    g0_list = data["g0_list"]
else:
    U_matrix = None
    beta = None
    g0_list = None
U_matrix = bcast(U_matrix)
beta = bcast(beta)
g0_list = bcast(g0_list)

flavor = U_matrix.shape[0]
S = Solver(beta=beta, gf_struct=[(str(i), 1) for i in range(flavor)], n_l=200)

for i, val in enumerate(S.G0_iw):
    name, g0 = val
    g0 << g0_list[i]

h_int = 0
for i, row in enumerate(U_matrix):
    for j, u_ij in enumerate(row):
        if i == j:
            continue
        elif i > j:
            h_int += 2*u_ij * n(str(i), 0) * n(str(j), 0)

S.solve(h_int=h_int, length_cycle=100, n_cycles=100000)

if is_master_node():
    result = {
        "G_iw":  S.G_iw["0"].data.flatten(),
        "G_tau": S.G_tau["0"].data.flatten(),
        "param": data
    }
    now = datetime.datetime.now().strftime("%m%d_%H%M")
    with open(f"./r.pickle", "wb") as f:
        pickle.dump(result, f)
