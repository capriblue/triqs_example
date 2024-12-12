""" 
不純物模型のグリーン関数を求めるプログラム
複数の値を得たい場合はGreenFuncクラスで
行列を事前に生成し、
その行列に対する演算をすると高速。
正直なところもう少し改良の余地はあるが面倒なので放棄
作者：藤井淳太朗
mail：capri4wrk@gmail.com
"""

import numpy as np
from fermion import FermionMatrix
from scipy.linalg import expm


class GreenFunc:
    def __init__(self, beta=10,U=2, mu=1, flavor=2,  bath_num=1, eps_k=np.full(1,3),V_k=np.full(1,1)):
        self.beta = beta 
        self.U = U 
        self.mu = mu 
        self.flavor = flavor 
        self.bath_num = bath_num
        m = bath_num
        f = FermionMatrix(m+1, flavor)
        Himp = U * f.n(0,0) @ f.n(0,1)
        for k in range(1, m+1):
            for spin in range(flavor):
                Himp += eps_k[k-1] * f.n(k,spin)
                Himp += V_k[k-1] *( f.cdag(0, spin)@ f.c(k, spin) + f.cdag(k, spin)@ f.c(0, spin))
        N = f.zero()
        # for i in range(m):
        #     for spin in range(flavor):
        #         N += f.n(i, spin)
        self.gHimp =Himp -  mu * N
        self.bunbo = np.trace(expm( -beta *self.gHimp))
        self.c00 = f.c(0,0)
        self.cdag00 = f.cdag(0,0)
    def green(self, tau):
        bunshi =  -1 * np.trace(expm(self.gHimp * (tau - self.beta ) )@  self.c00 @ expm(-tau * self.gHimp ) @ self.cdag00)
        return bunshi /self.bunbo
    def greenarray(self, taus):
        taus = np.array(taus)
        gv = np.vectorize(self.green)
        return gv(taus)
def green_func(tau, beta=10,U=2, mu=1, flavor=2,  bath_num=1, eps_k=np.full(4,3),V_k=np.full(4,1)):
    m = bath_num
    f = FermionMatrix(m+1, flavor)
    Himp = U * f.n(0,0) @ f.n(0,1)
    for k in range(1, m+1):
        for spin in range(flavor):
            Himp += eps_k[k-1] * f.n(k,spin)
            Himp += V_k[k-1] *( f.cdag(0, spin)@ f.c(k, spin) + f.cdag(k, spin)@ f.c(0, spin))
    N = f.zero()
    for i in range(m):
        for spin in range(flavor):
            N += f.n(i, spin)
    gHimp =Himp -  mu * N
    bunbo = np.trace(expm( -beta *gHimp))
    bunshi = -1 * np.trace(expm(gHimp * (tau - beta ) )@f.c(0,0) @ expm(-tau * gHimp ) @ f.cdag(0,0))
    return bunshi/bunbo
if __name__ == "__main__":
    beta = 30
    taus = np.linspace(0,beta,100)
    green = []
    for tau in taus:
        green.append(green_func(tau,beta))
    np.savetxt("result.csv", -np.array(green))