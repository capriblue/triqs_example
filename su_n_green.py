
import numpy as np
from fermion import FermionMatrix
from scipy.linalg import expm


class GreenFunc:
    def __init__(self, beta=1,U_matrix=np.array([[20,20],[20,20]]), flavor="auto",  bath_num=1, eps_k=[0],V_k=[10]):
        self.beta = beta 
        self.U_matrix = U_matrix
        if flavor=="auto":
            flavor = U_matrix.shape[0]
            self.flavor = flavor
        else:
            self.flavor = flavor 
        self.bath_num = bath_num
        m = bath_num
        f = FermionMatrix(m+1, self.flavor)
        Himp = f.zero()
        for i,r in enumerate(U_matrix):
            for j,c in enumerate(r):
                if i == j:
                    Himp += -1 * c * f.n(0,i)
                else:
                    Himp += c * f.n(0,i) @ f.n(0,j)
        for k in range(1, m+1):
            for spin in range(flavor):
                Himp += eps_k[k-1] * f.n(k,spin)
                Himp += V_k[k-1] *( f.cdag(0, spin)@ f.c(k, spin) + f.cdag(k, spin)@ f.c(0, spin))
        self.gHimp =Himp 
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

if __name__ == "__main__":
    beta = 1
    gf = GreenFunc(beta=beta)
    taus = np.linspace(0,beta,100)
    green = gf.greenarray(taus)
    np.savetxt("result.csv", np.column_stack([taus,-(green)]))

## すべてのフレーバーを出力するようにする
class GreenFunc2:
    def __init__(self, beta=1,U_matrix=np.array([[20,20],[20,20]]), flavor="auto",  bath_num=1, eps_k=[0],V_k=[10]):
        self.beta = beta 
        self.U_matrix = U_matrix
        if flavor=="auto":
            flavor = U_matrix.shape[0]
            self.flavor = flavor
        else:
            self.flavor = flavor 
        self.bath_num = bath_num
        m = bath_num
        f = FermionMatrix(m+1, self.flavor)
        Himp = f.zero()
        for i,r in enumerate(U_matrix):
            for j,c in enumerate(r):
                if i == j:
                    Himp += -1 * c * f.n(0,i)
                else:
                    Himp += c * f.n(0,i) @ f.n(0,j)
        for k in range(1, m+1):
            for spin in range(flavor):
                Himp += eps_k[k-1] * f.n(k,spin)
                Himp += V_k[k-1] *( f.cdag(0, spin)@ f.c(k, spin) + f.cdag(k, spin)@ f.c(0, spin))
        self.gHimp =Himp 
        self.bunbo = np.trace(expm( -beta *self.gHimp))
        impurity_c = []
        impurity_cdag =  []
        for i in range(self.flavor):
            impurity_c.append(f.c(0,i))
            impurity_cdag.append(f.cdag(0,i))
        self.i_c = impurity_c 
        self.i_cdag = impurity_cdag 
        
    def green(self, tau, flavor):
        bunshi =  -1 * np.trace(expm(self.gHimp * (tau - self.beta ) )@  self.i_c[flavor] @ expm(-tau * self.gHimp ) @ self.i_cdag[flavor])
        return bunshi /self.bunbo
    def greenarray(self, taus, flavor):
        taus = np.array(taus)
        gv = np.vectorize(self.green)
        return gv(taus, flavor)
    def getAllgreenarray(self, taus):
        g_list = []
        for i in range(self.flavor):
            g = self.greenarray(taus, i)
            g_list.append(g)
        return np.column_stack(g_list)