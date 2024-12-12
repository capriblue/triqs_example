"""
第二量子化された演算子の行列表現を求めるためのプログラム
エネルギーを求めるには、ハミルトニアンを対角化する必要があり、それには行列表現が必要である。

サイトの数とフレーバーの数を初期化リストとして持ち、入力することで演算子を作ることができる。
基底は次の形式とした
| n_a1, n_a2, n_a3, ...; n_b1, n_b2, n_b3, ...; ...>
これを右から2進数と解釈し、10進数に直したときの昇順でとっている。また順番は上に記載の通りである。フェルミオンの反交換関係に従って期数番目の演算子は条件により負符号となる。

example:
```
>>>f = FermionMatrix(1,2)
>>>f.cdag(0,0)
array([[0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [1., 0., 0., 0.],
       [0., 1., 0., 0.]])
>>>f.c(0,1)
array([[ 0.,  1.,  0.,  0.],
       [ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0., -1.],
       [ 0.,  0.,  0.,  0.]])
>>>f.n(0,0)
array([[0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 1., 0.],
       [0., 0., 0., 1.]])
```
"""

import numpy as np


class FermionMatrix:
    def __init__(self, total_site, total_flavor=2):
        self.tot_state = (2**(total_site * total_flavor))
        self.tot_site = total_site 
        self.tot_flavor = total_flavor
    def cdag(self, site, spin):
        m = np.zeros((self.tot_state,self.tot_state))
        place = spin * self.tot_site + site 
        for i in range(self.tot_state):
            bina = format(i, f'0{self.tot_site*self.tot_flavor}b')
            # bina = bin(i)[2:] # 0x111 となる0xを削除して純粋な2進数に
            if bina[place] == "0":
                sign = 1
                for str in bina[:place]:
                    if str == "1":
                        sign *= -1
                hop_to = int("0b" + bina[:place] + "1" + bina[(place+1):],2)
                m[hop_to, i] = sign 
            elif bina[place] == "1":
                continue
        return m
    def c(self, site, spin):
        mdag = self.cdag(site, spin)
        return mdag.T
    def n(self, site, spin):
        return self.cdag(site, spin) @ self.c(site, spin)
    def zero(self):
        return np.zeros((self.tot_state,self.tot_state))