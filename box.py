import numpy as np
# from sklearn import preprocessing
import zono

class Box(object):

    def __init__(self, LB, UB):
        n1, m1 = LB.shape
        n2, m2 = UB.shape

        if m1 != 1 | m2 != 1:
            raise Exception("LB and UB should be a vector")

        if n1 != n2:
            raise Exception("Inconsistent dimensions between LB and UB")

        self.lb = LB
        self.ub = UB
        self.dim = len(LB)
        self.center = 0.5*(UB+LB)
        self.generators = None
        vec = 0.5*(UB-LB)

        for i in range(n1):
            if vec[i] != 0:
                gen = np.zeros((n1,1))
                gen[i] = vec[i]
                if self.generators is None:
                    self.generators = gen
                else:
                    self.generators = np.concatenate((self.generators, gen), axis=1)

        if np.linalg.norm(vec) == 0:
            self.generators = np.zeros((self.dim, 1))

    def toZono(self):
        z = zono.Zono(self.center, self.generators)
        return z

    def toStar(self):
        Z = self.toZono()
        return Z.toStar()
