import numpy as np
import star

class Zono(object):

    def __init__(self, *args):
        nargs = len(args)
        if nargs == 2:
            c = args[0]
            V = args[1]
            nC, mC = c.shape
            nV = V.shape[0]

            if mC != 1:
                raise Exception("center vector should have one column")
            if nC != nV:
                raise Exception("Inconsistent dimension between center vector and generator matrix")

            self.c = c
            self.V = V
            self.dim = nV

        elif nargs == 0:
            None
        else:
            raise Exception("Invalid number of inputs, 0 or 2")

    def toStar(self):
        n = self.V.shape[1]
        lb = -np.ones((n,1))
        ub = np.ones((n,1))
        C = np.concatenate((np.eye(n),-np.eye(n)))
        d = np.ones((2*n,1))
        new_V = np.concatenate((self.c,self.V),axis=1)
        s = star.Star(new_V, C, d, lb, ub, self)
        return s
