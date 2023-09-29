import numpy as np
import box
import zono
import imageStar
from scipy import optimize
from matplotlib.patches import Polygon
import pypoman as pyp

class Star(object):

    def __init__(self, *args):
        narg = len(args)
        if narg == 7:
            None
        elif narg == 6:
            V = args[0]
            C = args[1]
            d = args[2]
            pred_lb = args[3]
            pred_ub = args[4]
            outer_zono = args[5]

            nV, mV = V.shape
            nC, mC = C.shape
            if len(d.shape) == 2:
                nd, md = d.shape
            elif len(d.shape) == 1:
                nd = d.shape[0]
                md = 1
            else:
                raise Exception("Invalid d shape")

            if outer_zono is not None:
                if isinstance(outer_zono, zono.Zono) is False:
                    raise Exception("Outer zonotope is not a Zono object")

            if outer_zono is not None:
                nZ = outer_zono.V.shape[0]
                if nZ != nV:
                    raise Exception("Inconsistent dimension between outer zonotope and star set")

            if mV != mC + 1:
                raise Exception("Inconsistency between basic matrix and constraint matrix")

            if nC != nd:
                raise Exception("Inconsistency between constraint matrix and constraint vector")

            if md != 1:
                raise Exception("constraint vector should have one column")

            if isinstance(pred_lb, np.ndarray) & isinstance(pred_ub, np.ndarray):
                n1, m1 = pred_lb.shape
                n2, m2 = pred_ub.shape
                if m1 != 1 | m2 != 1:
                    raise Exception("predicate lower- or upper-bounds vector should have one column")
                if n1 != n2 | n1 != mC:
                    raise Exception("Inconsistency between number of predicate variables and predicate lower- or upper-bounds vector")

            self.V = V
            self.C = C
            self.d = d
            self.dim = nV
            self.nVar = mC
            self.predicate_lb = pred_lb
            self.predicate_ub = pred_ub
            self.Z = outer_zono
        elif narg == 5:
            V = args[0]
            C = args[1]
            d = args[2]
            pred_lb = args[3]
            pred_ub = args[4]

            nV, mV = V.shape
            nC, mC = C.shape
            if len(d.shape) == 2:
                nd, md = d.shape
            elif len(d.shape) == 1:
                nd = d.shape[0]
                md = 1
            else:
                raise Exception("Invalid d shape")

            if mV != mC + 1:
                raise Exception("Inconsistency between basic matrix and constraint matrix")

            if nC != nd:
                raise Exception("Inconsistency between constraint matrix and constraint vector")

            if md != 1:
                raise Exception("constraint vector should have one column")

            if isinstance(pred_lb, np.ndarray) & isinstance(pred_ub, np.ndarray):
                n1, m1 = pred_lb.shape
                n2, m2 = pred_ub.shape
                if m1 != 1 | m2 != 1:
                    raise Exception("predicate lower- or upper-bounds vector should have one column")
                if n1 != n2 | n1 != mC:
                    raise Exception("Inconsistency between number of predicate variables and predicate lower- or upper-bounds vector")

            self.V = V
            self.C = C
            self.d = d
            self.dim = nV
            self.nVar = mC
            self.predicate_lb = pred_lb
            self.predicate_ub = pred_ub
        elif narg == 2:
            LB = args[0]
            UB = args[1]

            B = box.Box(LB,UB)
            S = B.toStar()
            self.V = S.V
            self.C = np.zeros((1, S.nVar))
            self.d = np.zeros((1,1))
            self.dim = S.dim
            self.nVar = S.nVar
            self.state_lb = LB
            self.state_ub = UB
            self.predicate_lb = -np.ones((S.nVar, 1))
            self.predicate_ub = np.ones((S.nVar, 1))
            self.Z = B.toZono

        else:
            raise Exception("Invalid number of input arguments (should be 0 or 2 or 3 or 5 or 6)")


    def toImageStar(self, height, width, numChannel):

        if self.dim != height*width*numChannel:
            Exception("Inconsistent dimension in the ImageStar and the original Star set")

        IS = imageStar.ImageStar(self.V.reshape((height, width, numChannel, self.nVar+1)), self.C, self.d,
                                 self.predicate_lb, self.predicate_ub)

        return IS

    def toPolyhedron(self):
        bias = self.V[:, 0]
        weight = self.V[:, 1:self.nVar+1]

        if hasattr(self, 'predicate_ub'):
            C1 = np.concatenate((np.eye(self.nVar), -np.eye(self.nVar)))
            d1 = np.concatenate((self.predicate_ub, -self.predicate_lb))
            A = np.concatenate((self.C, C1))
            temp = self.d.copy()
            if len(temp.shape) == 1:
                b = np.concatenate((temp[:, np.newaxis], d1))
            else:
                b = np.concatenate((self.d, d1))
            Pa = pyp.compute_polytope_vertices(A, b.flatten())
            temp1 = bias[:, np.newaxis]
            temp2 = np.dot(weight, np.array(Pa).T)
            temp2 = np.unique(temp2, axis=1)
            P = temp2 + temp1
        else:
            Pa = pyp.compute_polytope_vertices(self.C, self.d)
            P = np.unique(np.dot(weight, np.array(Pa).T), axis=1) + bias[:, np.newaxis]

        return P

    def getRange(self, index):
        if (index < 0) | (index > self.dim):
            raise Exception("Invalid index")

        f = self.V[index, 1:(self.nVar+1)]
        if all(f[:] == 0):
            xmin = self.V[index, 0]
            xmax = self.V[index, 0]
        else:
            bound = []
            for i in range(self.predicate_lb.shape[0]):
                bound.append((self.predicate_lb[i, 0], self.predicate_ub[i, 0]))
            r = optimize.linprog(f, self.C, self.d, method='highs', bounds=bound)
            if r.success:
                fval = np.dot(f, r.x.T)
                xmin = fval + self.V[index, 0]
            else:
                Exception("Can't find an optimal solution")

            r = optimize.linprog(-f, self.C, self.d, method='highs', bounds=bound)
            if r.success:
                fval = np.dot(-f, r.x.T)
                xmax = -fval + self.V[index, 0]
            else:
                raise Exception("Can't find an optimal solution")

        return xmin, xmax

    def estimateRanges(self):

        pos_mat = self.V.copy()
        neg_mat = self.V.copy()
        pos_mat[pos_mat < 0] = 0
        neg_mat[neg_mat > 0] = 0
        new_lb = np.concatenate((np.zeros((1,1)), self.predicate_lb))
        new_ub = np.concatenate((np.zeros((1,1)), self.predicate_ub))
        xmin1 = np.dot(pos_mat, new_lb).squeeze()
        xmax1 = np.dot(pos_mat, new_ub).squeeze()
        xmin2 = np.dot(neg_mat, new_ub).squeeze()
        xmax2 = np.dot(neg_mat, new_lb).squeeze()
        lb = self.V[:, 0] + xmin1 + xmin2
        ub = self.V[:, 0] + xmax1 + xmax2
        # if (ub > lb).all():
        #     print(True)
        # else:
        #     print(False)

        return lb.flatten(), ub.flatten()

    def getMax(self, idx):
        index = idx
        f = self.V[index, 1:self.nVar+1]
        if (f[:] == 0).all():
            xmax = self.V[index,0]
        else:
            bound = []
            for i in range(self.predicate_lb.shape[0]):
                bound.append((self.predicate_lb[i, 0], self.predicate_ub[i, 0]))
            r = optimize.linprog(-f, self.C, self.d, bounds=bound)
            if r.success:
                fval = np.dot(-f, r.x.T)
                xmax = -fval + self.V[index,0]
            else:
                raise Exception("Can't find an optimal solution")

        return xmax

    def getMaxs(self, *args):
        map = args[0]
        n = len(map)
        xmax = np.zeros((n,1))

        for i in range(n):
            xmax[i,0] = self.getMax(map[i])
        return xmax

    def getMin(self, idx):
        index = idx
        f = self.V[index, 1:self.nVar+1]
        if (f[:] == 0).all():
            xmin = self.V[index, 0]
        else:
            bound = []
            for i in range(self.predicate_lb.shape[0]):
                bound.append((self.predicate_lb[i, 0], self.predicate_ub[i, 0]))
            r = optimize.linprog(f, self.C, self.d, bounds=bound)
            if r.success:
                fval = np.dot(f, r.x.T)
                xmin = fval + self.V[index,0]
            else:
                raise Exception("Can't find an optimal solution")
        return xmin

    def getMins(self, *args):
        map = args[0]
        n = len(map)
        xmin = np.zeros((n,1))

        for i in range(n):
            xmin[i,0] = self.getMin(map[i])
        return xmin

    def resetRow(self, map):
        V1 = self.V.copy()
        V1[map,:] = 0
        if hasattr(self, 'Z'):
            c2 = self.Z.c.copy()
            c2[map,:] = 0
            V2 = self.Z.V.copy()
            V2[map, :] = 0
            new_Z = zono.Zono(c2,V2)
        else:
            new_Z = None

        S = Star(V1, self.C.copy(), self.d.copy(), self.predicate_lb.copy(), self.predicate_ub.copy(), new_Z)
        return S

