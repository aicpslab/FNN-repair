# nnv/code/nnv/engine/set/ImageStar.m

import numpy as np
import zono
import star
# import torch
# import torch.nn as nn
import scipy


class ImageStar(object):
    '''
    Class for representing set of images using Star set
    An
    '''
    def __init__(self, *args):
        nargs = len(args)
        self.MaxIdxs = []
        self.InputSizes = []

        if nargs == 3:
            IM1 = args[0].copy()
            LB1 = args[1].copy()
            UB1 = args[2].copy()
            n = IM1.shape
            l = LB1.shape
            u = UB1.shape

            if n[0] != l[0] | n[0] != u[0] | n[1] != l[1] | n[1] != u[1]:
                raise Exception("Inconsistency between center image and attack bound matrices")

            if len(n) != len(l) | len(n) != len(u):
                Exception("Inconsistency between center image and attack bound matrices")

            if len(n) == 2 & len(l) == 2 & len(u) == 2:
                self.numChannel = 1
                self.IM = IM1
                self.LB = LB1
                self.UB = UB1
                self.height = n[0]
                self.width = n[1]
            elif len(n) == 3 & len(l) == 3 & len(u) == 3:
                if n[2] == l[2] & n[2] == u[2]:
                    self.numChannel = 1
                    self.IM = IM1
                    self.LB = LB1
                    self.UB = UB1
                    self.height = n[0]
                    self.width = n[1]
                else:
                    raise Exception("Inconsistency between center image and attack bound matrices")
            else:
                raise Exception("Inconsistency between center image and attack bound matrices")
            self.im_lb = IM1 + LB1
            self.im_ub = IM1 + UB1

            n_lb = self.im_lb.shape
            if len(n_lb) == 3:
                I = star.Star(self.im_lb.reshape([n_lb[0]*n_lb[1]*n_lb[2],1]),self.im_ub.reshape([n_lb[0]*n_lb[1]*n_lb[2],1]))
                self.V = I.V.reshape([n_lb[0],n_lb[1],n_lb[2],I.nVar+1]).astype(np.single)
            else:
                I = star.Star(self.im_lb.reshape([n_lb[0] * n_lb[1], 1]), self.im_ub.reshape([n_lb[0] * n_lb[1], 1]))
                self.V = I.V.reshape([n_lb[0],n_lb[1],1,I.nVar+1]).astype(np.single)
            self.C = I.C.astype(np.single).copy()
            self.d = I.d.astype(np.single).copy()
            self.pred_lb = I.predicate_lb.copy()
            self.pred_ub = I.predicate_ub.copy()
            self.numPred = I.nVar

        elif nargs == 5:
            V1 = args[0].copy()
            C1 = args[1].copy()
            d1 = args[2].copy()
            lb1 = args[3].copy()
            ub1 = args[4].copy()

            if C1.shape[0] != d1.shape[0]:
                raise Exception("Inconsistent dimension between constraint matrix and constraint vector")
            if len(d1.shape) != 1:
                if d1.shape[1] != 1:
                    raise Exception("Invalid constraint vector, vector should have one column")

            self.numPred = C1.shape[1]
            self.C = C1
            self.d = d1

            if C1.shape[1] != lb1.shape[0] | C1.shape[1] != ub1.shape[0]:
                raise Exception("Number of predicates is different from the size of the lower/upper bound predicate vector")
            if lb1.shape[1] != 1 | ub1.shape[1]!= 1:
                raise Exception("Invalid lower/upper bound predicate vector, vector should have one column")

            self.pred_lb = lb1
            self.pred_ub = ub1
            # self.im_lb = None
            # self.im_ub = None

            n = V1.shape
            if len(n) == 3:
                self.V = V1
                self.height = n[0]
                self.width = n[1]
                self.numChannel = n[2]
            elif len(n) == 4:
                if n[3] != self.numPred +1:
                    raise Exception("Inconsistency between the basis matrix and the number of predicate variables")
                else:
                    self.numChannel = n[2]
                    self.V = V1
                    self.height = n[0]
                    self.width = n[1]
            elif len(n) ==2:
                self.numChannel = 1
                self.V = V1
                self.height = n[0]
                self.width = n[1]
            else:
                raise Exception("Invalid basis matrix")

        elif nargs == 7:
            V1 = args[0].copy()
            C1 = args[1].copy()
            d1 = args[2].copy()
            lb1 = args[3].copy()
            ub1 = args[4].copy()
            im_lb1 = args[5].copy()
            im_ub1 = args[6].copy()

            if C1.shape[0] != d1.shape[0]:
                raise Exception("Inconsistent dimension between constraint matrix and constraint vector")
            if d1.shape[1] != 1:
                raise Exception("Invalid constraint vector, vector should have one column")

            self.numPred = C1.shape[1]
            self.C = C1
            self.d = d1

            if C1.shape[1] != lb1.shape[0] | C1.shape[1] != ub1.shape[0]:
                raise Exception("Number of predicates is different from the size of the lower/upper bound predicate vector")
            if lb1.shape[1] != 1 | ub1.shape[1] != 1:
                raise Exception("Invalid lower/upper bound predicate vector, vector should have one column")

            self.pred_lb = lb1
            self.pred_ub = ub1

            n = V1.shape
            if len(n) == 3:
                self.numPred = 0
                self.V = V1
                self.height = n[0]
                self.width = n[1]
                self.numChannel = n[2]
            elif len(n) == 4:
                if n[3] != self.numPred +1:
                    raise Exception("Inconsistency between the basis matrix and the number of predicate variables")
                else:
                    self.numChannel = n[3]
                    self.V = V1
                    self.height = n[0]
                    self.width = n[1]
            elif len(n) == 2:
                self.numChannel = 1
                self.numPred = 0
                self.V = V1
                self.height = n[0]
                self.width = n[1]
            else:
                raise Exception("Invalid basis matrix")

            if im_lb1 is not None:
                if im_lb1.shape[0] != self.height | im_lb1.shape[1] != self.width:
                    raise Exception("Inconsistent dimension between lower bound image and the constructed imagestar")
            else:
                self.im_lb = im_lb1

            if im_ub1 is not None:
                if im_ub1.shape[0] != self.height | im_ub1.shape[1] != self.width:
                    raise Exception("Inconsistent dimension between upper bound image and the constructed imagestar")
            else:
                self.im_ub = im_ub1

        else:
            raise Exception("Invalid number of input arguments (should be 0/3/5/7)")


    def evaluate(self, pred_val):
        if self.V is None:
            Exception("The ImageStar is an empty set")
        if pred_val.shape[1] != 1:
            Exception("Invalid predicate vector")
        if pred_val.shape[0] != self.numPred:
            Exception("Inconsistency between the size of the predicate vector "
                      "and the number or predicate in hte ImageStar")

        image = np.zeros((self.height, self.width, self.numChannel))
        for i in range(self.numChannel):
            image[:,:,i] = self.V[:,:,i,0]
            for j in range(1,self.numPred+1):
                image[:,:,i] += pred_val[j-1]*self.V[:,:,i,j]

        return image


    def toStar(self):
        nc = self.numChannel
        h = self.height
        w = self.width
        np1 = self.numPred

        N = h*w*nc
        V1 = np.zeros((N,np1+1), dtype=np.single)
        for j in range(np1+1):
            # print(V1[:,j].shape)
            # print(self.V[:,:,:,j].reshape(N).shape)
            V1[:,j] = self.V[:,:,:,j].reshape(N)
        if (hasattr(self, 'im_lb')) & (hasattr(self, 'im_ub')):
            state_lb = self.im_lb.reshape((N,1))
            state_ub = self.im_ub.reshape((N,1))
            S = star.Star(V1, self.C, self.d, self.pred_lb, self.pred_ub, state_lb, state_ub)
        else:
            S = star.Star(V1, self.C, self.d, self.pred_lb, self.pred_ub)

        return S


    def affineMap(self, scale, offset):

        if scale.shape[2] != self.numChannel:
            Exception("Inconsistent number of channels between scale array and the ImageStar")

        if scale is not None:
            new_V = scale * self.V
        else:
            new_V = self.V

        if offset is not None:
            new_V[:,:,:,0] = new_V[:,:,:,0] + offset

        image = ImageStar(new_V, self.C, self.d, self.pred_lb, self.pred_ub)
        return image

    def addMaxIdx(self, name, maxIdx):
        self.MaxIdxs.append([name, maxIdx.copy()])

    def addInputSize(self, name, inputSize):
        self.InputSizes.append([name, inputSize.copy()])

    def inheritMaxIdx(self, maxidx):
        self.MaxIdxs = maxidx.copy()

    def inheritInputSize(self, inputsize):
        self.InputSizes = inputsize.copy()

    def isMax(*args):
        nargs = len(args)
        if nargs == 4:
            maxMap = args[0]
            ori_image = args[1]
            center = args[2]
            others = args[3]
            lp_solver = 'linprog'
        elif nargs == 5:
            maxMap = args[0]
            ori_image = args[1]
            center = args[2]
            others = args[3]
            lp_solver = args[4]

        if maxMap.numPred != ori_image.numPred:
            raise Exception('Inconsistency between number of predicates in the current maxMap and original image!')

        n = others.shape[0]
        # the center may be the max point with some extra constraints on the predicate variables
        new_C = np.zeros((n, maxMap.numPred))
        new_d = np.zeros((n, 1))

        for i in range(n):
            # add new constraint
            new_d[i] = ori_image.V[center[0], center[1], center[2], 0] - ori_image.V[others[i, 0], others[i, 1], others[i, 2], 0]
            for j in range(maxMap.numPred):
                new_C[i, j] = -ori_image.V[center[0], center[1], center[2], j+1] + ori_image.V[others[i, 0], others[i, 1], others[i, 2], j+1]

        C1 = np.concatenate((maxMap.C, new_C))
        if len(maxMap.d.shape) == 1:
            d1 = np.concatenate((maxMap.d[:, np.newaxis], new_d))
        else:
            d1 = np.concatenate((maxMap.d, new_d))
        # remove redundant constraints
        E = np.concatenate((C1, d1), axis=1)
        E = np.unique(E, axis=0)
        C1 = E[:, 0:ori_image.numPred]
        d1 = E[:, ori_image.numPred]
        f = np.zeros((1, ori_image.numPred))

        if lp_solver == 'linprog':
            bound = []
            for i in range(ori_image.pred_lb.shape[0]):
                bound.append((ori_image.pred_lb[i, 0], ori_image.pred_ub[i, 0]))
            r = scipy.optimize.linprog(f, C1, d1, method='highs', bounds=bound)
            if r.success:
                new_C = C1
                new_d = d1
            else:
                new_C = None
                new_d = None

        return new_C, new_d

    def updateMaxIdx(self, name, maxIdx, pos):
        n = len(self.MaxIdxs)
        ct = 0
        for i in range(n):
            if self.MaxIdxs[i][0] == name:
                temp1 = self.MaxIdxs[i]
                temp2 = self.MaxIdxs[i][1]
                self.MaxIdxs[i][1][pos[0]][pos[1]][pos[2]] = maxIdx
                break
            else:
                ct += 1
        if ct == n:
            raise Exception('Unknown name of the maxpooling layer')


    def getRange(self, *args):
        nargs = len(args)
        if nargs == 3:
            vert_ind = args[0]
            horiz_ind = args[1]
            chan_ind = args[2]
            lp_solver = 'linprog'
        elif nargs ==4:
            vert_ind = args[0]
            horiz_ind = args[1]
            chan_ind = args[2]
            lp_solver = args[3]
        else:
            Exception("Invalid numebr of input arguments, should be 3/4")

        if (vert_ind < 0) | (vert_ind > self.height):
            raise Exception("Invalid vertical index")

        if (horiz_ind < 0)| (horiz_ind > self.width):
            raise Exception("Invalid horizonal index")

        if (chan_ind < 0) | (chan_ind > self.numChannel):
            raise Exception("Invalid channel index")

        f = self.V[vert_ind, horiz_ind, chan_ind, 1:self.numPred+1]

        # use Gorubi for linprog or glpk
        if lp_solver == 'linprog':
            bound = []
            for i in range(self.pred_lb.shape[0]):
                bound.append((self.pred_lb[i, 0], self.pred_ub[i, 0]))
            r = scipy.optimize.linprog(f, self.C, self.d, method='highs', bounds=bound)
            if r.success:
                fval = np.dot(f, r.x.T)
                xmin = fval + self.V[vert_ind, horiz_ind, chan_ind, 0]
            else:
                Exception("Can't find an optimal solution")

            r = scipy.optimize.linprog(-f, self.C, self.d, method='highs', bounds=bound)
            if r.success:
                fval = np.dot(-f, r.x.T)
                xmax = -fval + self.V[vert_ind, horiz_ind, chan_ind, 0]
            else:
                raise Exception("Can't find an optimal solution")
        elif lp_solver == 'glpk':
            raise Exception("Can't use glpk method")
        else:
            raise Exception("Unknown lp solver method")

        # if (hasattr(self, 'im_lb') is False) & (hasattr(self, 'im_ub') is False):
        #     self.im_lb = np.empty((self.height, self.width, self.numChannel))
        # else:
        self.im_lb[vert_ind, horiz_ind, chan_ind] = xmin
        self.im_ub[vert_ind, horiz_ind, chan_ind] = xmax

        return xmin, xmax


    def getRanges(self, *args):

        nargs = len(args)
        if nargs == 0:
            lp_solver = 'linprog'
        elif nargs == 1:
            lp_solver = args[0]
        else:
            raise Exception("Invalid number of input arguments")

        image_lb = np.zeros((self.height, self.width, self.numChannel))
        image_ub = np.zeros((self.height, self.width, self.numChannel))

        reverseStr = ''
        N = self.height*self.width*self.numChannel

        for i in range(self.height):
            for j in range(self.width):
                for k in range(self.numChannel):
                    image_lb[i,j,k], image_ub[i,j,k] = self.getRange(i,j,k,lp_solver)

        self.im_lb = image_lb
        self.im_ub = image_ub

        return image_lb, image_ub


    def estimateRange(self, vert_ind, horiz_ind, chan_ind):
        if vert_ind <1 | vert_ind>self.height:
            Exception("Invalid vertical index")
        if horiz_ind<1 | horiz_ind>self.width:
            Exception("Invalid horizontal index")
        if chan_ind<1 | chan_ind>self.numChannel:
            Exception("Invalid channel index")

        f = self.V[vert_ind, horiz_ind, chan_ind, 0:self.numPred+1]
        xmin = f[0]
        xmax = f[0]

        for i in range(1,self.numPred+1):
            if f[i] >= 0:
                xmin += f[i]*self.pred_lb[i-1]
                xmax += f[i]*self.pred_ub[i-1]
            else:
                xmin += f[i]*self.pred_ub[i-1]
                xmax += f[i]*self.pred_lb[i-1]

        return xmin, xmax


    def estimateRanges(self,*args):
        nargs = len(args)
        if nargs == 0:
            dis_opt = []
        elif nargs == 1:
            dis_opt = args[0]
        else:
            Exception("Invalid number of input, should be 0/1")

        if (hasattr(self, 'im_lb') is False) | (hasattr(self, 'im_ub') is False):
            image_lb = np.zeros((self.height, self.width, self.numChannel))
            image_ub = np.zeros((self.height, self.width, self.numChannel))
            N = self.height*self.width*self.numChannel
            for i in range(self.height):
                for j in range(self.width):
                    for k in range(self.numChannel):
                        image_lb[i,j,k], image_ub[i,j,k] = self.estimateRange(i,j,k)
            self.im_lb = image_lb
            self.im_ub = image_ub
        else:
            image_lb = self.im_lb
            image_ub = self.im_ub

        return image_lb, image_ub



    def updateRanges(self,*args):
        nargs = len(args)
        if nargs == 1:
            points = args[0]
            lp_solver = 'glpk'
        elif nargs == 2:
            points = args[0]
            lp_solver = args[1]
        else:
            raise Exception("Invalid number of input, should be 1/2")

        n = points.shape[0]
        for i in range(n):
            self.getRange(points[i,0], points[i,1], points[i,2], lp_solver)


    def get_localBound(self, *args):

        nargs = len(args)
        if nargs == 3:
            startpoint = args[0]
            PoolSize = args[1]
            channel_id = args[2]
            lp_solver = 'glpk'
        elif nargs == 4:
            startpoint = args[0]
            PoolSize = args[1]
            channel_id = args[2]
            lp_solver = args[3]
        else:
            Exception("Invalid number of input arguments, should be 3/4")

        points = self.get_localPoints(startpoint, PoolSize)
        n = len(points)

        if (self.im_lb is None) | (self.im_ub is None):
            image_lb, image_ub = self.getRanges(lp_solver)
        else:
            image_lb = self.im_lb
            image_ub = self.im_ub

        lb = image_lb[points[0,0], points[0,1], channel_id]
        ub = image_ub[points[0,0], points[0,1], channel_id]

        for i in range(1,n):
            if image_lb[points[i,0], points[i,1], channel_id] < lb:
                lb = image_lb[points[i,0], points[i,1], channel_id]
            if image_ub[points[i,0], points[i,1], channel_id] > ub:
                ub = image_ub[points[i,0], points[i,1], channel_id]

        return lb, ub


    def get_localPoints(self, startpoint, PoolSize):

        x0 = int(startpoint[0])
        y0 = int(startpoint[1])
        h = PoolSize[0]
        w = PoolSize[1]

        if (x0<0) | (y0<0)| (x0+h > self.height) | (y0+w > self.width):
            raise Exception("Invalid startpoint or Poolsize")

        points = np.zeros((h*w,2)).astype(int)
        for i in range(h):
            if i == 0:
                x1 = x0
            else:
                x1 += 1

            for j in range(w):
                if j == 0:
                    y1 = y0
                else:
                    y1 += 1
                points[i*w+j, :] = [x1, y1]

        return points


    def get_localMax_index(self, *args):
        nargs = len(args)
        if nargs == 3:
            startpoint = args[0]
            PoolSize = args[1]
            channel_id = args[2]
            lp_solver = 'glpk'
        elif nargs == 4:
            startpoint = args[0]
            PoolSize = args[1]
            channel_id = args[2]
            lp_solver = args[3]
        else:
            raise Exception("Invalid number of input arguments, should be 3/4")

        points = self.get_localPoints(startpoint, PoolSize)
        if (hasattr(self, 'im_lb') is False) | (hasattr(self, 'im_ub') is False):
            self.estimateRanges()

        h = PoolSize[0]
        w = PoolSize[1]
        n = h*w

        lb = np.zeros((1,n))
        ub = np.zeros((1,n))

        for i in range(n):
            point_i = points[i,:]
            lb[0, i] = self.im_lb[point_i[0], point_i[1], channel_id]
            ub[0, i] = self.im_ub[point_i[0], point_i[1], channel_id]

        max_lb_val = lb.max(axis=1)
        max_lb_ind = lb.argmax(axis=1) # return a row vector
        # a = np.argwhere(ub - max_lb_val > 0).flatten() # return a row vector
        a_ind = np.argwhere(ub - max_lb_val > 0) # it gives index of the element in 2d, so need to change it
        a = np.zeros(a_ind.shape[0], dtype=int)
        for i in range(a_ind.shape[0]):
            a[i] = a_ind[i, 1]
        a1_ind = np.argwhere(ub - max_lb_val >= 0)
        a1 = np.zeros(a1_ind.shape[0], dtype=int)
        for i in range(a1_ind.shape[0]):
            a1[i] = a1_ind[i, 1]

        if (len(a) == 0) | ((len(a) > 0) & (a == max_lb_ind).all()):
            max_id = points[max_lb_ind, :].flatten()
            max_id.astype(int)
        else:
            candidates = a1
            m = len(candidates)
            new_points = np.zeros((m,3), dtype=int)
            new_points1 = np.zeros((m,2), dtype=int)
            for i in range(m):
                p = points[candidates[i],:]
                new_points[i, :] = np.append(p, [channel_id])
                # new_points[i, :] = np.array([p, channel_id])
                new_points1[i,:] = p
            self.updateRanges(new_points, lp_solver)

            lb = np.zeros((1, m))
            ub = np.zeros((1, m))

            for i in range(m):
                point_i = points[candidates[i], :]
                lb[:, i] = self.im_lb[point_i[0], point_i[1], channel_id]
                ub[:, i] = self.im_ub[point_i[0], point_i[1], channel_id]
            max_lb_val = lb.max(axis=1)
            max_lb_ind = lb.argmax(axis=1)
            a_ind = np.argwhere(ub - max_lb_val > 0) # it gives index of the element in 2d, so need to change it
            a = np.zeros(a_ind.shape[0], dtype=int)
            for i in range(a_ind.shape[0]):
                a[i] = a_ind[i, 1]

            if (a == max_lb_ind).all():
                max_id = new_points1[max_lb_ind, :].flatten()
                max_id.astype(int)
            else:
                candidates1_ind = np.argwhere(ub - max_lb_val >= 0)
                candidates1 = np.zeros(candidates1_ind.shape[0], dtype=int)
                for i in range(candidates1_ind.shape[0]):
                    candidates1[i] = candidates1_ind[i, 1]
                max_id = new_points1[max_lb_ind, :].flatten()
                candidates1 = candidates1[candidates1 != max_lb_ind]
                m = len(candidates1)
                max_id1 = max_id
                for j in range(m):
                    p1 = new_points1[candidates1[j],:]
                    if self.is_p1_larger_p2([p1[0], p1[1], channel_id],[max_id[0],max_id[1], channel_id], lp_solver):
                        if len(max_id1.shape) == 1:
                            max_id1 = np.concatenate((max_id1[np.newaxis, :], p1[np.newaxis, :]))
                        else:
                            max_id1 = np.concatenate((max_id1, p1[np.newaxis, :]))
                max_id = max_id1
        if len(max_id.shape) == 1:
            max_id = np.append(max_id, channel_id)
        else:
            n = max_id.shape[0]
            channels = channel_id*np.ones((n,1), dtype=int)
            max_id = np.concatenate((max_id,channels),axis=1)

        return max_id


    def is_p1_larger_p2(self, *args):
        nargs = len(args)
        if nargs == 2:
            p1 = args[0]
            p2 = args[1]
            lp_solver = 'linprog'
        elif nargs == 3:
            p1 = args[0]
            p2 = args[1]
            lp_solver = args[2]
        else:
            Exception("Invalid number of input, should be 2/3")

        C1 = np.zeros((1,self.numPred))
        for i in range(1, self.numPred+1):
            C1[0,i-1] = self.V[p2[0], p2[1], p2[2], i] - self.V[p1[0], p1[1], p1[2], i]

        d1 = self.V[p1[0], p1[1], p1[2], 0] - self.V[p2[0], p2[1], p2[2], 0]
        new_C = np.concatenate((self.C, C1), axis=0)
        # new_d = np.concatenate((self.d, d1), axis=1)
        new_d = np.insert(self.d, self.d.shape[0], d1, axis=0)
        f = np.zeros((1, self.numPred))

        if lp_solver == 'linprog':
            bound = []
            for i in range(self.pred_lb.shape[0]):
                bound.append((self.pred_lb[i, 0], self.pred_ub[i, 0]))
            r = scipy.optimize.linprog(f, new_C, new_d, method='highs', bounds=bound)
            if r.success:
                b = 1
            else:
                b = 0

        return b


