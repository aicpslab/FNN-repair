
import numpy as np
import imageStar
import layer
import star
import zono
import torch
import torch.nn as nn
import math


class ImageInputLayer(object):

    def __init__(self, *args):
        nargs = len(args)
        if nargs%2 != 0:
            Exception("Invalid number of input arguments")

        for i in range(0,nargs-1,2):
            if args[i] == 'Name':
                self.Name = args[i+1]
            elif args[i] == 'InputSize':
                self.InputSize = args[i+1]
            elif args[i] == 'Mean':
                self.Mean = args[i+1]

    def evaluate(self, input):

        if self.Mean is None:
            y = input
        else:
            y = input - self.Mean

        return y

    def reach_star_single_input(self, in_image):

        if isinstance(in_image, imageStar.ImageStar) is False:
            Exception("Input is not an ImageStar")
        image = in_image.affineMap([], -self.Mean)
        return image

    def reach_star_multipleInputs(self, in_images, option):

        n = len(in_images)
        images = []
        for i in range(n):
            images.append(self.reach_star_single_input(in_images[i]))
        return images

    # def reach_zono(self, in_image):
    #     if isinstance(in_image, zono.Zono) is False:
    #         Exception("Input is not an ImageZono")
    #     image = in_image.affineMap([], -self.Mean)
    #     return image

    # def reach_zono_multipleInputs(self, in_images, option):
    #
    #     n = len(in_images)
    #     images = []
    #     for i in range(n):
    #         images.append(self.reach_zono(in_images[i]))
    #     return images

    def reach(self, *args):

        nargs = len(args)
        if nargs == 2:
            in_images = args[0]
            method = args[1]
            option = 'single'

        elif nargs == 3:
            in_images = args[0]
            method = args[1]
            option = args[2]

        elif nargs == 4:
            in_images = args[0]
            method = args[1]
            option = args[2]
            relaxFactor = args[3]

        elif nargs == 5:
            in_images = args[0]
            method = args[1]
            option = args[2]
            relaxFactor = args[3]
            dis_opt = args[4]

        elif nargs == 6:
            in_images = args[0]
            method = args[1]
            option = args[2]
            relaxFactor = args[3]
            dis_opt = args[4]
            lp_solver = args[5]

        if method == 'approx-star' | method == 'exact-star' | method == 'abs-dom' | method == 'relax-star':
            images = self.reach_star_multipleInputs(in_images, option)
        else:
            Exception("unknown reachability method")

        return images

    def parse(self,input_image_layer):

        if input_image_layer != 'layer.ImageInputLayer':
            Exception("Input is not a ImageInputLayer class")

        L = ImageInputLayer('Name', input_image_layer.Name, 'InputSize', input_image_layer, 'Mean', input_image_layer.Mean)

        return L

class MaxPooling2d(object):

    def __init__(self, poolsize, stride, paddingsize, name='MaxPooling2d'):
        # name = 'MaxPooling2d'
        hasunpoolingoutputs = 0
        numinputs = 1
        inputnames = 'input'
        numoutputs = 1
        outputnames = 'output'

        if isinstance(name, str):
            self.Name = name
        else:
            Exception("Name is not string")
        if poolsize.shape != (2,):
            Exception("Invalid pool size")
        else:
            self.PoolSize = poolsize
        if stride.shape == (2,):
            self.Stride = stride
        else:
            Exception("Invalid stride size")
        if paddingsize.shape == (2,):
            self.PaddingSize = paddingsize
        else:
            Exception("Invalid padding size")
        if hasunpoolingoutputs >= 0:
            self.HasUnpoolingOutputs = hasunpoolingoutputs
        else:
            Exception("Invalid HasUnpoolingOutputs parameter")
        if numinputs >= 1:
            self.NumInputs = numinputs
        else:
            Exception("Invalid number of inputs")
        if numoutputs >= 1:
            self.NumOutputs = numoutputs
        else:
            Exception("Invalid number of outputs")

        self.InputNames = inputnames
        self.OutputNames = outputnames

        # elif nargs == 4:
        #     name = args[0]
        #     poolsize = args[1]
        #     stride = args[2]
        #     paddingsize = args[3]
        #     if isinstance(name, str):
        #         self.Name = name
        #     else:
        #         Exception("Name is not string")
        #     if poolsize.shape[0] != 1 | poolsize.shape[1] != 2:
        #         Exception("Invalid pool size")
        #     else:
        #         self.PoolSize = poolsize
        #     if stride.shape == (1,2):
        #         self.Stride = stride
        #     else:
        #         Exception("Invalid stride size")
        #     if paddingsize.shape == (1,4):
        #         self.PaddingSize = paddingsize
        #     else:
        #         Exception("Invalid padding size")
        #
        # elif nargs == 3:
        #     poolsize = args[1]
        #     stride = args[2]
        #     paddingsize = args[3]
        #
        #     if poolsize.shape[0] != 1 | poolsize.shape[1] != 2:
        #         Exception("Invalid pool size")
        #     else:
        #         self.PoolSize = poolsize
        #     if stride.shape == (1, 2):
        #         self.Stride = stride
        #     else:
        #         Exception("Invalid stride size")
        #     if paddingsize.shape == (1, 4):
        #         self.PaddingSize = paddingsize
        #     else:
        #         Exception("Invalid padding size")
        #
        # elif nargs == 0:
        #     self.Name = "max_pooling_2d_layer"
        #     self.PoolSize = [2,2]
        #     self.Stride = [1,1]
        #     self.PaddingMode = 'manual'
        #     self.PaddingSize = [0,0,0,0]
        #
        # else:
        #     Exception("Invalid number of inputs, shoudld be 0/3/4/9")

    def evaluate(self, *args):
        nargs = len(args)
        if nargs == 1:
            input = args[0]
            option = 'cnn'
        elif nargs == 2:
            input = args[0]
            option = args[1]
        else:
            Exception("Invalid number of input")

        y = nn.MaxPool2d(self.PoolSize, self.Stride, self.PaddingSize)
        return y

    def get_zero_padding_input(self, input):
        n = input.shape
        t = self.PaddingSize[0]
        b = self.PaddingSize[0]
        l = self.PaddingSize[1]
        r = self.PaddingSize[1]

        if len(n) == 2:
            # Input has one channel
            h = n[0]
            w = n[1]

            padded_I = np.zeros((t+h+b, l+w+r))
            padded_I[t:t+h, l:l+w] = input

        elif len(n) > 2:
            h = n[0]
            w = n[1]
            d = n[2]

            padded_I = np.zeros((t+h+b, l+w+r, d))
            for i in range(d):
                padded_I[t:t+h, l:l+w, i] = input[:,:,i]

        return padded_I

    def get_zero_padding_imagestar(self, ims):

        if self.PaddingSize.sum() == 0:
            # pad_ims = ims
            if hasattr(ims, 'im_lb'):
                pad_ims = imageStar.ImageStar(ims.V.copy(), ims.C.copy(), ims.d.copy(), ims.pred_lb.copy(),
                                          ims.pred_ub.copy(), ims.im_lb.copy(), ims.im_ub.copy())
            else:
                pad_ims = imageStar.ImageStar(ims.V.copy(), ims.C.copy(), ims.d.copy(), ims.pred_lb.copy(),
                                              ims.pred_ub.copy())
        else:
            c = self.get_zero_padding_input(ims.V[:,:,:,1])
            k = c.shape
            n = ims.numPred
            new_size = (k[0], k[1], k[2], n+1)
            V1 = np.zeros(new_size, dtype=np.single)
            V1[:,:,:,0] = c
            for i in range(n):
                V1[:,:,:,i+1] = self.get_zero_padding_input(ims.V[:,:,:,i+1])
            if hasattr(ims, 'im_lb'):
                new_im_lb = self.get_zero_padding_input(ims.im_lb)
                new_im_ub = self.get_zero_padding_input(ims.im_ub)
            else:
                new_im_lb = None
                new_im_ub = None
            pad_ims = imageStar.ImageStar(V1, ims.C, ims.d, ims.pred_lb, ims.pred_ub, new_im_lb, new_im_ub)

        return pad_ims

    def compute_maxmap(self, input):
        I = self.get_zero_padding_input(input)
        m = self.PoolSize
        h, w = self.get_size_maxmap(input)
        map = self.get_startpoints(input)
        maxmap = np.zeros((1,h*w))

        for l in range(h*w):
            a  = l%w
            if a == 0:
                i = math.floor(l/w)
                j = w
            else:
                i = a
                j = math.floor(l/w) + 1

            i0 = map[i-1, j-1][0]
            j0 = map[i-1, j-1][1]
            val = I[i0-1, j0-1]
            for i in range(i0,i0+m[0]):
                for j in range(j0, j0+m[1]):
                    if val < I[i,j]:
                        val = I[i,j]
            maxmap[l] = val
        maxmap = maxmap.reshape([h, w]).T
        return maxmap

    def get_size_maxmap(self, input):
        I = self.get_zero_padding_input(input)
        n = I.shape
        m = self.PoolSize
        h = math.floor((n[0]-m[0])/self.Stride[0]+1)
        w = math.floor((n[1]-m[1])/self.Stride[1]+1)

        return h, w

    def get_startpoints(self, input):
        I = self.get_zero_padding_input(input)
        m = self.PoolSize

        h, w = self.get_size_maxmap(input)

        startPoints = [[] for i in range(h)]

        for i in range(h):
            for j in range(w):
                startPoints[i].append(np.zeros(2))
                if i == 0:
                    startPoints[i][j][0] = 0
                elif i >= 1:
                    startPoints[i][j][0] = startPoints[i-1][j][0] + self.Stride[0]
                if j == 0:
                    startPoints[i][j][1] = 0
                elif j >= 1:
                    startPoints[i][j][1] = startPoints[i][j-1][1] + self.Stride[1]
        return startPoints

    def construct_maxmap(self, input):
        if isinstance(input, imageStar.ImageStar) is False:
            Exception("Input is not an ImageStar")

        h, w = self.get_size_maxmap(input.V[:,:,0,0])
        new_V = np.zeros((h,w,input.numChannel, input.numPred+1))

        channel_maxPoints = np.zeros((3, h*w, input.numChannel))
        for i in range(input.numChannel):
            channel_maxPoints[:,:,i] = input.max_points[:,i*h*w:(i+1)*h*w]

        for p in range(input.numPred+1):
            for k in range(input.numChannel):
                for i in range(h):
                    for j in range(w):
                        ind = i*w + j
                        max_ind = channel_maxPoints[:, ind, k]
                        new_V[:,:,k,p] = input.V[max_ind[0], max_ind[1], k, p]

        image = imageStar.ImageStar(new_V, input.C, input.d, input.pred_lb, input.pred_ub)
        return image

    def stepSplit(self, *args):
        nargs = len(args)
        if nargs == 4:
            in_image = args[0]
            ori_image = args[1]
            pos =args [2]
            split_index = args[3]
            lp_solver = 'linprog'
        elif nargs == 5:
            in_image = args[0]
            ori_image = args[1]
            pos = args[2]
            split_index = args[3]
            lp_solver = args[4]

        if isinstance(in_image, imageStar.ImageStar) is False:
            raise Exception('input maxMap is not an ImageStar')
        if isinstance(ori_image, imageStar.ImageStar) is False:
            raise  Exception('reference image is not an ImageStar')

        n = split_index.shape
        if (n[1] != 3) | (n[0] < 1):
            raise Exception('Invalid split index, it should have 3 columns and at least 1 row')

        images = []
        for i in range(n[0]):
            center = split_index[i].copy()
            others = split_index.copy()
            others = np.delete(others, i, axis=0)
            # others[i] = np.zeros(n[1])
            new_C, new_d = imageStar.ImageStar.isMax(in_image, ori_image, center, others, lp_solver)
            if (new_C is not None) & (new_d is not None):
                V = in_image.V
                V[pos[0], pos[1], pos[2], :] = ori_image.V[center[0], center[1], center[2], :]
                if hasattr(in_image, 'im_lb'):
                    im = imageStar.ImageStar(V, new_C, new_d, in_image.pred_lb, in_image.pred_ub, in_image.im_lb, in_image.im_ub)
                else:
                    im = imageStar.ImageStar(V, new_C, new_d, in_image.pred_lb, in_image.pred_ub)
                im.MaxIdxs = in_image.MaxIdxs.copy()
                im.InputSizes = in_image.InputSizes.copy()
                im.updateMaxIdx(self.Name, center, pos)
                images.append(im)
        return images


    def stepSplitMultipleInputs(self, *args):
        nargs = len(args)
        if nargs == 5:
            in_images = args[0]
            ori_image = args[1]
            pos = args[2].copy()
            split_index = args[3].copy()
            option = args[4].copy()
            lp_solver = 'linprog'
        elif nargs == 6:
            in_images = args[0]
            ori_image = args[1]
            pos = args[2].copy()
            split_index = args[3].copy()
            option = args[4].copy()
            lp_solver = args[5]
        else:
            raise Exception('Invalid number of input arguments!')

        images = []
        if isinstance(in_images, list):
            n = len(in_images)
            for i in range(n):
                images.extend(self.stepSplit(in_images[i], ori_image, pos, split_index, lp_solver))
        elif isinstance(in_images, imageStar.ImageStar):
            images.extend(self.stepSplit(in_images, ori_image, pos, split_index, lp_solver))

        return images


    def reach_star_approx(self, *args):
        nargs = len(args)
        if nargs == 1:
            in_image = args[0]
            dis_opt = []
            lp_solver = 'linprog'
        elif nargs == 2:
            in_image = args[0]
            dis_opt = args[1]
            lp_solver = 'linprog'
        elif nargs == 3:
            in_image = args[0]
            dis_opt = args[1]
            lp_solver = args[2]
        else:
            raise Exception("Invalid number of input arguments, should be 1/2/3")

        if isinstance(in_image, imageStar.ImageStar) is False:
            raise Exception("Input image is not an ImageStar")

        h, w = self.get_size_maxmap(in_image.V[:,:,0,0])
        startPoints = self.get_startpoints(in_image.V[:,:,0,0])
        max_index = [[[] for i in range(w)] for j in range(h)]

        pad_image = self.get_zero_padding_imagestar(in_image)

        np1 = pad_image.numPred
        l = 0
        for k in range(pad_image.numChannel):
            for i in range(h):
                for j in range(w):
                    max_index[i][j].append(pad_image.get_localMax_index(startPoints[i][j], self.PoolSize, k, lp_solver))
                    max_id = max_index[i][j][k]
                    if (len(max_id.shape) > 1) & (max_id.shape[0] > 1):
                        np1 += 1
                        l += 1

        new_V = np.zeros((h, w, pad_image.numChannel, pad_image.numPred + 1))
        new_pred_index = 0
        for k in range(pad_image.numChannel):
            for i in range(h):
                for j in range(w):
                    max_id = max_index[i][j][k]
                    if (len(max_id.shape) == 1) | ((len(max_id.shape) > 1) & (max_id.shape[0] == 1)):
                        for p in range(pad_image.numPred+1):
                            new_V[i, j, k, p] = pad_image.V[max_id[0], max_id[1],k, p]
                    else:
                        new_V[i,j,k,0] = 0
                        new_pred_index += 1
                        empty_V = np.zeros((h, w, pad_image.numChannel, 1))
                        new_V = np.concatenate((new_V, empty_V), axis=3)
                        # new_V.resize((h,w,pad_image.numChannel,pad_image.numPred+1+new_pred_index),refcheck=False)
                        new_V[i,j,k,pad_image.numPred+new_pred_index] = 1

        N = self.PoolSize[0] * self.PoolSize[1]
        new_C = np.zeros((new_pred_index*(N+1), np1))
        new_d = np.zeros((new_pred_index*(N+1), 1))
        new_pred_lb = np.zeros((new_pred_index, 1))
        new_pred_ub = np.zeros((new_pred_index, 1))
        new_pred_index = 0
        for k in range(pad_image.numChannel):
            for i in range(h):
                for j in range(w):
                    max_id = max_index[i][j][k]
                    if (len(max_id.shape) > 1) & (max_id.shape[0] > 1):
                        new_pred_index += 1
                        startpoint = startPoints[i][j]
                        points = pad_image.get_localPoints(startpoint, self.PoolSize)
                        C1 = np.zeros((1,np1))
                        C1[0, pad_image.numPred+new_pred_index-1] = 1
                        lb, ub = pad_image.get_localBound(startpoint, self.PoolSize, k, lp_solver)
                        new_pred_lb[new_pred_index-1, 0] = lb
                        new_pred_ub[new_pred_index-1, 0] = ub
                        d1 = ub
                        C2 = np.zeros((N, np1))
                        d2 = np.zeros((N, 1))
                        for g in range(N):
                            point = points[g, :]
                            C2[g, 1:pad_image.numPred] = pad_image.V[point[0], point[1], k, 1:pad_image.numPred]
                            C2[g, pad_image.numPred + new_pred_index - 1] = -1
                            d2[g] = -pad_image.V[point[0],point[1], k, 0]

                        C = np.concatenate((C1, C2))
                        d = np.concatenate((np.array([[d1]]), d2))

                        new_C[(new_pred_index-1)*(N+1):new_pred_index*(N+1), :] = C
                        new_d[(new_pred_index-1)*(N+1):new_pred_index*(N+1)] = d

        n = pad_image.C.shape[0]
        C = np.concatenate((pad_image.C, np.zeros((n, new_pred_index))),axis=1)
        new_C = np.concatenate((C, new_C))
        new_d = np.concatenate((pad_image.d, new_d))
        new_pred_lb = np.concatenate((pad_image.pred_lb, new_pred_lb))
        new_pred_ub = np.concatenate((pad_image.pred_ub, new_pred_ub))

        image = imageStar.ImageStar(new_V, new_C, new_d, new_pred_lb, new_pred_ub)
        # image.addInputSize(self.Name, [pad_image.height, pad_image.width])
        # image.addMaxIdx(self.Name, max_index)
        return image

    def reach_star_approx_multipleinputs(self, *args):
        nargs = len(args)
        if nargs == 2:
            in_images = args[0]
            option = args[1]
            dis_opt = []
            lp_solver = 'linprog'
        elif nargs == 3:
            in_images = args[0]
            option = args[1]
            dis_opt = args[2]
            lp_solver = 'linprog'
        elif nargs == 4:
            in_images = args[0]
            option = args[1]
            dis_opt = args[2]
            lp_solver = args[3]
        else:
            Exception("Invalid number of input arguments")

        n = len(in_images)
        IS = []

        for i in range(n):
            IS.append(self.reach_star_approx(in_images[i], dis_opt, lp_solver))

        return IS

    def reach_star_exact(self, *args):
        nargs = len(args)
        if nargs == 2:
            in_image = args[0]
            option = args[1]
            dis_opt = None
            lp_solver = 'linprog'
        elif nargs == 3:
            in_image = args[0]
            option = args[1]
            dis_opt = args[2]
            lp_solver = 'linprog'
        elif nargs == 4:
            in_image = args[0]
            option = args[1]
            dis_opt = args[2]
            lp_solver = args[3]
        else:
            raise Exception('Invalid number of input arguments')

        if isinstance(in_image, imageStar.ImageStar) is False:
            raise Exception("Input image is not an ImageStar")

        startPoints = self.get_startpoints(in_image.V[:, :, 0, 0])
        h, w = self.get_size_maxmap(in_image.V[:, :, 0, 0])
        pad_image = self.get_zero_padding_imagestar(in_image)

        # check max_id first
        # max_index = [[[] for i in range(w)] for j in range(h)]
        max_index = np.zeros((h, w, pad_image.numChannel), dtype=object)
        maxMap_basis_V = np.zeros((h, w, pad_image.numChannel, pad_image.numPred + 1), dtype=np.single)
        split_pos = []

        # maxidx = [[[] for i in range(w)] for j in range(h)]
        maxidx = np.zeros((h, w, pad_image.numChannel), dtype=object)
        for k in range(pad_image.numChannel):
            for i in range(h):
                for j in range(w):
                    # max_index[i][j].append(pad_image.get_localMax_index(startPoints[i][j], self.PoolSize, k, lp_solver))
                    max_index[i, j, k] = pad_image.get_localMax_index(startPoints[i][j], self.PoolSize, k, lp_solver)
                    # construct the basis image for the maxMap
                    if len(max_index[i][j][k].shape) == 1:
                        maxMap_basis_V[i, j, k, :] = pad_image.V[max_index[i][j][k][0], max_index[i][j][k][1], k, :].copy()
                        # maxidx[i][j].append(max_index[i][j][k])
                        maxidx[i, j, k] = max_index[i, j, k]
                    else:
                        split_pos.append([i, j, k])

        n = len(split_pos)
        images = imageStar.ImageStar(maxMap_basis_V, pad_image.C, pad_image.d, pad_image.pred_lb, pad_image.pred_ub)
        images.inheritMaxIdx(in_image.MaxIdxs)
        images.inheritInputSize(in_image.InputSizes)
        images.addMaxIdx(self.Name, maxidx)
        images.addInputSize(self.Name, [pad_image.height, pad_image.width])
        if n > 0:
            for i in range(n):
                if isinstance(images, list):
                    m1 = len(images)
                else:
                    m1 = 1
                # print(split_pos)
                images = self.stepSplitMultipleInputs(images, pad_image, split_pos[i],
                                                      max_index[split_pos[i][0]][split_pos[i][1]][split_pos[i][2]], [])
                if isinstance(images, list):
                    m2 = len(images)
                else:
                    m2 = 1
                print('\nSplit %d images into %d images' % (m1, m2))

        return images

    def reach_star_exact_multipleInputs(self, *args):
        nargs = len(args)
        if nargs == 2:
            in_images = args[0]
            option = args[1]
            dis_opt = None
            lp_solver = 'linprog'
        elif nargs == 3:
            in_images = args[0]
            option = args[1]
            dis_opt = args[2]
            lp_solver = 'linprog'
        elif nargs == 4:
            in_images = args[0]
            option = args[1]
            dis_opt = args[2]
            lp_solver = args[3]
        else:
            raise Exception('Invalid number of input arguments')

        IS = []
        if isinstance(in_images, list):
            n = len(in_images)
            for i in range(n):
                temp1 = self.reach_star_exact(in_images[i], dis_opt, lp_solver)
                if isinstance(temp1, list):
                    IS.extend(temp1)
                else:
                    IS.append(temp1)
        elif isinstance(in_images, imageStar.ImageStar):
            temp1 = self.reach_star_exact(in_images, dis_opt, lp_solver)
            if isinstance(temp1, list):
                IS.extend(temp1)
            else:
                IS.append(temp1)

        return IS

    def reach(self, im, me):
        in_images = im
        method = me
        option = []
        relaxFactor = []
        dis_opt = []
        lp_solver = 'linprog'
        # elif nargs == 3:
        #     in_images = args[0]
        #     method = args[1]
        #     option = args[2]
        #     relaxFactor = []
        #     dis_opt = []
        #     lp_solver = 'linprog'
        # elif nargs == 4:
        #     in_images = args[0]
        #     method = args[1]
        #     option = args[2]
        #     relaxFactor = args[3]
        #     dis_opt = []
        #     lp_solver = 'linprog'
        # elif nargs == 5:
        #     in_images = args[0]
        #     method = args[1]
        #     option = args[2]
        #     relaxFactor = args[3]
        #     dis_opt = args[4]
        #     lp_solver = 'linprog'
        # elif nargs == 6:
        #     in_images = args[0]
        #     method = args[1]
        #     option = args[2]
        #     relaxFactor = args[3]
        #     dis_opt = args[4]
        #     lp_solver = args[5]
        # else:
        #     Exception("Invalid number of input arguments, should be 2/3/4/5/6")

        if method == 'approx-star':
            # IS = self.reach_star_approx_multipleinputs(in_images, option, dis_opt, lp_solver)
            IS = self.reach_star_approx(in_images, dis_opt, lp_solver)
        elif method == 'abs-dom':
            IS = self.reach_star_approx_multipleinputs(in_images, option, dis_opt, lp_solver)
        elif method == 'exact-star':
            IS = self.reach_star_exact_multipleInputs(in_images, option, dis_opt, lp_solver)

        return IS

    def parse(self, input):

        if isinstance(input, layer.MaxPooling2d) is False:
            Exception("Input is not a MaxPooling2d")

        L = MaxPooling2d(input.Name, input.PoolSize, input.Stride, input.PaddingSize, input.HasUnpoolingOutputs,
                              input.NumInputs, input.InputNames, input.NumOutputs, input.OutputNames)
        return L

class FC(object):

    def __init__(self, weight1, bias1, weight2, bias2, option=None):
        if option is None:
            if weight1.shape[0] != bias1.shape[0]:
                raise Exception("Inconsistent dimension between the weight1 matrix and bias1 vector")
            if weight2.shape[0] != bias2.shape[0]:
                raise Exception("Inconsistent dimension between the weight2 matrix and bias2 vector")
            self.Weights1 = weight1
            self.Bias1 = bias1
            self.Weights2 = weight2
            self.Bias2 = bias2
            self.option = option
            self.InputSize = weight1.shape[1]
            self.OutputSize = weight1.shape[0]
        elif option == 'last':
            self.Weights1 = weight1
            self.Bias1 = bias1
            self.Weights2 = weight2
            self.Bias2 = bias2
            self.option = option
            self.InputSize = weight1.shape[1] * 2
            self.OutputSize = weight1.shape[0]
        # elif isinstance(weight1, torch.Tensor):
        #     if self.option is None:
        #         self.InputSize = weight1.shape[1]*2
        #         self.OutputSize = weight1.shape[0]*2
        #     elif self.option == 'last':
        #         self.InputSize = weight1.shape[1] * 2
        #         self.OutputSize = weight1.shape[0]

        # nargs = len(args)
        # if nargs == 0:
        #     self.Name = "FC"
        #     self.InputSize = 0
        #     self.OutputSize = 0
        #     self.Weights = []
        #     self.Bias = []
        # elif nargs == 2:
        #     W = args[0]
        #     b = args[1]
        #     self.Name = "FC"
        #
        #     if W.shape[0] != b.shape[0]:
        #         Exception("Inconsistent dimension between the weight matrix and bias vector")
        #     if b.shape[1] != 1:
        #         Exception("Bias vector should have one column")
        #
        #     self.InputSize = W.shape[1]
        #     self.OutputSize = W.shape[0]
        #     self.Weights = W
        #     self.Bias = b
        # elif nargs == 3:
        #     name = args[0]
        #     W = args[1]
        #     b = args[2]
        #
        #     if W.shape[0] != b.shape[0]:
        #         Exception("Inconsistent dimension between the weight matrix and bias vector")
        #     if b.shape[1] != 1:
        #         Exception("Bias vector should have one column")
        #
        #     self.Name = name
        #     self.InputSize = W.shape[1]
        #     self.OutputSize = W.shape[0]
        #     self.Weights = W
        #     self.Bias = b
        # else:
        #     Exception("Invalid number of input, should be 0/2/3")

    def merge_reach1(self, image1, image2):
        if isinstance(image1, imageStar.ImageStar) is False:
            raise Exception("image1 is not an ImageStar")
        if isinstance(image2, imageStar.ImageStar) is False:
            raise Exception("image2 is not an ImageStar")

        numPred = image1.numPred
        N = image1.height*image1.width*image1.numChannel
        if N != self.InputSize:
            raise Exception("Inconsistency between the size of the input image and the InputSize of the network")
        V1 = np.zeros((1,1,self.OutputSize, numPred+1))
        V2 = np.zeros((1,1,self.OutputSize, numPred+1))

        fc1 = nn.Linear(self.InputSize, self.OutputSize)
        fc1.weight = nn.Parameter(torch.from_numpy(self.Weights1),requires_grad=False)
        fc1.bias = nn.Parameter(torch.from_numpy(self.Bias1),requires_grad=False)

        fc2 = nn.Linear(self.InputSize, self.OutputSize)
        fc2.weight = nn.Parameter(torch.from_numpy(self.Weights2), requires_grad=False)
        fc2.bias = nn.Parameter(torch.from_numpy(self.Bias2), requires_grad=False)

        fc1_v = nn.Linear(self.InputSize, self.OutputSize, bias=False)
        fc1_v.weight = nn.Parameter(torch.from_numpy(self.Weights1), requires_grad=False)
        fc2_v = nn.Linear(self.InputSize, self.OutputSize, bias=False)
        fc2_v.weight = nn.Parameter(torch.from_numpy(self.Weights2), requires_grad=False)

        if isinstance(self.Weights1, np.ndarray):
            fc1.weight = nn.Parameter(torch.from_numpy(self.Weights1), requires_grad=False)
            fc1.bias = nn.Parameter(torch.from_numpy(self.Bias1), requires_grad=False)
            fc2.weight = nn.Parameter(torch.from_numpy(self.Weights2), requires_grad=False)
            fc2.bias = nn.Parameter(torch.from_numpy(self.Bias2), requires_grad=False)
            fc1_v.weight = nn.Parameter(torch.from_numpy(self.Weights1), requires_grad=False)
            fc2_v.weight = nn.Parameter(torch.from_numpy(self.Weights2), requires_grad=False)
        elif isinstance(self.Weights1, torch.Tensor):
            fc1.weight = nn.Parameter(self.Weights1, requires_grad=False)
            fc1.bias = nn.Parameter(self.Bias1, requires_grad=False)
            fc2.weight = nn.Parameter(self.Weights2, requires_grad=False)
            fc2.bias = nn.Parameter(self.Bias2, requires_grad=False)
            fc1_v.weight = nn.Parameter(self.Weights1, requires_grad=False)
            fc2_v.weight = nn.Parameter(self.Weights2, requires_grad=False)

        with torch.no_grad():
            for i in range(numPred):
                I1 = torch.from_numpy(image1.V[:,:,:,i].flatten()).unsqueeze_(0)
                I2 = torch.from_numpy(image2.V[:,:,:,i].flatten()).unsqueeze_(0)
                if i == 0:
                    V1[0,0,:,i] = fc1(I1)
                    V2[0,0,:,i] = fc2(I2)
                else:
                    V1[0,0,:,i] = fc1_v(I1)
                    V2[0,0,:,i] = fc2_v(I2)

        Istar1 = imageStar.ImageStar(V1, image1.C, image1.d, image1.pred_lb, image1.pred_ub)
        Istar2 = imageStar.ImageStar(V2, image2.C, image2.d, image2.pred_lb, image2.pred_ub)

        return Istar1, Istar2

    def merged_reach(self, image):
        if isinstance(image, imageStar.ImageStar) is False:
            raise Exception("image is not an ImageStar")

        numPred = image.numPred
        N = image.height * image.width * image.numChannel
        if (N/2 != self.InputSize) & (self.option != 'last'):
            raise Exception("Inconsistency between the size of the input image and the InputSize of the network")
        elif (N != self.InputSize) & (self.option == 'last'):
            raise Exception("Inconsistency between the size of the input image and the InputSize of the network")

        temp_V = image.V.copy()
        temp_V = temp_V.transpose(2, 3, 0, 1)

        if self.option == 'last':
            V = np.zeros((1, 1, self.OutputSize, numPred + 1), dtype=np.single)
            merge_w = np.concatenate((self.Weights1, self.Weights2), axis=1)

            fc = nn.Linear(self.InputSize, self.OutputSize, bias=False)
            fc.weight = nn.Parameter(torch.from_numpy(merge_w), requires_grad=False)

            with torch.no_grad():
                for i in range(numPred+1):
                    I = torch.from_numpy(temp_V[:, i, :, :].flatten()).unsqueeze_(0)
                    V[0, 0, :, i] = fc(I)

            Istar = imageStar.ImageStar(V, image.C, image.d, image.pred_lb, image.pred_ub)
            return Istar
        else:
            # V = np.zeros((1, 1, self.OutputSize * 2, numPred + 1))
            # w1_m = np.concatenate((self.Weights1, np.zeros(self.Weights1.shape)))
            # w2_m = np.concatenate((np.zeros(self.Weights2.shape), self.Weights2))
            # merge_w = np.concatenate((w1_m, w2_m), axis=1)
            # merge_b = np.concatenate((self.Bias1, self.Bias2))
            #
            # fc = nn.Linear(self.InputSize, self.OutputSize)
            # fc.weight = nn.Parameter(torch.from_numpy(merge_w), requires_grad=False)
            # fc.bias = nn.Parameter(torch.from_numpy(merge_b), requires_grad=False)
            #
            # fc_v = nn.Linear(self.InputSize, self.OutputSize, bias=False)
            # fc_v.weight = nn.Parameter(torch.from_numpy(merge_w), requires_grad=False)
            #
            # with torch.no_grad():
            #     for i in range(numPred):
            #         I = torch.from_numpy(temp_V[:, i, :, :].flatten()).unsqueeze_(0)
            #         if (temp_V.shape[3] != 1)&(i==0):
            #             print(I[0:400].to(torch.float32))
            #         if i == 0:
            #             V[0, 0, :, i] = fc(I)
            #         else:
            #             V[0, 0, :, i] = fc_v(I)

            V = np.zeros((1, 1, self.OutputSize * 2, numPred + 1), dtype=np.single)
            if isinstance(self.Weights1, torch.Tensor):
                fc1 = nn.Linear(self.InputSize, self.OutputSize)
                fc1.weight = nn.Parameter(self.Weights1, requires_grad=False)
                fc1.bias = nn.Parameter(self.Bias1, requires_grad=False)
                fc2 = nn.Linear(self.InputSize, self.OutputSize)
                fc2.weight = nn.Parameter(self.Weights2, requires_grad=False)
                fc2.bias = nn.Parameter(self.Bias2, requires_grad=False)

                fc_v1 = nn.Linear(self.InputSize, self.OutputSize, bias=False)
                fc_v1.weight = nn.Parameter(self.Weights1, requires_grad=False)
                fc_v2 = nn.Linear(self.InputSize, self.OutputSize, bias=False)
                fc_v2.weight = nn.Parameter(self.Weights2, requires_grad=False)
            else:
                fc1 = nn.Linear(self.InputSize, self.OutputSize)
                fc1.weight = nn.Parameter(torch.from_numpy(self.Weights1), requires_grad=False)
                fc1.bias = nn.Parameter(torch.from_numpy(self.Bias1), requires_grad=False)
                fc2 = nn.Linear(self.InputSize, self.OutputSize)
                fc2.weight = nn.Parameter(torch.from_numpy(self.Weights2), requires_grad=False)
                fc2.bias = nn.Parameter(torch.from_numpy(self.Bias2), requires_grad=False)

                fc_v1 = nn.Linear(self.InputSize, self.OutputSize, bias=False)
                fc_v1.weight = nn.Parameter(torch.from_numpy(self.Weights1), requires_grad=False)
                fc_v2 = nn.Linear(self.InputSize, self.OutputSize, bias= False)
                fc_v2.weight = nn.Parameter(torch.from_numpy(self.Weights2), requires_grad=False)

            with torch.no_grad():
                if temp_V.shape[3] != 1:
                    h = temp_V.shape[3]
                    for i in range(numPred+1):
                        # I1 = torch.from_numpy(image.V[0:h, :, : , i].flatten()).unsqueeze_(0)
                        # I2 = torch.from_numpy(image.V[h:, :, :, i].flatten()).unsqueeze_(0)
                        I1 = torch.flatten(torch.from_numpy(temp_V[:, i, 0:h, :])).unsqueeze_(0)
                        I2 = torch.flatten(torch.from_numpy(temp_V[:, i, h:, :])).unsqueeze_(0)
                        if i == 0:
                            # print(image.V[:,:,5,0])
                            # print(I1)
                            V[0, 0, 0:self.OutputSize, i] = fc1(I1)
                            V[0,0,self.OutputSize:,i] = fc2(I2)
                        else:
                            V[0, 0, 0:self.OutputSize, i] = fc_v1(I1)
                            V[0, 0, self.OutputSize:, i] = fc_v2(I2)
                elif (temp_V.shape[3] == 1) & (temp_V.shape[2] == 2):
                    h = temp_V.shape[3]
                    for i in range(numPred+1):
                        # I1 = torch.from_numpy(image.V[0:h, :, : , i].flatten()).unsqueeze_(0)
                        # I2 = torch.from_numpy(image.V[h:, :, :, i].flatten()).unsqueeze_(0)
                        I1 = torch.flatten(torch.from_numpy(temp_V[:, i, 0:h, :])).unsqueeze_(0)
                        I2 = torch.flatten(torch.from_numpy(temp_V[:, i, h:, :])).unsqueeze_(0)
                        if i == 0:
                            # print(image.V[:,:,5,0])
                            # print(I1)
                            V[0, 0, 0:self.OutputSize, i] = fc1(I1)
                            V[0, 0, self.OutputSize:, i] = fc2(I2)
                        else:
                            V[0, 0, 0:self.OutputSize, i] = fc_v1(I1)
                            V[0, 0, self.OutputSize:, i] = fc_v2(I2)
                else:
                    c = int(temp_V.shape[0]/2)
                    for i in range(numPred+1):
                        I1 = torch.from_numpy(temp_V[0:c, i, :, :].flatten()).unsqueeze_(0)
                        I2 = torch.from_numpy(temp_V[c:, i, :, :].flatten()).unsqueeze_(0)
                        if i == 0:
                            V[0, 0, 0:self.OutputSize, i] = fc1(I1)
                            V[0,0,self.OutputSize:,i] = fc2(I2)
                        else:
                            V[0, 0, 0:self.OutputSize, i] = fc_v1(I1)
                            V[0, 0, self.OutputSize:, i] = fc_v2(I2)

            Istar = imageStar.ImageStar(V, image.C, image.d, image.pred_lb, image.pred_ub)
            Istar.inheritMaxIdx(image.MaxIdxs)
            Istar.inheritInputSize(image.InputSizes)
            return Istar

    def multi_merged_reach(self, in_images):
        IS = []
        if isinstance(in_images, list):
            n = len(in_images)
            for i in range(n):
                IS.append(self.merged_reach(in_images[i]))
        elif isinstance(in_images, imageStar.ImageStar):
            IS.append(self.merged_reach(in_images))

        return IS

    def reach(self, images):
        IS = self.multi_merged_reach(images)

        return IS

    # def evaluate(self, input):
    #     n = input.shape
    #     N = 1
    #     for i in range(len(n)):
    #         N = N*n[i]
    #
    #     if N != self.InputSize:
    #         Exception("Inconsistency between the input dimension and InputSize of the network")
    #
    #     I = input.reshape((N,1))
    #     y = self.Weights*I + self.Bias
    #     y = y.reshape((1,1,self.Bias.shape[0]))
    #
    #     return y
    #
    #
    # def reach_star_single_input(self, in_image):
    #
    #     if isinstance(in_image, imageStar.ImageStar) is False:
    #         Exception("Input set is not an ImageStar")
    #
    #     N = in_image.height*in_image.width*in_image.numChannel
    #     if N != self.InputSize:
    #         Exception("Inconsistency between the size of the input image and the InputSize of the network")
    #
    #     n = in_image.numPred
    #     V = np.zeros((1,1,self.OutputSize,in_image.numPred+1))
    #     for i in range(n):
    #         I = in_image.V[:,:,:,i].faltten()
    #         if i == 1:
    #             V[0,0,:,i] = self.Weights*I.T + self.Bias
    #         else:
    #             V[0,0,:,i] = self.Weights*I.T
    #
    #     image = imageStar.ImageStar(V, in_image.C, in_image.d, in_image.pred_lb, in_image.pred_ub)
    #     return image
    #
    #
    # def reach_star_multipleInputs(self, inputs, option):
    #     n = len(inputs)
    #     S = []
    #     for i in range(n):
    #         S.append(self.reach_star_single_input(inputs[i]))
    #
    #     return S
    #
    #
    # def reach(self, *args):
    #     nargs = len(args)
    #     if nargs == 6:
    #         in_images = args[0]
    #         method = args[1]
    #         option = args[2]
    #     elif nargs == 5:
    #         in_images = args[0]
    #         method = args[1]
    #         option = args[2]
    #     elif nargs == 4:
    #         in_images = args[0]
    #         method = args[1]
    #         option = args[2]
    #     elif nargs == 3:
    #         in_images = args[0]
    #         method = args[1]
    #         option = args[2]
    #     elif nargs == 2:
    #         in_images = args[0]
    #         method = args[1]
    #         option = []
    #     else:
    #         Exception("Invalid number of input, should be 2/3/4/5/6")
    #
    #     if method == 'approx-star':
    #         IS = self.reach_star_multipleInputs(in_images, option)
    #     else:
    #         Exception("Unknown reachability method")
    #
    #     return IS

    # def parse(self, fc_layer):
    #     if isinstance(fc_layer, layer.FC) is False:
    #         Exception("Input is not a layer.FC")
    #
    #     L = FC(fc_layer.Name, fc_layer.Weights, fc_layer.Bias)
    #     return L\

class MergedConv2d(object):

    def __init__(self, weight1, bias1, weight2, bias2, pad1=(0, 0), stride1=1, dila1=1, pad2=(0, 0), stride2=1, dila2=1):
        layer_name = 'Merged Conv2d'
        padding_mat1 = pad1
        stride_mat1 = stride1
        dilation_mat1 = dila1
        padding_mat2 = pad2
        stride_mat2 = stride2
        dilation_mat2 = dila2
        self.NumInputs = 1
        self.InputNames = "input"
        self.NumOutputs = 1
        self.OutputNames = "output"

        if isinstance(layer_name, str):
            self.Name = layer_name
        else:
            Exception("Layer name should be a string")

        if isinstance(weight1, np.ndarray):
            filter_weights1 = weight1.astype(np.float64)
            filter_bias1 = bias1.astype(np.float64)
            filter_weights2 = weight2.astype(np.float64)
            filter_bias2 = bias2.astype(np.float64)

            w1 = filter_weights1.shape
            b1 = filter_bias1.shape
            if len(w1) != 4:
                Exception("Invalid weights 1 array")
            if len(b1) != 3:
                Exception("Invalid bias 1 array")
            if w1[3] != b1[1]:
                Exception("Inconsistency between filter weights 1 and filter bias 1")

            self.NumFilters1 = w1[3]
            self.NumChannels1 = w1[2]
            self.FilterSize1 = np.array([w1[0], w1[1]])
            self.Weights1 = filter_weights1
            self.Bias1 = filter_bias1

            # p1 = padding_mat1.shape
            # if len(p1) !=2 | p1[1] != 4 | p1[0] != 1:
            #     Exception("Invalid padding matrix")
            self.PaddingSize1 = padding_mat1

            # s = stride_mat1.shape
            # if len(s) != 2 | s[0] != 1:
            #     Exception("Invalid stride matrix")
            self.Stride1 = stride_mat1

            # d = dilation_mat.shape
            # if len(d) != 2 | d[0] != 1:
            #     Exception("Invalid dilation matrix")
            self.DilationFactor1 = dilation_mat1

            w2 = filter_weights2.shape
            b2 = filter_bias2.shape
            if len(w2) != 4:
                Exception("Invalid weights 2 array")
            if len(b2) != 3:
                Exception("Invalid bias 2 array")
            if w2[3] != b2[1]:
                Exception("Inconsistency between filter weights 2 and filter bias 2")

            self.NumFilters2 = w2[3]
            self.NumChannels2 = w2[2]
            self.FilterSize2 = np.array([w2[0], w2[1]])
            self.Weights2 = filter_weights2
            self.Bias2 = filter_bias2
            self.PaddingSize2 = padding_mat2
            self.Stride2 = stride_mat2
            self.DilationFactor2 = dilation_mat2
        elif isinstance(weight1, torch.Tensor):
            filter_weights1 = weight1
            filter_bias1 = bias1
            filter_weights2 = weight2
            filter_bias2 = bias2

            w1 = filter_weights1.shape
            self.NumFilters1 = w1[0]
            self.NumChannels1 = w1[1]
            self.FilterSize1 = np.array([w1[2], w1[3]])
            self.Weights1 = filter_weights1
            self.Bias1 = filter_bias1

            # p1 = padding_mat1.shape
            # if len(p1) !=2 | p1[1] != 4 | p1[0] != 1:
            #     Exception("Invalid padding matrix")
            self.PaddingSize1 = padding_mat1

            # s = stride_mat1.shape
            # if len(s) != 2 | s[0] != 1:
            #     Exception("Invalid stride matrix")
            self.Stride1 = stride_mat1

            # d = dilation_mat.shape
            # if len(d) != 2 | d[0] != 1:
            #     Exception("Invalid dilation matrix")
            self.DilationFactor1 = dilation_mat1

            w2 = filter_weights2.shape
            self.NumFilters2 = w2[0]
            self.NumChannels2 = w2[1]
            self.FilterSize2 = np.array([w2[2], w2[3]])
            self.Weights2 = filter_weights2
            self.Bias2 = filter_bias2
            self.PaddingSize2 = padding_mat2
            self.Stride2 = stride_mat2
            self.DilationFactor2 = dilation_mat2
        else:
            raise Exception('wrong parameter type')

        # elif nargs == 6:
        #     layer_name = args[0]
        #     filter_weights = args[1]
        #     filter_bias = args[2]
        #     padding_mat = args[3]
        #     stride_mat = args[4]
        #     dilation_mat = args[5]
        #     self.NumInputs = 1
        #     self.InputNames = 'in'
        #     self.NumOutputs = 1
        #     self.OutputNames = 'out'
        #
        #     if isinstance(layer_name, str):
        #         self.Name = layer_name
        #     else:
        #         Exception("Layer name should be a string")
        #
        #     w = filter_weights.shape
        #     b = filter_bias.shape
        #     if len(w) != 4:
        #         Exception("Invalid weights array")
        #     if len(b) != 3:
        #         Exception("Invalid bias array")
        #     if w[3] != b[2]:
        #         Exception("Inconsistency between filter weights and filter bias")
        #
        #     self.NumFilters = w[3]
        #     self.NumChannels = w[2]
        #     self.FilterSize = np.array([w[0], w[1]])
        #     self.Weights = filter_weights
        #     self.Bias = filter_bias
        #
        #     p = padding_mat.shape
        #     if len(p) != 2 | p[1] != 4 | p[0] != 1:
        #         Exception("Invalid padding matrix")
        #     self.PaddingSize = padding_mat
        #
        #     s = stride_mat.shape
        #     if len(s) != 2 | s[0] != 1:
        #         Exception("Invalid stride matrix")
        #     self.Stride = stride_mat
        #
        #     d = dilation_mat.shape
        #     if len(d) != 2 | d[0] != 1:
        #         Exception("Invalid dilation matrix")
        #     self.DilationFactor = dilation_mat
        #
        # elif nargs == 5:
        #     filter_weights = args[0]
        #     filter_bias = args[1]
        #     padding_mat = args[2]
        #     stride_mat = args[3]
        #     dilation_mat = args[4]
        #     self.NumInputs = 1
        #     self.InputNames = 'in'
        #     self.NumOutputs = 1
        #     self.OutputNames = 'out'
        #     self.Name = 'merged_conv_layer'
        #
        #     w = filter_weights.shape
        #     b = filter_bias.shape
        #     if len(w) == 2:
        #         self.NumFilters = 1
        #         self.NumChannels = 1
        #         self.FilterSize = np.array([w[0], w[1]])
        #         self.Weights = filter_weights
        #     elif len(w) == 3:
        #         self.NumFilters = 1
        #         self.NumChannels = w[2]
        #         self.FilterSize = np.array([w[0], w[1]])
        #         self.Weights = filter_weights
        #     elif len(w) == 4:
        #         self.NumFilters = w[3]
        #         self.NumChannels = w[2]
        #         self.FilterSize = np.array([w[0], w[1]])
        #         self.Weights = filter_weights
        #     else:
        #         Exception("Invalid weight matrix")
        #
        #     if len(b) != 3:
        #         Exception("Invalid bias array")
        #     else:
        #         self.Bias = filter_bias
        #
        #     if len(w) == 4 & w[3] != b[2]:
        #         Exception("Inconsistency between filter weights and filter bias")
        #
        #     p = padding_mat.shape
        #     if len(p) != 2 | p[1] != 4 | p[0] != 1:
        #         Exception("Invalid padding matrix")
        #     self.PaddingSize = padding_mat
        #
        #     s = stride_mat.shape
        #     if len(s) != 2 | s[0] != 1:
        #         Exception("Invalid stride matrix")
        #     self.Stride = stride_mat
        #
        #     d = dilation_mat.shape
        #     if len(d) != 2 | d[0] != 1:
        #         Exception("Invalid dilation matrix")
        #     self.DilationFactor = dilation_mat
        # elif nargs == 2:
        #     filter_weights = args[0]
        #     filter_bias = args[1]
        #     self.Name = 'merged_conv_layer'
        #
        #     w = filter_weights.shape
        #     b = filter_bias.shape
        #     if len(w) == 2:
        #         self.NumFilters = 1
        #         self.NumChannels = 1
        #         self.FilterSize = np.array([w[0], w[1]])
        #         self.Weights = filter_weights
        #     elif len(w) == 3:
        #         self.NumFilters = 1
        #         self.NumChannels = w[2]
        #         self.FilterSize = np.array([w[0], w[1]])
        #         self.Weights = filter_weights
        #     elif len(w) == 4:
        #         self.NumFilters = w[3]
        #         self.NumChannels = w[2]
        #         self.FilterSize = np.array([w[0], w[1]])
        #         self.Weights = filter_weights
        #     else:
        #         Exception("Invalid weight matrix")
        #
        #     if len(b) != 3:
        #         Exception("Invalid bias array")
        #     else:
        #         self.Bias = filter_bias
        #
        #     if len(w) == 4 & w[3] != b[2]:
        #         Exception("Inconsistency between filter weights and filter bias")
        #
        #     self.Stride = np.array([1,1])
        #     self.DilationFactor = np.array([1, 1])
        #     self.PaddingMode = 'manual'
        #     self.PaddingSize = np.array([0,0,0,0])
        # else:
        #     Exception("Invalid number of input, should be 2/5/6/10 ")


    # def set_stride(self, stride):
    #     n, m = stride.shape
    #     if n != 1:
    #         Exception("Stride matrix should have one row")
    #     if m == 1:
    #         self.Stride = np.array([stride, stride])
    #     elif m == 2:
    #         self.Stride = np.array([stride[0], stride[1]])
    #     else:
    #         Exception("Invalid stride matrix")
    #
    #
    # def set_padding(self, padding):
    #     n, m = padding.shape
    #     if n != 1:
    #         Exception("Padding matrix should have one row")
    #
    #     if m == 1:
    #         self.PaddingSize = np.array([padding, padding])
    #     elif m == 2:
    #         self.PaddingSize = np.array([padding[0], padding[1]])
    #     elif m == 4:
    #         self.PaddingSize = np.array([padding[0], padding[1], padding[2], padding[3]])
    #     else:
    #         Exception("Invalid padding matrix")
    #
    #
    # def set_weights_biases(self, weights, biases):
    #     w = weights.shape
    #     b = biases.shape
    #
    #     if len(w) == 2:
    #         self.NumFilters = 1
    #         self.NumChannels = 1
    #         self.Weights = weights
    #     elif len(w) == 3:
    #         self.NumFilters = 1
    #         self.NumChannels = w[2]
    #         self.Weights = weights
    #     elif len(w) == 4:
    #         self.NumFilters = w[3]
    #         self.NumChannels = w[2]
    #         self.Weights = weights
    #     else:
    #         Exception("Invalid weight matrix")
    #
    #     if len(b) != 3:
    #         Exception("Invalid bias array")
    #     else:
    #         self.Bias = biases


    def evaluate(self, *args):
        nargs = len(args)
        if nargs == 1:
            input = args[0]
            option = []
        elif nargs == 2:
            input = args[0]
            option = args[1]
        else:
            Exception("Invalid number of inputs, should be 1/2")

        y = nn.Conv2d(input, self.Weights, self.Bias, stride = self.Stride, padding = self.PaddingSize)
        return y


    def merged_reach(self, image):

        if isinstance(image, imageStar.ImageStar) is False:
            raise Exception("image is not ImageStar")

        n = image.width
        c1 = image.V[:n, :, :, 0].copy()
        v1 = image.V[:n, :, :, 1:image.numPred+1].copy()
        c2 = image.V[n:, :, :, 0].copy()
        v2 = image.V[n:, :, :, 1:image.numPred+1].copy()

        k_l = self.FilterSize1[0]

        conv1 = nn.Conv2d(c1.shape[2], self.NumFilters1, k_l, stride=self.Stride1, padding=self.PaddingSize1,
                          dilation=self.DilationFactor1)
        conv2 = nn.Conv2d(c2.shape[2], self.NumFilters2, k_l, stride=self.Stride2, padding=self.PaddingSize2,
                          dilation=self.DilationFactor2)
        # if self.NumFilters1 == 1:
        #     conv1.bias = nn.Parameter(torch.from_numpy(self.Bias1.flatten()), requires_grad=False)
        #     conv2.bias = nn.Parameter(torch.from_numpy(self.Bias2.flatten()), requires_grad=False)
        # else:
        #     conv1.bias = nn.Parameter(torch.from_numpy(self.Bias1), requires_grad=False)
        #     conv2.bias = nn.Parameter(torch.from_numpy(self.Bias2), requires_grad=False)
        conv1_v = nn.Conv2d(c1.shape[2], self.NumFilters1, k_l, bias=False, stride=self.Stride1,
                            padding=self.PaddingSize1, dilation=self.DilationFactor1)
        conv2_v = nn.Conv2d(c2.shape[2], self.NumFilters2, k_l, bias=False, stride=self.Stride2,
                            padding=self.PaddingSize2, dilation=self.DilationFactor2)

        if isinstance(self.Weights1, torch.Tensor):
            conv1.weight = nn.Parameter(self.Weights1, requires_grad=False)
            conv2.weight = nn.Parameter(self.Weights2, requires_grad=False)
            conv1.bias = nn.Parameter(self.Bias1, requires_grad=False)
            conv2.bias = nn.Parameter(self.Bias2, requires_grad=False)
            conv1_v.weight = nn.Parameter(self.Weights1, requires_grad=False)
            conv2_v.weight = nn.Parameter(self.Weights2, requires_grad=False)
        else:
            conv1.weight = nn.Parameter(torch.from_numpy(self.Weights1.transpose(3, 2, 0, 1)), requires_grad=False)
            conv2.weight = nn.Parameter(torch.from_numpy(self.Weights2.transpose(3, 2, 0, 1)), requires_grad=False)
            conv1.bias = nn.Parameter(torch.from_numpy(self.Bias1.flatten()), requires_grad=False)
            conv2.bias = nn.Parameter(torch.from_numpy(self.Bias2.flatten()), requires_grad=False)
            conv1_v.weight = nn.Parameter(torch.from_numpy(self.Weights1.transpose(3, 2, 0, 1)), requires_grad=False)
            conv2_v.weight = nn.Parameter(torch.from_numpy(self.Weights2.transpose(3, 2, 0, 1)), requires_grad=False)

        with torch.no_grad():
            new_c1 = conv1(torch.from_numpy(c1.transpose(2, 0, 1)).unsqueeze_(0))
            new_c2 = conv2(torch.from_numpy(c2.transpose(2, 0, 1)).unsqueeze_(0))
            for i in range(v1.shape[3]):
                new_v1_temp = conv1_v(torch.from_numpy(v1.transpose(2, 3, 0, 1))[:,i,:,:].unsqueeze(0))
                new_v2_temp = conv2_v(torch.from_numpy(v2.transpose(2, 3, 0, 1))[:,i,:,:].unsqueeze(0))
                if i == 0:
                    new_v1 = new_v1_temp
                    new_v2 = new_v2_temp
                else:
                    new_v1 = torch.cat((new_v1, new_v1_temp), 0)
                    new_v2 = torch.cat((new_v2, new_v2_temp), 0)

            new_y1 = np.concatenate((new_c1.numpy().transpose(2, 3, 1, 0), new_v1.numpy().transpose(2, 3, 1, 0)), axis=3)
            new_y2 = np.concatenate((new_c2.numpy().transpose(2, 3, 1, 0), new_v2.numpy().transpose(2, 3, 1, 0)), axis=3)

        new_V = np.concatenate((new_y1, new_y2))
        S = imageStar.ImageStar(new_V, image.C, image.d, image.pred_lb, image.pred_ub)
        S.inheritMaxIdx(image.MaxIdxs)
        S.inheritInputSize(image.InputSizes)
        return S

    def merged_reach_test(self, image1, image2):
        if isinstance(image1, imageStar.ImageStar) is False:
            raise Exception("image1 is not ImageStar")
        if isinstance(image2, imageStar.ImageStar) is False:
            raise Exception("image2 is not ImageStar")

        c1 = image1.V[:, :, :, 0].copy()
        v1 = image1.V[:, :, :, 1:image1.numPred + 1].copy()
        c2 = image2.V[:, :, :, 0].copy()
        v2 = image2.V[:, :, :, 1:image2.numPred + 1].copy()

        k_l = self.FilterSize1[0]
        m_l = c1.shape[0]
        shift_c = m_l - k_l + 1
        col_m = shift_c * k_l
        row_m = col_m

        conv1 = nn.Conv2d(c1.shape[2], self.NumFilters1, k_l)
        conv1.weight = nn.Parameter(torch.from_numpy(self.Weights1.transpose(2, 3, 0, 1)), requires_grad=False)
        conv2 = nn.Conv2d(c2.shape[2], self.NumFilters2, k_l)
        conv2.weight = nn.Parameter(torch.from_numpy(self.Weights2.transpose(2, 3, 0, 1)), requires_grad=False)

        if self.NumFilters1 == 1:
            conv1.bias = nn.Parameter(torch.from_numpy(self.Bias1.flatten()), requires_grad=False)
            conv2.bias = nn.Parameter(torch.from_numpy(self.Bias2.flatten()), requires_grad=False)
        else:
            conv1.bias = nn.Parameter(torch.from_numpy(self.Bias1.transpose(1, 0)), requires_grad=False)
            conv2.bias = nn.Parameter(torch.from_numpy(self.Bias2.transpose(1, 0)), requires_grad=False)

        conv1_v = nn.Conv2d(c1.shape[2], v1.shape[3], k_l, bias=False)
        conv1_v.weight = nn.Parameter(torch.from_numpy(self.Weights1.transpose(2, 3, 0, 1)), requires_grad=False)
        conv2_v = nn.Conv2d(c2.shape[2], self.NumFilters2, k_l, bias=False)
        conv2_v.weight = nn.Parameter(torch.from_numpy(self.Weights2.transpose(2, 3, 0, 1)), requires_grad=False)

        with torch.no_grad():
            new_c1 = conv1(torch.from_numpy(c1.transpose(2, 0, 1)).unsqueeze_(0))
            new_c2 = conv2(torch.from_numpy(c2.transpose(2, 0, 1)).unsqueeze_(0))
            for i in range(v1.shape[3]):
                new_v1_temp = conv1_v(torch.from_numpy(v1.transpose(2, 3, 0, 1))[:, i, :, :].unsqueeze(1))
                new_v2_temp = conv2_v(torch.from_numpy(v2.transpose(2, 3, 0, 1))[:, i, :, :].unsqueeze(1))
                if i == 0:
                    new_v1 = new_v1_temp
                    new_v2 = new_v2_temp
                else:
                    new_v1 = torch.cat((new_v1, new_v1_temp), 1)
                    new_v2 = torch.cat((new_v2, new_v2_temp), 1)

            new_y1 = np.concatenate((new_c1.numpy().transpose(2, 3, 0, 1), new_v1.numpy().transpose(2, 3, 0, 1)),
                                    axis=3)
            new_y2 = np.concatenate((new_c2.numpy().transpose(2, 3, 0, 1), new_v2.numpy().transpose(2, 3, 0, 1)),
                                    axis=3)

        S1 = imageStar.ImageStar(new_y1, image1.C, image1.d, image1.pred_lb, image1.pred_ub)
        S2 = imageStar.ImageStar(new_y2, image2.C, image2.d, image2.pred_lb, image2.pred_ub)
        return S1, S2

    def multi_merged_reach(self, in_images):
        IS = []
        if isinstance(in_images, list):
            n = len(in_images)
            for i in range(n):
                IS.append(self.merged_reach(in_images[i]))
        elif isinstance(in_images, imageStar.ImageStar):
            IS.append(self.merged_reach(in_images))

        return IS

    def reach_star_single_input(self, input):

        if isinstance(input, imageStar.ImageStar) is False:
            Exception("The input is not an ImageStar")

        if input.numChannel != self.NumChannels:
            Exception("Input set contains %d channels while the conv layer has %d channels"
                      % (input.numChannel, self.NumChannels))

        c = nn.Conv2d(input.V[:,:,:,0])
        V = nn.Conv2d(input.V[:,:,:,1:input.numPred])
        Y = np.concatenate((c,V),axis=3)
        S = imageStar.ImageStar(Y, input.C, input.d, input.pred_lb, input.pred_ub)
        return S

    def reach_star_multipleInputs(self, in_images, option):

        n = len(in_images)
        images = []
        for i in range(n):
            images.append(self.reach_star_single_input(in_images[i]))

        return images

    def reach(self, *args):
        nargs = len(args)
        if nargs == 6:
            in_images = args[0]
            method = args[1]
            option = args[2]
        elif nargs == 5:
            in_images = args[0]
            method = args[1]
            option = args[2]
        elif nargs == 4:
            in_images = args[0]
            method = args[1]
            option = args[2]
        elif nargs == 3:
            in_images = args[0]
            method = args[1]
            option = args[2]
        elif nargs == 2:
            in_images = args[0]
            method = args[1]
            option = 'single'
        elif nargs == 1:
            in_images = args[0]
            method = 'approx-star'
            option = 'single'
        else:
            Exception("Invalid number of input, should be 1/2/3/4/5/6")

        # if method == 'approx-star':
        #     images = self.reach_star_multipleInputs(in_images,option)
        IS = self.multi_merged_reach(in_images)

        return IS

    def parse(self, conv2d):
        if isinstance(conv2d, layer.MergedConv2d) is False:
            Exception("Input is not a MergedConv2d")

        layer_name = conv2d.Name
        filter_weights = conv2d.Weights
        filter_bias = conv2d.Bias
        padding_mat = conv2d.PaddingSize
        stride_mat = conv2d.Stride
        dilation_mat = conv2d.DilationFactor

        L = MergedConv2d(layer_name,filter_weights, filter_bias, padding_mat, stride_mat, dilation_mat,
                         conv2d.NumInputs, conv2d.InputNames, conv2d.NumOutputs, conv2d.OutputNames)
        return L

class ReLULayer(object):

    def __init__(self, *args):
        nargs = len(args)
        if nargs == 5:
            self.Name = args[0]
            self.NumInputs = args[1]
            self.InputNames = args[2]
            self.NumOutputs = args[3]
            self.OutputNames = args[4]
        elif nargs == 1:
            self.Name = args[0]
        elif nargs == 0:
            self.Name = 'relu_layer'
        else:
            raise Exception("Invalid number of input")


    def evaluate(self, input):

        y = nn.Relu(input)
        return y

    def reach_star_exact(self, *args):
        # using Star to compute approximation reachability
        nargs = len(args)
        lp_solver = 'linprog'
        if nargs == 3:
            I = args[0]
            method = args[1]
            option = args[2]
        elif nargs == 2:
            I = args[0]
            method = args[1]
            option = 'single'
        elif nargs == 1:
            I = args[0]
            method = 'approx-star'
            option = 'single'
        else:
            raise Exception("Invalid number of input, should be 1/2/3")

        if isinstance(I, star.Star) is False:
            raise Exception("Input is not a Star")

        if I is None:
            S = []
        else:
            lb, ub = I.estimateRanges()
            if (lb is None)|(ub is None):
                S = []
            else:
                map = np.argwhere(ub<=0)
                V = I.V.copy()
                V[map, :] = 0
                if hasattr(I, 'Z') is False:
                    new_Z = None
                elif I.Z is not None:
                    c1 = I.Z.c.copy()
                    c1[map, :] = 0
                    V1 = I.Z.V.copy()
                    V1[map, :] = 0
                    new_Z = zono.Zono(c1, V1)
                else:
                    new_Z = None
                In = star.Star(V, I.C, I.d, I.predicate_lb, I.predicate_ub, new_Z)
                map = np.argwhere(np.logical_and(lb<0, ub>0))
                m = len(map)
                for i in range(m):
                    In = self.stepReachMultipleInputs(In, map[i], option, lp_solver)
                    # print("Iter %d, %d star" % (i, len(In)))
                S = In

        return S

    def reach_star_approx(self, *args):
        # using Star to compute approximation reachability
        nargs = len(args)
        if nargs == 3:
            I = args[0]
            method = args[1]
            option = args[2]
        elif nargs == 2:
            I = args[0]
            method = args[1]
            option = 'single'
        elif nargs == 1:
            I = args[0]
            method = 'approx-star'
            option = 'single'
        else:
            raise Exception("Invalid number of input, should be 1/2/3")

        if isinstance(I, star.Star) is False:
            raise Exception("Input is not a Star")

        if I is None:
            S = []
        else:
            lb, ub = I.estimateRanges() # column vector
            if (lb is None) | (ub is None):
                S = []
            else:
                map1 = np.argwhere(ub<=0) # gives index
                map2 = np.argwhere(np.logical_and(lb<0,ub>0))
                xmax = I.getMaxs(map2)
                map3 = np.argwhere(xmax <= 0)
                n = len(map3)
                map4 = np.zeros((n,1), dtype=map1.dtype)
                In_exist = True
                for i in range(n):
                    map4[i,0] = map2[map3[i]]
                # if (map1.size != 0) & (map4.size != 0):
                # if len(map4) == 0:
                #     In = I.resetRow(map4)
                # else:
                map11 = np.concatenate((map1, map4))
                In = I.resetRow(map11) # reset to zero at the element having ub <= 0
                # In_exist = True

                # find all indexes that have lb < 0 and ub > 0, then apply
                # the over-approximation rule for ReLU
                map5 = np.argwhere(xmax > 0)
                map6 = map2[map5[:]]    # all indexes having ub > 0
                xmax1 = xmax[map5[:]]   # upper bound of all neurons having ub > 0

                xmin = I.getMins(map6)
                map7 = np.argwhere(xmin < 0)
                map8 = map6[map7[:]]    # all indexes having lb < 0 & ub > 0
                lb1 = xmin[map7[:]]     # lower bound of all indexes having lb < 0 & ub > 0
                ub1 = xmax1[map7[:]]    # upper bound of all neurons having lb < 0 & ub > 0

                if In_exist:
                    S = self.multipleStepReachStarApprox_at_one(In, map8, lb1, ub1)
                else:
                    S = I
        return S

    def stepReachMultipleInputs(self, *args):
        nargs = len(args)
        if nargs == 4:
            I = args[0]
            index = args[1]
            option = args[2]
            lp_solver = args[3]
        elif nargs == 3:
            I = args[0]
            index = args[1]
            option = args[2]
            lp_solver = 'linprog'
        else:
            raise Exception("Invalid number of input, should be 3/4")

        if isinstance(I, star.Star):
            S = []
            temp = self.stepReach(I, index, lp_solver)
            if isinstance(temp, star.Star):
                S.append(temp)
            elif isinstance(temp, list):
                S.extend(temp)
        elif isinstance(I, list):
            p = len(I)
            S = []
            for i in range(p):
                temp = self.stepReach(I[i], index, lp_solver)
                if isinstance(temp, star.Star):
                    S.append(temp)
                elif isinstance(temp, list):
                    S.extend(temp)
        else:
            raise Exception("Wrong data type I")

        return S

    def stepReach(self, *args):
        nargs = len(args)
        if nargs == 3:
            I = args[0]
            index = args[1]
            lp_solver = args[2]
        elif nargs == 2:
            I = args[0]
            index = args[1]
            lp_solver = 'linprog'
        else:
            raise Exception("Invalid number of input, should be 2/3")

        if isinstance(I, star.Star) is False:
            raise Exception("Input is not a star set")

        xmin = I.getMin(index)

        if xmin >= 0:
            S = I
        else:
            xmax = I.getMax(index)
            if xmax <= 0:
                V1 = I.V.copy()
                V1[index, :] = 0
                if hasattr(I, 'Z') is False:
                    new_Z = None
                elif I.Z is not None:
                    c = I.Z.c.copy()
                    c[index] = 0
                    V = I.Z.V.copy()
                    V[index, :] = 0
                    new_Z = zono.Zono(c, V)
                else:
                    new_Z = None
                S = star.Star(V1, I.C, I.d, I.predicate_lb, I.predicate_ub, new_Z)
            else:
                # S1 = I & x[index]<0
                c = I.V[index, 0].copy()
                V = I.V[index, 1:(I.nVar+1)].copy()
                new_C = np.concatenate((I.C, V))
                if len(I.d.shape) == 2:
                    new_d = np.concatenate((I.d, -c[:, np.newaxis]))
                else:
                    new_d = np.concatenate((I.d, -c))
                new_V = I.V.copy()
                new_V[index, :] = np.zeros((1, I.nVar+1))

                # update outer-zono
                if hasattr(I, 'Z') is False:
                    new_Z = None
                elif I.Z is not None:
                    c1 = I.Z.c.copy()
                    c1[index] = 0
                    V1 = I.Z.V.copy()
                    V1[index, :] = 0
                    new_Z = zono.Zono(c1, V1)
                else:
                    new_Z = None

                S1 = star.Star(new_V, new_C, new_d, I.predicate_lb, I.predicate_ub, new_Z)

                # S2 = I & x[index]>=0
                new_C = np.concatenate((I.C, -V))
                if len(I.d.shape) == 2:
                    new_d = np.concatenate((I.d, c[:, np.newaxis]))
                else:
                    new_d = np.concatenate((I.d, c))
                S2 = star.Star(I.V, new_C, new_d, I.predicate_lb, I.predicate_ub, I.Z)

                S = [S1, S2]

        return S

    def multipleStepReachStarApprox_at_one(self, I, index, lb, ub):

        if isinstance(I,star.Star) is False:
            raise Exception("Input is not a Star")

        N = I.dim
        m = len(index)

        if m > 0:
            V1 = I.V
            V1[index,:] = 0
            V2 = np.zeros((N,m))
            for i in range(m):
                V2[index[i], i ] = 1
            new_V = np.concatenate((V1,V2),axis=1)
            # case 0: keep the old constraints on the old predicate
            n = I.nVar
            C0 = np.concatenate((I.C, np.zeros((I.C.shape[0],m))), axis=1)
            d0 = I.d
            # case 1: y[index] >= 0
            C1 = np.concatenate((np.zeros((m,n)), -np.eye(m)), axis=1)
            d1 = np.zeros((m, 1))
            # case 2: y[index] >= x[index]
            C2 = np.concatenate((I.V[index, 1:n+1], -V2[index, 0:m]), axis=1)
            d2 = -I.V[index, 0]
            # case 3: y[index] <= (ub/(ub-lb))*(x-lb)
            a = ub/(ub-lb)
            b = a*lb
            C3 = np.concatenate((-a*I.V[index, 1:n+1], V2[index, 0:m]), axis=1)
            d3 = a*I.V[index, 0] - b

            new_C = np.concatenate((C0, C1, C2, C3))
            new_d = np.concatenate((d0, d1, d2, d3))
            new_pred_lb = np.concatenate((I.predicate_lb, np.zeros((m, 1))))
            new_pred_ub = np.concatenate((I.predicate_ub, ub))
            S = star.Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub)
        else:
            S = star.Star(I.V.copy(), I.C.copy(), I.d.copy(), I.predicate_lb.copy(), I.predicate_ub.copy())

        return S

    def reach_star_single_input(self, in_image, method, option):

        if isinstance(in_image, imageStar.ImageStar) is False:
            raise Exception("Input is not an ImageStar")

        h = in_image.height
        w = in_image.width
        c = in_image.numChannel
        if method == 'approx-star':
            Y = self.reach_star_approx(in_image.toStar(), method, option)
        elif method == 'exact-star':
            Y = self.reach_star_exact(in_image.toStar(), method, option)

        if isinstance(Y, star.Star):
            images = Y.toImageStar(h, w, c)
            images.inheritMaxIdx(in_image.MaxIdxs)
            images.inheritInputSize(in_image.InputSizes)
        elif isinstance(Y, list):
            n = len(Y)
            images = []
            for i in range(n):
                image = Y[i].toImageStar(h, w, c)
                image.inheritMaxIdx(in_image.MaxIdxs)
                image.inheritInputSize(in_image.InputSizes)
                images.append(image)

        return images

    def reach_star_multipleInputs(self, in_images, method, option):
        if isinstance(in_images, list) is False:
            images = self.reach_star_single_input(in_images, method, option)
        else:
            n = len(in_images)
            images = []
            for i in range(n):
                image = self.reach_star_single_input(in_images[i], method, option)
                if isinstance(image, imageStar.ImageStar):
                    images.append(image)
                elif isinstance(image, list):
                    images.extend(image)

        return images

    def reach(self, *args):
        nargs = len(args)
        if nargs == 6:
            in_images = args[0]
            method = args[1]
            option = 'single'
            relaxFactor = 0
            dis_opt = []
            lp_solver = 'linprog'
        elif nargs == 5:
            in_images = args[0]
            method = args[1]
            option = 'single'
            relaxFactor = 0
            dis_opt = []
            lp_solver = 'linprog'
        elif nargs == 4:
            in_images = args[0]
            method = args[1]
            option = 'single'
            relaxFactor = 0
            dis_opt = []
            lp_solver = 'linprog'
        elif nargs == 3:
            in_images = args[0]
            method = args[1]
            option = 'single'
            relaxFactor = 0
            dis_opt = []
            lp_solver = 'linprog'
        elif nargs == 2:
            in_images = args[0]
            method = args[1]
            option = 'single'
            relaxFactor = 0
            dis_opt = []
            lp_solver = 'linprog'
        else:
            raise Exception("Invalid number of input, should be 2/3/4/5/6")

        images = self.reach_star_multipleInputs(in_images, method, option)
            #images = self.reach_star_approx(in_images, method, option)
        return images

    def parse(self, relu_layer):

        if isinstance(relu_layer, ReLULayer) is False:
            raise Exception("Input is not a ReluLayer")

        L = ReLULayer(relu_layer.Name, relu_layer.NumInputs, relu_layer.InputNames,
                      relu_layer.NumOutputs, relu_layer.OutputNames)
        return L

class BatchNormLayer(object):

    def __init__(self, nc, m1, v1, scale1, offset1, m2, v2, scale2, offset2, eps):
        self.NumChannels = nc
        self.Epsilon = eps
        self.Mean1 = m1
        self.Var1 = v1
        self.Offset1 = offset1
        self.Scale1 = scale1
        self.Mean2 = m2
        self.Var2 = v2
        self.Offset2 = offset2
        self.Scale2 = scale2
        # self.TrainedMean = trained_m
        # self.TrainedVariance = trained_v
        # self.Offset = offset
        # self.Scale = scale

    def reach_star_single_input(self, in_image):
        if isinstance(in_image, imageStar.ImageStar) is False:
            raise Exception("Wrong input images type")

        m1 = self.Mean1
        v1 = self.Var1
        s1 = self.Scale1
        o1 = self.Offset1
        m2 = self.Mean2
        v2 = self.Var2
        s2 = self.Scale2
        o2 = self.Offset2

        new_V = in_image.V.copy()
        if m1 == None:
            None
        else:
            if (len(m1.shape) == 2) & (in_image.V.shape[0] == 1) & (in_image.V.shape[1] == 1) & (len(in_image.V.shape) == 4):
                self.Mean1 = np.reshape(m1, (1, 1, m1.shape[0]))
                self.Var1 = np.reshape(v1, (1, 1, v1.shape[0]))
                if len(o1.shape) != 3:
                    self.Offset1 = np.reshape(o1, (1, 1, o1.shape[0]))
                if len(s1.shape) != 3:
                    self.Scale1 = np.reshape(s1, (1, 1, s1.shape[0]))
                self.NumChannels = 1

            nc = self.NumChannels
            l1 = np.zeros((1,1,nc))
            width = new_V.shape[1]

            for i in range(nc):
                l1[0, 0, i] = 1 / np.sqrt(self.Var1[i] + self.Epsilon)

            temp1 = np.multiply(-l1, self.Mean1)
            for i in range(nc):
                new_V[0:width, :, i, :] = new_V[0:width, :, i, :] * l1[0, 0, i] + temp1[0,0,i].item()
            # new_V[0:width, :, :, 0] = new_V[0:width, :, :, 0] + temp1
            for i in range(nc):
                new_V[0:width, :, i, :] = new_V[0:width, :, i, :] * self.Scale1[i].item() + self.Offset1[i].item()
            # new_V[0:width, :, :, 0] = new_V[0:width, :, :, 0] + self.Offset1

        if m2 == None:
            None
        else:
            if (len(m2.shape)==2) & (in_image.V.shape[0] == 1) & (in_image.V.shape[1] == 1) & (len(in_image.V.shape) == 4):
                self.Mean2 = np.reshape(m2, (1,1,m2.shape[0]))
                self.Var2 = np.reshape(v2, (1,1,v2.shape[0]))
                if len(o2.shape) != 3:
                    self.Offset2 = np.reshape(o2, (1,1,o2.shape[0]))
                if len(s2.shape) != 3:
                    self.Scale2 = np.reshape(s2, (1,1,s2.shape[0]))
                self.NumChannels = 1

            nc = self.NumChannels
            l2 = np.zeros((1,1,nc))
            width = new_V.shape[1]

            for i in range(nc):
                l2[0,0,i] = 1/np.sqrt(self.Var2[i]+self.Epsilon)

            temp2 = np.multiply(-l2, self.Mean2)
            for i in range(nc):
                new_V[width:, :, i, :] = new_V[width:, :, i, :] * l2[0, 0, i] + temp2[0,0,i].item()
            # new_V[width:, :, :, 0] = new_V[width, :, :, 0] + temp2
            for i in range(nc):
                new_V[width:, :, i, :] = new_V[width:, :, i, :] * self.Scale2[i].item() + self.Offset2[i].item()
            # new_V[width:, :, :, 0] = new_V[width, :, :, 0] + self.Offset2

        image = imageStar.ImageStar(new_V, in_image.C, in_image.d, in_image.pred_lb, in_image.pred_ub)
        # if (len(mean.shape) == 2) & (in_image.V[0] == 1) & (in_image.V[1] == 1) & (len(in_image.V.shape) == 4):
        #     self.TrainedMean = np.reshape(mean, (1,1,mean.shape[0]))
        #     self.TrainedVariance = np.reshape(var, (1,1,var.shape[0]))
        #     self.Epsilon = np.reshape(eps, (1,1,eps.shape[0]))
        #
        #     if len(offset.shape) != 3:
        #         self.Offset = np.reshape(offset, (1,1,offset.shape[0]))
        #
        #     if len(scale.shape) != 3:
        #         self.Scale = np.reshape(scale, (1,1,scale.shape[0]))
        #     self.NumChannels = 1
        #
        # nc = self.NumChannels
        # l = np.zeros((1,1,nc))
        #
        # for i in range(nc):
        #     l[0, 0, i] = 1/np.sqrt(self.TrainedVariance[0, 0, i] + self.Epsilon)
        # x = in_image.affineMap(l, np.multiply(-l, self.TrainedMean))
        # image = x.affineMap(self.Scale, self.Offset)

        return image

    def reach_star_multipleInputs(self, in_images, option):
        if isinstance(in_images, list):
            images = []
            n = len(in_images)
            for i in range(n):
                images.append(self.reach_star_single_input(in_images[i]))
        elif isinstance(in_images, imageStar.ImageStar):
            images = []
            images.append(self.reach_star_single_input(in_images))
        else:
            raise Exception("Wrong input images type")

        return images

    def reach(self, *args):
        n = len(args)
        if n == 3:
            in_images = args[0]
            method = args[1]
            option = args[2]
        elif n == 2:
            in_images = args[0]
            method = args[1]
            option = None
        else:
            raise Exception("Invalid number of input arguments, should be 3/2")

        if (method == 'exact-star') | (method == 'approx'):
            images = self.reach_star_multipleInputs(in_images, option)

        return images