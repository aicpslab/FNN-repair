from torchvision import datasets
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.quantization import QuantStub, DeQuantStub
from torch.utils.data import Dataset
import imageio
import numpy as np
import matplotlib.pyplot as plt

from layer import FC
from layer import ReLULayer
from imageStar import ImageStar


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
        self.r1 = nn.ReLU(inplace=False)
        self.r2 = nn.ReLU(inplace=False)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.r1(self.fc1(x))
        x = self.r2(self.fc2(x))
        x = self.fc3(x)
        return x


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
        self.r1 = nn.ReLU()
        self.r2 = nn.ReLU()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = x.to('cpu')
        x = self.quant(x)
        x = torch.flatten(x, 1)
        x = self.r1(self.fc1(x))
        x = self.r2(self.fc2(x))
        x = self.fc3(x)
        x = self.dequant(x)
        return x


def quant_star(input, scale, offset):
    if isinstance(input, list):
        w = input[0].V.shape[1]
        ch = input[0].V.shape[2] // 2
        if w != 1:
            for i in range(len(input)):
                input[i].V[w:, :, :, 0] = np.round(input[i].V[w:, :, :, 0] / scale) + offset
                input[i].V[w:, :, :, 0] = (input[i].V[w:, :, :, 0] - offset) * scale
        elif (w == 1) & (input[0].V.shape[0] == 2):
            for i in range(len(input)):
                input[i].V[1, 0, :, 0] = np.round(input[i].V[1, 0, :, 0] / scale) + offset
                input[i].V[1, 0, :, 0] = (input[i].V[1, 0, :, 0] - offset) * scale
        else:
            for i in range(len(input)):
                input[i].V[0, 0, ch:, 0] = np.round(input[i].V[0, 0, ch:, 0] / scale) + offset
                input[i].V[0, 0, ch:, 0] = (input[i].V[0, 0, ch:, 0] - offset) * scale
    elif isinstance(input, ImageStar):
        w = input.V.shape[1]
        input.V[w:, :, :, 0] = np.round(input.V[w:, :, :, 0] / scale) + offset
        input.V[w:, :, :, 0] = (input.V[w:, :, :, 0] - offset) * scale
    return input


def err_range(Istar):
    diff_list = []
    n = len(Istar)
    for i in range(n):
        Star1 = Istar[i].toStar()
        vertices = Star1.toPolyhedron()
        if i == 0:
            for ind in range(10):
                diff_list.append([min(vertices[ind, :]), max(vertices[ind, :])])
        else:
            for ind in range(10):
                diff_list[ind] = [min([diff_list[ind][0], min(vertices[ind, :])]),
                                  max([diff_list[ind][1], max(vertices[ind, :])])]
    return diff_list


def compute_fnnnet_diff(result_path1, result_path2, IM, lb, ub):
    net1 = torch.load(result_path1, map_location=torch.device('cpu'))  # only load for parameter reading
    net2 = torch.load(result_path2, map_location=torch.device('cpu'))

    fc1_weight1 = net1['fc1.weight'].to(torch.float32)
    fc1_bias1 = net1['fc1.bias'].to(torch.float32)
    fc2_weight1 = net1['fc2.weight'].to(torch.float32)
    fc2_bias1 = net1['fc2.bias'].to(torch.float32)
    fc3_weight1 = net1['fc3.weight'].to(torch.float32)
    fc3_bias1 = net1['fc3.bias'].to(torch.float32)

    if 'fc1._packed_params._packed_params' in net2.keys():
        fc1_weight2 = net2['fc1._packed_params._packed_params'][0].dequantize().to(torch.float32)
        fc1_bias2 = net2['fc1._packed_params._packed_params'][1].dequantize().to(torch.float32)
        fc2_weight2 = net2['fc2._packed_params._packed_params'][0].dequantize().to(torch.float32)
        fc2_bias2 = net2['fc2._packed_params._packed_params'][1].dequantize().to(torch.float32)
        fc3_weight2 = net2['fc3._packed_params._packed_params'][0].dequantize().to(torch.float32)
        fc3_bias2 = net2['fc3._packed_params._packed_params'][1].dequantize().to(torch.float32)
    elif 'fc1.weight' in net2.keys():
        fc1_weight2 = net2['fc1.weight'].to(torch.float32)
        fc1_bias2 = net2['fc1.bias'].to(torch.float32)
        fc2_weight2 = net2['fc2.weight'].to(torch.float32)
        fc2_bias2 = net2['fc2.bias'].to(torch.float32)
        fc3_weight2 = net2['fc3.weight'].to(torch.float32)
        fc3_bias2 = net2['fc3.bias'].to(torch.float32)
    else:
        raise Exception('Wrong model')

    method = 'exact-star'
    LB = np.zeros((28, 28, 1), dtype=np.single)
    UB = np.zeros((28, 28, 1), dtype=np.single)
    LB[13:15, 13:15, :] = lb
    UB[13:15, 13:15, :] = ub
    IM_m = np.concatenate((IM, IM))
    LB_m = np.concatenate((LB, LB))
    UB_m = np.concatenate((UB, UB))
    I1_m = ImageStar(IM_m, LB_m, UB_m)

    l_fc1 = FC(fc1_weight1, fc1_bias1, fc1_weight2, fc1_bias2)
    l_fc2 = FC(fc2_weight1, fc2_bias1, fc2_weight2, fc2_bias2)
    l_fc3 = FC(fc3_weight1, fc3_bias1, fc3_weight2, fc3_bias2)
    fc4_weight1 = np.eye(10, dtype=np.single)
    fc4_weight2 = -np.eye(10, dtype=np.single)
    fc4_b1 = np.zeros((1, 2))
    fc4_b2 = np.zeros((1, 2))
    l_fc4 = FC(fc4_weight1, fc4_b1, fc4_weight2, fc4_b2, 'last')
    l_relu = ReLULayer()

    if 'fc1._packed_params._packed_params' in net2.keys():
        I1_m = quant_star(I1_m, net2['quant.scale'].numpy(), net2['quant.zero_point'].numpy())
        Istar1 = l_fc1.reach(I1_m)
        Istar2 = l_relu.reach(Istar1, method)

        Istar2 = quant_star(Istar2, net2['fc1.scale'].numpy(), net2['fc1.zero_point'].numpy())
        Istar3 = l_fc2.reach(Istar2)
        Istar4 = l_relu.reach(Istar3, method)

        Istar4 = quant_star(Istar4, net2['fc2.scale'].numpy(), net2['fc2.zero_point'].numpy())
        Istar5 = l_fc3.reach(Istar4)

        Istar5 = quant_star(Istar5, net2['fc3.scale'].numpy(), net2['fc3.zero_point'].numpy())
        Istar6 = l_fc4.reach(Istar5)
    elif 'fc1.weight' in net2.keys():
        Istar1 = l_fc1.reach(I1_m)
        Istar2 = l_relu.reach(Istar1, method)
        Istar3 = l_fc2.reach(Istar2)
        Istar4 = l_relu.reach(Istar3, method)
        Istar5 = l_fc3.reach(Istar4)
        Istar6 = l_fc4.reach(Istar5)
    else:
        raise Exception('Wrong model!')

    return err_range(Istar6)


def fnn_repair(net1, net2, large_path, small_path, temp_path, optimizer, loss_func):
    image_folder = ['./data/image/28.jpg', './data/image/14.jpg', './data/image/38.jpg', './data/image/32.jpg',
                    './data/image/19.jpg', './data/image/53.jpg', './data/image/21.jpg', './data/image/34.jpg',
                    './data/image/110.jpg', './data/image/12.jpg']
    for idx, image_path in enumerate(image_folder):
        print('Train for label %d' % idx)
        IM = imageio.v2.imread(image_path)
        IM = IM[:, :, np.newaxis]
        IM = IM / 255
        IM = (IM - 0.5) / 0.5
        lb = -0.05
        ub = 0.05
        max_range = compute_fnnnet_diff(large_path, small_path, IM, lb, ub)
        max_range = np.array(max_range)

        for k in range(10):
            print("%d iteration" % k)

            if k == 0:
                delta_y = np.mean(max_range, 1)[np.newaxis, :]
            else:
                delta_y = np.mean(new_range, 1)[np.newaxis, :]

            with torch.no_grad():
                inputs = torch.from_numpy(IM.copy().transpose(2, 0, 1)).unsqueeze_(0).to(torch.float32)
                outputs2 = net2(inputs)
                labels = outputs2.squeeze() + (torch.from_numpy(delta_y).to(torch.float32)) / 10

            print("Start re-train")

            for i in range(3):
                if (i + 1) % 5 == 0:
                    print("%d epoch re-train" % (i + 1))
                outputs = net2(inputs.to(torch.float32))
                loss = loss_func(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            torch.save(net2.state_dict(), temp_path)
            new_range = compute_fnnnet_diff(large_path, temp_path, IM, lb, ub)
            new_range = np.array(new_range)
            for label in range(10):
                if (max(abs(new_range[label]))) < (max(abs(max_range[label]))):
                    break

        print(f"Original guaranteed output error range (Avg. {np.mean(np.mean(abs(max_range), 1))})")
        print(max_range)
        print(f"New guaranteed output error range after retrain (Avg. {np.mean(np.mean(abs(new_range), 1))})")
        print(new_range)

    return net2


def MNIST_FNN_repair():
    small_path = './mnist_fnn_s_quan.pth'
    large_path = './mnist_fnn_l_ori.pth'
    temp_path = './mnist_temp.pth'
    fused_list = [['fc1', 'r1'], ['fc2', 'r2']]
    net1 = Net1()
    net1.load_state_dict(torch.load(large_path, map_location=torch.device('cpu')))
    net2 = Net2()
    net2.load_state_dict(torch.load(large_path, map_location=torch.device('cpu')))
    optimizer = optim.SGD(net2.parameters(), lr=0.0001, momentum=0.9)
    loss_func = torch.nn.MSELoss()

    net2.eval()
    net2.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    net2 = torch.quantization.fuse_modules(net2, fused_list, inplace=True)
    net2 = torch.quantization.prepare_qat(net2.train(), inplace=True)

    net2 = fnn_repair(net1, net2, large_path, small_path, temp_path, optimizer, loss_func)

    net2 = torch.quantization.convert(net2.eval(), inplace=True)
    torch.save(net2.state_dict(), temp_path)

    root_pth = './data'
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(0.5, 0.5)])
    test_dataset = datasets.MNIST(root=root_pth, train=True,
                                  download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4,
                                             shuffle=False, num_workers=2)

    print("Accuracy of original network")
    # test on whole dataset
    correct = 0
    total = 0
    net2.load_state_dict(torch.load(small_path, map_location=torch.device('cpu')))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net2(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    print("Accuracy of re-train network")
    # test on whole dataset
    correct = 0
    total = 0
    net_new = Net2()
    net_new.eval()
    net_new.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    net_new = torch.quantization.fuse_modules(net_new, fused_list, inplace=True)
    net_new = torch.quantization.prepare_qat(net_new.train(), inplace=True)
    net_new = torch.quantization.convert(net_new.eval(), inplace=True)
    net_new.load_state_dict(torch.load(temp_path, map_location=torch.device('cpu')))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net_new(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # bar plot
    test_img = imageio.v2.imread('./data/image/12.jpg')
    test_img = test_img[:, :, np.newaxis]
    test_img = (test_img/255 - 0.5) / 0.5
    lb = -0.05
    ub = 0.05
    old_range = compute_fnnnet_diff(large_path, small_path, test_img, lb, ub)
    new_range = compute_fnnnet_diff(large_path, temp_path, test_img, lb, ub)
    x = np.linspace(0, 9, 10, dtype=int)
    with torch.no_grad():
        ori_output = net1(torch.from_numpy(test_img.copy().transpose(2, 0, 1)).unsqueeze_(0).to(torch.float32)).squeeze().numpy()
        old_upp_list = np.zeros(10)
        old_low_list = np.zeros(10)

        new_upp_list = np.zeros(10)
        new_low_list = np.zeros(10)
        for la in range(10):
            old_upp_list[la] = (ori_output[la] - old_range[la][0]).squeeze()
            old_low_list[la] = (ori_output[la] - old_range[la][1]).squeeze()
            new_upp_list[la] = (ori_output[la] - new_range[la][0]).squeeze()
            new_low_list[la] = (ori_output[la] - new_range[la][1]).squeeze()
            plt.plot(x[la], ori_output[la], color='b', marker='.')
            plt.plot([la, la], [old_low_list[la], old_upp_list[la]], linewidth=1, color='g')
            plt.plot(la, old_upp_list[la], marker='_', markersize=15, color='g')
            plt.plot(la, old_low_list[la], marker='_', markersize=15, color='g')
            plt.plot([la, la], [new_low_list[la], new_upp_list[la]], linewidth=1, color='r')
            plt.plot(la, new_upp_list[la], marker='_', markersize=15, color='r')
            plt.plot(la, new_low_list[la], marker='_', markersize=15, color='r')

    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    plt.xticks(x, classes)
    plt.title('MNIST Experiment')
    plt.xlabel('Label')
    plt.ylabel('Ranges')
    plt.show()


def FNN_plot():
    large_path = './mnist_fnn_l_ori.pth'
    small_path = './mnist_fnn_s_quan.pth'
    temp_path = './mnist_temp.pth'
    fused_list = [['fc1', 'r1'], ['fc2', 'r2']]
    alpha_list = [2, 5, 10, 20]
    root_pth = './data'
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(0.5, 0.5)])
    test_dataset = datasets.MNIST(root=root_pth, train=True,
                                  download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4,
                                             shuffle=False, num_workers=2)
    image_path = './data/image/12.jpg'
    IM = imageio.v2.imread(image_path)
    IM = IM[:, :, np.newaxis]
    IM = IM / 255
    IM = (IM - 0.5) / 0.5
    lb = -0.05
    ub = 0.05
    max_range = compute_fnnnet_diff(large_path, small_path, IM, lb, ub)
    max_range = np.array(max_range)
    dis_total = []
    acc_total = []
    for alpha in alpha_list:
        net2 = Net2()
        net2.load_state_dict(torch.load(large_path, map_location=torch.device('cpu')))
        optimizer = optim.SGD(net2.parameters(), lr=0.0001, momentum=0.9)
        loss_func = torch.nn.MSELoss()

        net2.eval()
        net2.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        net2 = torch.quantization.fuse_modules(net2, fused_list, inplace=True)
        net2 = torch.quantization.prepare_qat(net2.train(), inplace=True)
        acc_list = [91]
        dis_list = [np.mean(np.mean(abs(max_range), 1))]

        for k in range(10):
            print("%d iteration" % k)

            if k == 0:
                delta_y = np.mean(max_range, 1)[np.newaxis, :]
            else:
                delta_y = np.mean(new_range, 1)[np.newaxis, :]

            with torch.no_grad():
                inputs = torch.from_numpy(IM.copy().transpose(2, 0, 1)).unsqueeze_(0).to(torch.float32)
                outputs2 = net2(inputs)
                labels = outputs2.squeeze() + (torch.from_numpy(delta_y).to(torch.float32)) / alpha

            print("Start re-train")
            for i in range(3):
                if (i + 1) % 5 == 0:
                    print("%d epoch re-train" % (i + 1))
                outputs = net2(inputs.to(torch.float32))
                loss = loss_func(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            torch.save(net2.state_dict(), temp_path)
            new_range = compute_fnnnet_diff(large_path, temp_path, IM, lb, ub)
            new_range = np.array(new_range)
            dis_list.append(np.mean(np.mean(abs(new_range), 1)))

            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    outputs = net2(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            acc_list.append(100 * correct // total)
        dis_total.append(dis_list)
        acc_total.append(acc_list)

    x = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
    plt.figure()
    plt.plot(x, dis_total[0], color='blue', label='alpha=2')
    plt.plot(x, dis_total[1], color='yellow', label='alpha=5')
    plt.plot(x, dis_total[2], color='red', label='alpha=10')
    plt.plot(x, dis_total[3], color='purple', label='alpha=20')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Discrepancy')
    plt.show()

    plt.figure()
    plt.plot(x, acc_total[0], color='blue', label='alpha=2')
    plt.plot(x, acc_total[1], color='yellow', label='alpha=5')
    plt.plot(x, acc_total[2], color='red', label='alpha=10')
    plt.plot(x, acc_total[3], color='purple', label='alpha=20')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()


if __name__ == '__main__':
    MNIST_FNN_repair()
    FNN_plot()
