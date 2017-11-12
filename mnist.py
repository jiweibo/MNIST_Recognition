from  __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import time
import matplotlib.pyplot as plt

# settings
DATA_DIR = '../../repository/data/mnist'
enable_cuda = True

# hyper parameters
batch_size = 64
num_epoches = 10
learning_rate = 1e-3

# datasets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(DATA_DIR, train=False, download=False, transform=transform),
    batch_size=batch_size,
    shuffle=False
)


class Net(nn.Module):
    def __init__(self, cuda):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=5, stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(4096, 84)
        self.fc2 = nn.Linear(84, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.use_cuda = cuda

    def forward(self, x):
        x = self.relu(self.max_pool(self.conv1(x)))  # 12 * 12 * 16
        x = self.relu(self.conv2(x))  # 8 * 8 *64
        x = x.view(-1, 64 * 64)  # flatten
        x = self.dropout(self.relu(self.fc1(x)))  # size * 84
        x = self.fc2(x)  # size * 10
        return F.log_softmax(x)


def evaluate(model, X):
    model.eval()
    if model.use_cuda:
        X = torch.from_numpy(X.astype(np.float32)).cuda()
    else:
        X = torch.from_numpy(X.astype(np.float32))
    X = Variable(X)
    output = model(X)
    output = F.softmax(output)
    pred = output.data.max(dim=1)[1]

    if model.use_cuda:
        output, pred = output.cpu(), pred.cpu()

    c = list(range(0, 10))
    output = list(output.data.numpy().squeeze())
    dic = dict(zip(c, output))
    pred = pred.numpy().squeeze()
    return dic, pred


def build_model(enable_cuda):
    model = Net(enable_cuda)
    if model.use_cuda:
        model = model.cuda()
    if os.path.exists('mnist_params.pkl'):
        model.load_state_dict(torch.load('mnist_params.pkl'))

    optimizer = optim.Adam(model.parameters(), learning_rate, betas=(0.9, 0.999))
    return model, optimizer


def train(model, optimizer, train_loader, num_epoches):
    start_time = time.time()
    model.train()

    for epoch in range(num_epoches):
        for (batch_index, (data, target)) in enumerate(train_loader):
            correct = 0
            if model.use_cuda:
                data, target = data.cuda(), target.cuda()

            data, target = Variable(data), Variable(target)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            pred = output.data.max(dim=1)[1]
            correct += pred.eq(target.data).cpu().sum()
            if batch_index % 200 == 0:
                print('Train Epoch: {} [{} / {}]\tLoss: {:.6f}\tAccuracy: {:.4f}'.format(
                    epoch,
                    batch_index * len(data),
                    len(train_loader.dataset),
                    loss.data[0],
                    correct / target.size()[0]
                ))
    torch.save(model.state_dict(), 'mnist_params.pkl')
    print('Time: ', time.time() - start_time)


def model_eval(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    error_data = []
    error_rlabel = []
    error_flabel = []
    for data, target in test_loader:
        if model.use_cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(dim=1)[1]
        correct += pred.eq(target.data).cpu().sum()

        # Error Analysis
        error_idx = 1 - pred.eq(target.data).cpu().numpy()
        if np.sum(error_idx) > 0:
            error_data.append(data.data.cpu().numpy()[error_idx == 1])
            error_flabel.append(pred.cpu().numpy()[error_idx == 1])
            error_rlabel.append(target.data.cpu().numpy()[error_idx == 1])

    show_samples(error_data, error_rlabel, error_flabel)
    test_loss /= len(test_loader)
    print('\nTest set: Average loss : {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss,
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))


def show_samples(data, label, pred):
    fig, axes = plt.subplots(figsize=(28, 28), nrows=6, ncols=6)
    for ax, img, lb, pr in zip(axes.flatten(), data, label, pred):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.imshow(img[0].reshape(28, 28), cmap='gray')
        ax.set_title('true: {}---pred: {}'.format(lb[0], pr[0]))
    plt.savefig('error_classification.jpg')
    plt.show()


if __name__ == "__main__":
    model, optimizer = build_model(enable_cuda)
    # train(model, optimizer, train_loader, num_epoches)
    model_eval(model, test_loader)
