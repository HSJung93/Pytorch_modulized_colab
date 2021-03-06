import torch
import torchvision
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader, Dataset

def datasets(mode, batchsize):

    print(mode)
    path = './'
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if mode == "CIFAR10":
        
        trainset = torchvision.datasets.CIFAR10(root = path + 'data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root = path + 'data', train=False, download=True, transform=transform)

    elif mode == 'CIFAR50':

        trainset = torchvision.datasets.FashionMNIST(root = path + 'data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root = path + 'data', train=False, download=True, transform=transform)

    elif mode == 'custom':

        trainset = torchvision.datasets.ImageFolder(root = path + 'data/train', transform=transform)
        testset = torchvision.datasets.ImageFolder(root = path + 'data/train', transform=transform)

    else:
        print('Please check your dataset once again')

    trainloader = torch.utils.data.DataLoader(trainset, batch_size = batchsize, shuffle = True)
    testloader = torch.utils.data.DataLoader(testset, batch_size = batchsize, shuffle = False)
    
    return trainloader, testloader
