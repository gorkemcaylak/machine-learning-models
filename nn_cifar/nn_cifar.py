
# In[]
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])  #(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


traindataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
 
trainset, _ = torch.utils.data.random_split(traindataset, [40000,10000])


valdataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
 
_, validset = torch.utils.data.random_split(valdataset, [40000,10000])


trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=8)
                                        
validloader = torch.utils.data.DataLoader(validset, batch_size=4,
                                          shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

! pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html 
! pip install tqdm matplotlib

use_gpu = True

if use_gpu:
    print(torch.zeros(10).cuda())
else:
    print(torch.zeros(10))


#In[]


# In[]


'''             a            
Fully-connected output, 0 hidden layers (logistic regression): this network has no
hidden layers and linearly maps the input layer to the output layer. This can be written as
x = Wvec(x) + b
'''

class aNet(nn.Module):
    def __init__(self):
        super(aNet, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 10)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = self.fc1(x)
        return x


# In[]


'''


a

Finished Training for optim:sgd LR:0.003 BS:1000 M:0.9
Finished Training for optim:sgd LR:0.01 BS:2000 M:0.9 *

'''
rate_list = [0.01]#[0.001, 0.01, 0.1]
# mom_list = [0.95, 0.9]
mom = 0.9
batch_list = [2000]#[500,2000]
optim_list = ['sgd']#['sgd', 'adam']
epochs = 12

for rate in rate_list:
    for batch_size in batch_list:
        for optimm in optim_list:
            train_acc = []
            valid_acc = []
            net = aNet()
            criterion = nn.CrossEntropyLoss()
            
            if(optimm == 'sgd'):
                optimizer = optim.SGD(net.parameters(), lr=rate, momentum=mom)

            elif(optimm == 'adam'):
                optimizer = optim.Adam(net.parameters(), lr=rate)      

            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                      shuffle=True,pin_memory=True, num_workers=8)

            if use_gpu:
                net = net.cuda()
                print("Using GPU")
            else:
                net = net.cpu()
                print("Using CPU")


            batch_count = len(trainloader)
            #In[]
            for epoch in range(epochs):  # loop over the dataset multiple times

                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data

                    if use_gpu:
                        inputs, labels = inputs.cuda(), labels.cuda()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    if i % (batch_count) == ((batch_count)-1):    
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i + 1, running_loss / batch_count))
                        running_loss = 0.0
                correct = 0
                total = 0
                with torch.no_grad():
                    for data in trainloader:
                        images, labels = data
                        if use_gpu:
                            images, labels = images.cuda(), labels.cuda()
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                print('Accuracy of the network on the 40000 train images: %.3f %%' % (
                    100 * correct / total))
                
                train_acc.append(correct / total)

                correct = 0
                total = 0
                with torch.no_grad():
                    for data in validloader:
                        images, labels = data
                        if use_gpu:
                            images, labels = images.cuda(), labels.cuda()
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                print('Accuracy of the network on the 10000 valid images: %.3f %%' % (
                    100 * correct / total))
                valid_acc.append(correct / total)


                if(epoch %10 == 9):
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for data in testloader:
                            images, labels = data
                            if use_gpu:
                                images, labels = images.cuda(), labels.cuda()
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()

                    print('Accuracy of the network on the 10000 test images: %.3f %%' % (
                    100 * correct / total))
            epoch_range = np.arange(1,epochs+1)
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    if use_gpu:
                        images, labels = images.cuda(), labels.cuda()
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            epoch_range = np.arange(1,epochs+1)
            
            plt.plot(epoch_range, train_acc, label='Training Accuracy')
            plt.plot(epoch_range, valid_acc, label='Validation Accuracy')
            plt.legend()
            plt.title(f'Model a with optim:{optimm} rate:{rate} momentum:{mom} batch:{batch_size}')
            plt.savefig(f'hw4_a5_a_{optimm}_lr{rate}_bs{batch_size}_m{mom}.png')
            print(f'Finished Training for optim:{optimm} LR:{rate} BS:{batch_size} M:{mom}')
            plt.close()

#In[]

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        if use_gpu:
            images, labels = images.cuda(), labels.cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %.3f %%' % (
    100 * correct / total))


# In[]

'''             

b


'''

class bNet(nn.Module):
    def __init__(self,M):
        super(bNet, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, M)
        self.fc2 = nn.Linear(M, 10)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# In[]

'''


b


'''



rate_list = [0.001]#[0.001, 0.01, 0.1]
mom = 0.9
batch_list = [500]
optim_list = ['adam']#['sgd', 'adam']
epochs = 15
M_list = [500]
for M in M_list:
    for rate in rate_list:
        for batch_size in batch_list:
            for optimm in optim_list:
                train_acc = []
                valid_acc = []
                net = bNet(M)
                criterion = nn.CrossEntropyLoss()
                
                if(optimm == 'sgd'):
                    optimizer = optim.SGD(net.parameters(), lr=rate, momentum=mom)

                elif(optimm == 'adam'):
                    optimizer = optim.Adam(net.parameters(), lr=rate)      

                trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                          shuffle=True,pin_memory=True, num_workers=8)

                if use_gpu:
                    net = net.cuda()
                    print("Using GPU")
                else:
                    net = net.cpu()
                    print("Using CPU")


                batch_count = len(trainloader)
                #In[]
                for epoch in range(epochs):  # loop over the dataset multiple times

                    running_loss = 0.0
                    for i, data in enumerate(trainloader, 0):
                        # get the inputs; data is a list of [inputs, labels]
                        inputs, labels = data

                        if use_gpu:
                            inputs, labels = inputs.cuda(), labels.cuda()

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward + backward + optimize
                        outputs = net(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        # print statistics
                        running_loss += loss.item()
                        # print(batch_size,batch_count)
                        if i % (batch_count) == ((batch_count)-1):    # print every 2000 mini-batches
                            print('[%d, %5d] loss: %.3f' %
                                  (epoch + 1, i + 1, running_loss / batch_count))
                            running_loss = 0.0
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for data in trainloader:
                            images, labels = data
                            if use_gpu:
                                images, labels = images.cuda(), labels.cuda()
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()

                    print('Accuracy of the network on the 40000 train images: %.3f %%' % (
                        100 * correct / total))
                    
                    train_acc.append(correct / total)

                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for data in validloader:
                            images, labels = data
                            if use_gpu:
                                images, labels = images.cuda(), labels.cuda()
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()

                    print('Accuracy of the network on the 10000 valid images: %.3f %%' % (
                        100 * correct / total))
                    valid_acc.append(correct / total)


                    if(epoch %10 == 9):
                        correct = 0
                        total = 0
                        with torch.no_grad():
                            for data in testloader:
                                images, labels = data
                                if use_gpu:
                                    images, labels = images.cuda(), labels.cuda()
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                total += labels.size(0)
                                correct += (predicted == labels).sum().item()

                        print('Accuracy of the network on the 10000 test images: %.3f %%' % (
                        100 * correct / total))
                epoch_range = np.arange(1,epochs+1)
                correct = 0
                total = 0
                with torch.no_grad():
                    for data in testloader:
                        images, labels = data
                        if use_gpu:
                            images, labels = images.cuda(), labels.cuda()
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                epoch_range = np.arange(1,epochs+1)
                
                plt.plot(epoch_range, train_acc, label='Training Accuracy')
                plt.plot(epoch_range, valid_acc, label='Validation Accuracy')
                plt.legend()
                plt.title(f'Model b with optim:{optimm} rate:{rate} momentum:{mom} batch:{batch_size} M:{M}')
                plt.savefig(f'hw4_a5_b_{optimm}_lr{rate}_bs{batch_size}_m{mom}_M{M}.png')
                print(f'Finished Training for optim:{optimm} LR:{rate} BS:{batch_size} M:{mom}')
                plt.close()
        #save trained model:
        # PATH = './cifar_net.pth'
        # torch.save(net.state_dict(), PATH)

        #load trained model:
        # net = Net()
        # net.load_state_dict(torch.load(PATH))


# In[]

'''               

c


'''


class cNet(nn.Module):
    def __init__(self,M):
        super(cNet, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=M, kernel_size=5, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=(14,14))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(400, 10)

    def forward(self, x):
        x = self.conv(x)
        # print("After conv: data shape:", x.shape)
        # print("Weight shape:", self.conv.weight.shape, "Bias shape:", self.conv.bias.shape)
        x = F.relu(x)
        x = self.pool(x) #size after this: 33-k/N,33-k/N,M
        # print("After pool: data shape:", x.shape)
        # x = x.view(-1, 32 * 32 * 3)
        x = self.flatten(x)
        # print("After flat(out): data shape:", x.shape)
        x = self.fc1(x)
        # print("Weight shape:", self.fc1.weight.shape, "Bias shape:", self.fc1.bias.shape)




        return x


#In[]


'''


c


'''
rate_list = [0.003, 0.001, 0.01, 0.1]
mom = 0.90
batch_list = [128, 256, 500, 1000]
optim_list = ['sgd', 'adam']
epochs = 15


for rate in rate_list:
    for batch_size in batch_list:
        for optimm in optim_list:
            train_acc = []
            valid_acc = []
            net = cNet(100)
            criterion = nn.CrossEntropyLoss()
            
            if(optimm == 'sgd'):
                optimizer = optim.SGD(net.parameters(), lr=rate, momentum=mom)

            elif(optimm == 'adam'):
                optimizer = optim.Adam(net.parameters(), lr=rate)      

            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                      shuffle=True,pin_memory=True, num_workers=8)

            if use_gpu:
                net = net.cuda()
                print("Using GPU")
            else:
                net = net.cpu()
                print("Using CPU")


            batch_count = len(trainloader)
            #In[]
            for epoch in range(epochs):  # loop over the dataset multiple times

                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data

                    if use_gpu:
                        inputs, labels = inputs.cuda(), labels.cuda()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    # print(batch_size,batch_count)
                    if i % (batch_count) == ((batch_count)-1):    # print every 2000 mini-batches
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i + 1, running_loss / batch_count))
                        running_loss = 0.0
                correct = 0
                total = 0
                with torch.no_grad():
                    for data in trainloader:
                        images, labels = data
                        if use_gpu:
                            images, labels = images.cuda(), labels.cuda()
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                print('Accuracy of the network on the 40000 train images: %.3f %%' % (
                    100 * correct / total))
                
                train_acc.append(correct / total)

                correct = 0
                total = 0
                with torch.no_grad():
                    for data in validloader:
                        images, labels = data
                        if use_gpu:
                            images, labels = images.cuda(), labels.cuda()
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                print('Accuracy of the network on the 10000 valid images: %.3f %%' % (
                    100 * correct / total))
                valid_acc.append(correct / total)

                if(epoch %10 == 9):
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for data in testloader:
                            images, labels = data
                            if use_gpu:
                                images, labels = images.cuda(), labels.cuda()
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()

                    print('Accuracy of the network on the 10000 test images: %.3f %%' % (
                    100 * correct / total))
            epoch_range = np.arange(1,epochs+1)
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    if use_gpu:
                        images, labels = images.cuda(), labels.cuda()
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('Final accuracy of the network on the 10000 test images: %.3f %%' % (
            100 * correct / total))

            plt.plot(epoch_range, train_acc, label='Training Accuracy')
            plt.plot(epoch_range, valid_acc, label='Validation Accuracy')
            plt.legend()
            plt.title(f'Model c with optim:{optimm} rate:{rate} momentum:{mom} batch:{batch_size}')
            plt.savefig(f'hw4_a5_c_{optimm}_lr{rate}_bs{batch_size}_m{mom}.png')
            print(f'Finished Training for optim:{optimm} LR:{rate} BS:{batch_size} M:{mom}')
            plt.close()


# In[]

class dNet(nn.Module):
    def __init__(self):
        super(dNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512, 196)
        self.fc2 = nn.Linear(196, 84)
        self.fc3 = nn.Linear(84, 10)
        self.flat = nn.Flatten()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = self.flat(x)
        # print(x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

#In[]



'''


d
Tune the different hyperparameters (number of convolutional filters, filter sizes, dimensionality
of the fully-connected layers, stepsize, etc.) and train for a sufficient number of iterations to achieve a test
accuracy of at least 70%.


'''
rate_list = [0.003, 0.001, 0.01, 0.1]
mom = 0.90
batch_list = [128, 256, 500, 1000]
optim_list = ['sgd', 'adam']
epochs = 15

for rate in rate_list:
    for batch_size in batch_list:
        for optimm in optim_list:
            train_acc = []
            valid_acc = []
            net = dNet()
            criterion = nn.CrossEntropyLoss()
            
            if(optimm == 'sgd'):
                optimizer = optim.SGD(net.parameters(), lr=rate, momentum=mom)

            elif(optimm == 'adam'):
                optimizer = optim.Adam(net.parameters(), lr=rate)      

            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                      shuffle=True,pin_memory=True, num_workers=8)

            if use_gpu:
                net = net.cuda()
                print("Using GPU")
            else:
                net = net.cpu()
                print("Using CPU")


            batch_count = len(trainloader)
            #In[]
            for epoch in range(epochs):  # loop over the dataset multiple times

                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data

                    if use_gpu:
                        inputs, labels = inputs.cuda(), labels.cuda()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    # print(batch_size,batch_count)
                    if i % (batch_count) == ((batch_count)-1):   
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i + 1, running_loss / batch_count))
                        running_loss = 0.0
                correct = 0
                total = 0
                with torch.no_grad():
                    for data in trainloader:
                        images, labels = data
                        if use_gpu:
                            images, labels = images.cuda(), labels.cuda()
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                print('Accuracy of the network on the 40000 train images: %.3f %%' % (
                    100 * correct / total))
                
                train_acc.append(correct / total)

                correct = 0
                total = 0
                with torch.no_grad():
                    for data in validloader:
                        images, labels = data
                        if use_gpu:
                            images, labels = images.cuda(), labels.cuda()
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                print('Accuracy of the network on the 10000 valid images: %.3f %%' % (
                    100 * correct / total))
                
                valid_acc.append(correct / total)

                correct = 0
                total = 0
                with torch.no_grad():
                    for data in testloader:
                        images, labels = data
                        if use_gpu:
                            images, labels = images.cuda(), labels.cuda()
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                print('Accuracy of the network on the 10000 test images: %.3f %%' % (
                    100 * correct / total))
            epoch_range = np.arange(1,epochs+1)
            
            plt.plot(epoch_range, train_acc, label='Training Accuracy')
            plt.plot(epoch_range, valid_acc, label='Validation Accuracy')
            plt.legend()
            plt.title(f'Model d with optim:{optimm} rate:{rate} momentum:{mom} batch:{batch_size}')
            plt.savefig(f'hw4_a5_d_{optimm}_lr{rate}_bs{batch_size}_m{mom}.png')
            print(f'Finished Training for optim:{optimm} LR:{rate} BS:{batch_size} M:{mom}')
            plt.close()


#In[]

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        if use_gpu:
            images, labels = images.cuda(), labels.cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %.3f %%' % (
    100 * correct / total))


