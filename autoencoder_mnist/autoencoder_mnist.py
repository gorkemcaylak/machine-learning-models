
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import datasets
from torch import nn
from torch.autograd import Variable

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


use_gpu = True

batch_size = 256
num_workers = 0 #?

transform = transforms.Compose([
    transforms.ToTensor()
])

train_data = datasets.MNIST(root='data', train=True,
                                   download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False,
                                  download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

print("init done")


#In[]

def randomWeights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)


epochs = 30
learning_rate = 1e-3

h_list = [128]#[32,64,128]

for h in h_list:
    autoencoderA = nn.Sequential(
                nn.Linear(28 * 28, h),
                nn.Linear(h, 28 * 28))


    autoencoderA.apply(randomWeights)
    model = autoencoderA

    # nn.init.kaiming_uniform_(coder.weight, a=1)
    if use_gpu:
        model = model.cuda()
        print("Using GPU")
    else:
        model = model.cpu()
        print("Using CPU")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #add decay 1e-5?

    losses = []

    for epoch in range(epochs):
        acc_loss = 0
        for data in train_loader:
            img, lab = data
            img = img.view(img.size(0), -1)
            img = Variable(img).cuda()
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, img)
            loss.backward()
            optimizer.step()
            acc_loss += loss

            losses.append(loss)

        if epoch % 2 == 0:
            print(f'epoch {epoch} loss {acc_loss/len(train_loader)}')
        
    it = iter(train_loader)
    img, lab = it.next()
    img = img.view(img.size(0), -1)
    img = Variable(img).cuda()
    out = model(img)
    train_loss = criterion(out, img)
    print(f'h {h} epoch {epoch} final loss {train_loss}')

    output = model(img)


    fig=plt.figure(figsize=(14,6))
    fig.suptitle(f"10 Reconstructed Digits h={h}", fontsize=10)
    digit = 0
    i = 0
    while digit<10:#for i in range(10):
        vec = output.cpu().data[i,:]
        label = lab.cpu().data[i]
        if(label == digit):
          vec = np.reshape(vec, (28,28))
          fig.add_subplot(2,5, digit+1)
          plt.xlabel(f"digit-{digit}")
          plt.imshow(vec)  
          digit +=1
        i +=1
    fig.savefig(f"HW4_A3a_10reconst_digit_h{h}_n.png")

    fig.show()  

    fig2=plt.figure(figsize=(14,6))
    fig2.suptitle(f"10 Original Digits h={h}", fontsize=10)
    digit = 0
    i = 0
    while digit<10:#for i in range(10):
        vec = img.cpu().data[i,:]
        label = lab.cpu().data[i]
        if(label == digit):
          vec = np.reshape(vec, (28,28))
          fig2.add_subplot(2,5, digit+1)
          plt.xlabel(f"digit-{digit}")
          plt.imshow(vec)  
          digit +=1
        i +=1
    fig2.savefig(f"HW4_A3a_10orig_digit_h{h}_n.png")

    fig2.show()  

    it = iter(test_loader)
    img, lab = it.next()
    img = img.view(img.size(0), -1)
    img = Variable(img).cuda()
    out = model(img)
    test_loss = criterion(out, img)
    print(f'h {h} epoch {epoch} test loss {test_loss}')

#In[]
acc_loss = 0
epochs=5
for epoch in range(epochs):
    for data in test_loader:
        img, lab = data
        img = img.view(img.size(0), -1)
        img = Variable(img).cuda()
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, img)
        loss.backward()
        optimizer.step()
        acc_loss += loss
print(f"reconstruction error: {acc_loss/len(test_loader)/epochs}")
#Average test error 


#In[]

'''

with activation layers


'''

epochs = 30
learning_rate = 1e-3

h_list = [32,64,128]

for h in h_list:
    autoencoderB = nn.Sequential(
                nn.Linear(28 * 28, h),
                nn.ReLU(),
                nn.Linear(h, 28 * 28),
                nn.ReLU()
                )
    
    autoencoderB.apply(randomWeights)
    model = autoencoderB
    
    if use_gpu:
        model = model.cuda()
        print("Using GPU")
    else:
        model = model.cpu()
        print("Using CPU")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []

    for epoch in range(epochs):
        acc_loss = 0
        for data in train_loader:
            img, lab = data
            img = img.view(img.size(0), -1)
            img = Variable(img).cuda()
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, img)
            loss.backward()
            optimizer.step()
            acc_loss += loss

            losses.append(loss)

          
        if epoch % 2 == 0:
            print(f'epoch {epoch} loss {acc_loss/len(train_loader)}')
    it = iter(train_loader)
    img, lab = it.next()
    img = img.view(img.size(0), -1)
    img = Variable(img).cuda()
    out = model(img)
    train_loss = criterion(out, img)
    print(f'h {h} epoch {epoch} final loss {train_loss}')

    # plt.plot(losses)
    # plt.title(f"Loss vs epoch A3b h={h}")
    # plt.savefig(f"Loss_A3b_h={h}")
    # plt.show()

    output = model(img)

    fig=plt.figure(figsize=(14,6))
    fig.suptitle(f"10 Reconstructed Digits h={h}", fontsize=10)
    digit = 0
    i = 0
    while digit<10:#for i in range(10):
        vec = output.cpu().data[i,:]
        label = lab.cpu().data[i]
        if(label == digit):
          vec = np.reshape(vec, (28,28))
          fig.add_subplot(2,5, digit+1)
          plt.xlabel(f"digit-{digit}")
          plt.imshow(vec)  
          digit +=1
        i +=1
    fig.savefig(f"HW4_A3b_10reconst_digit_h{h}_n.png")

    fig.show()  

    fig2=plt.figure(figsize=(14,6))
    fig2.suptitle(f"10 Original Digits h={h}", fontsize=10)
    digit = 0
    i = 0
    while digit<10:#for i in range(10):
        vec = img.cpu().data[i,:]
        label = lab.cpu().data[i]
        if(label == digit):
          vec = np.reshape(vec, (28,28))
          fig2.add_subplot(2,5, digit+1)
          plt.xlabel(f"digit-{digit}")
          plt.imshow(vec)  
          digit +=1
        i +=1
    fig2.savefig(f"HW4_A3b_10orig_digit_h{h}_n.png")

    fig2.show()  

    it = iter(test_loader)
    img, lab = it.next()
    img = img.view(img.size(0), -1)
    img = Variable(img).cuda()
    out = model(img)
    test_loss = criterion(out, img)
    print(f'h {h} epoch {epoch} test loss {test_loss}')

#In[]
acc_loss = 0
epochs=5
for epoch in range(epochs):
    for data in test_loader:
        img, lab = data
        img = img.view(img.size(0), -1)
        img = Variable(img).cuda()
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, img)
        loss.backward()
        optimizer.step()
        acc_loss += loss
print(f"reconstruction error: {acc_loss/len(test_loader)/epochs}")
#Average test error 