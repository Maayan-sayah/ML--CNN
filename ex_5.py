import os
import sys

from gcommand_dataset import GCommandLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def load(batch_size):
    path_train, path_valid, path_test = sys.argv[1], sys.argv[2], sys.argv[3]
    train_data = GCommandLoader(path_train)
    valid_data = GCommandLoader(path_valid)
    test_data = GCommandLoader(path_test)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, valid_loader, test_loader


def train(model, train_loader, num_epochs, criterion, optimizer):
    # Train the model
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for batch, (images, labels) in enumerate(train_loader):
            # Run the forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)
            # if (batch + 1) % 100 == 0:
            #     print('Epoch [ {}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
            #           .format(epoch + 1, num_epochs, batch + 1, total_step, loss.item(),
            #                   (correct / total) * 100))



class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer6 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        #self.drop_out = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1024*2*1, 30)
        #self.fc2 = nn.Linear(1024, 30)
        #self.fc3 = nn.Linear(1024, 30)


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)

        # print(out.shape)
        #out = out.reshape(out.size(0), -1)
        out = out.view(-1, 1024 * 2 * 1)
        #out = self.drop_out(out)
        out = self.fc1(out)
        #out = self.fc2(out)
       # out = self.fc3(out)
        return F.log_softmax(out, dim=1)


def test(model, test_loader):
    # Test the model
    model.eval()
    result = []
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            [result.append(x) for x in list(predicted.numpy().flatten())]

        print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))
    return (correct / total) * 100


def testReal(model, test_loader):
    # Test the model
    model.eval()
    result = np.array([])

    with torch.no_grad():
        correct = 0
        total = 0
        i = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # [result.append([names[i],x]) for x in list(predicted.numpy().flatten())]
            result = np.append(result, np.array(predicted))
            i += 1
    return result


def find_classes(path):
    classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    classes.sort()
    return classes


def main():
    num_epochs = 7
    num_classes = 10
    batch_size = 100
    learning_rate = 0.001
    max_accuracy = 0
    max_lr = 0
    train_loader, valid_loader, test_loader = load(batch_size)
    #for i in range(1,2):
    criterion = nn.CrossEntropyLoss()
    model = Conv()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    max_model = model
    for i in range(1, 11):
        train(model, train_loader, 1, criterion, optimizer)
        accuracy = test(model, valid_loader)
        # if (accuracy >= 91) :
        #     max_model = model
        #     break
        if (max_accuracy < accuracy):
           max_model = model
           max_accuracy = accuracy
    array = []
    perdict = find_classes(sys.argv[1])
    result = testReal(max_model, test_loader)
    names = [i[0].split("\\")[2] for i in test_loader.dataset.spects]
    index = 0
    for i in result:
        array.append("{},{}".format(names[index], perdict[int(i)]))
        index += 1

    array = sorted(array, key=lambda x: int(x.split('.')[0]))

    # orderName = np.array([])
    # perdict = find_classes(sys.argv[1])
    # result = testReal(max_model, test_loader)
    # for i in range(names.__len__()):
    #     orderName = np.append("{},{}".format(names[i], perdict[result[i]]))
    # #names = np.array(names).T
    # #result = testReal(max_model, test_loader)
    # # perd = np.array([], dtype=np.string_)
    # # for i in range(result.__len__()) :
    # #     perd = np.append(perd, result[i])
    # #result = [int(result[i]) for i in range(0, names.__len__())]
    # #result = sorted(array, key=lambda x: int(x.split('.')[0]))
    a_file = open("test_y", "w")
    # array = []
    # # temp = np.char.array(perd)
    # # temp2 = np.char.array(names)
    # # array = temp2 + "," + temp
    # array = sorted(array, key=lambda x: int(x.split('.')[0]))
    # temp = np.char.array(result)
    # temp2 = np.char.array(names)
    # array = temp2 + "," + result
    # array = []
    # for i in result:
    #     perdict = find_classes(sys.argv[1])
    #     array.append("{},{}".format(names[i], perdict[i]))
    np.savetxt(a_file, array, fmt='%s', delimiter=",")

    a_file.close()
    #     if (max_accuracy < accuracy):
    #         max_accuracy = accuracy
    #         max_lr = learning_rate
    # learning_rate+= 0.001
    #test(model)


main()
