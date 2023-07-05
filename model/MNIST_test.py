import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1, 1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1, 1)
        self.relu2 = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.linear1 = torch.nn.Linear(14 * 14 * 64, 128)
        self.relu3 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.linear2 = torch.nn.Linear(128, 10)

    def forward(self, input):
        res = self.conv1(input)
        res = self.relu1(res)
        res = self.conv2(res)
        res = self.relu2(res)
        res = self.pool(res)

        res = res.view(-1, 14*14*64)
        res = self.linear1(res)
        res = self.relu3(res)
        res = self.dropout(res)
        res = self.linear2(res)
        res = torch.nn.functional.log_softmax(res, dim=1)
        return res

# class Model(torch.nn.Module):
    
#     def __init__(self):
#         super(Model, self).__init__()
#         self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1),
#                                          torch.nn.ReLU(),
#                                          torch.nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
#                                          torch.nn.ReLU(),
#                                          torch.nn.MaxPool2d(stride=2,kernel_size=2))
#         self.dense = torch.nn.Sequential(torch.nn.Linear(14*14*64,128),
#                                          torch.nn.ReLU(),
#                                          torch.nn.Dropout(p=0.5),
#                                          torch.nn.Linear(128, 10))
#     def forward(self, x):
#         x = self.conv1(x)
#         #x = self.conv2(x)
#         x = x.view(-1, 14*14*64)
#         x = self.dense(x)
#         return x

# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(9216, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         output = F.log_softmax(x, dim=1)
#         return output

#----------------------------##
# for epoch in range(n_epochs):
#     running_loss = 0.0
#     running_correct = 0
#     print("Epoch {}/{}".format(epoch, n_epochs))
#     print("-"*10)
#     for idx, data in enumerate(data_loader_train):
#         X_train, y_train = data
#         X_train, y_train = Variable(X_train), Variable(y_train)
#         outputs = model(X_train)
#         _,pred = torch.max(outputs.data, 1)
#         optimizer.zero_grad()
#         loss = cost(outputs, y_train)
        
#         loss.backward()
#         optimizer.step()
#         running_loss += loss
#         running_correct += torch.sum(pred == y_train.data)
#         print(idx, "loss", loss.data, "correct", torch.sum(pred == y_train.data))
#     testing_correct = 0
#     for data in data_loader_test:
#         X_test, y_test = data
#         X_test, y_test = Variable(X_test), Variable(y_test)
#         outputs = model(X_test)
#         _, pred = torch.max(outputs.data, 1)
#         testing_correct += torch.sum(pred == y_test.data)
#     print("Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(running_loss/len(data_train),
#                                                                                       100*running_correct/len(data_train),
#                                                                                       100*testing_correct/len(data_test)))
# torch.save(model.state_dict(), "model_parameter.pkl")

def train(epoch, optimizer, data_loader_train, model, device):
    model.train()
    total_loss = 0
    for idx, batch in enumerate(data_loader_train):
        input, target = batch
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input)
        _, pred = torch.max(output, 1)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if idx % 10 == 0:
            print(idx, "loss", loss.item(), "correct", torch.sum(pred == target).item())
    print(epoch,': ',total_loss / len(data_loader_train))
    # torch.save(model.state_dict(), "model_parameter.pkl")

def test(data_loader_test, model, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for input, target in data_loader_test:
            input, target = input.to(device), target.to(device)
            output = model(input)
            total_loss += F.cross_entropy(output, target).item()
            _, pred = torch.max(output, 1)
            correct += torch.sum(pred == target).item()
    print("test loss", total_loss / len(data_loader_test), "correct", correct / len(data_loader_test))

# def train2(model, device, train_loader, optimizer, epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % 10 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.1307], [0.3081])])
    data_train = datasets.MNIST(root = "./data/",
                                transform=transform,
                                train = True,
                                download = True)
    data_test = datasets.MNIST(root="./data/",
                            transform = transform,
                            train = False)
    data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                    batch_size = 64,
                                                    shuffle = True,
                                                    num_workers=2)
    data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                                batch_size = 64,
                                                shuffle = True,
                                                    num_workers=2)
    model = Model()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    n_epochs = 5
    for i in range(n_epochs):
        train(i, optimizer, data_loader_train, model, device)
        test(data_loader_test, model, device)
        # train2(model, "cpu", data_loader_train, optimizer, 1)

if __name__ == '__main__':
    main()