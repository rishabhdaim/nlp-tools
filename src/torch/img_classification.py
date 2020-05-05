import torch
import torchvision
from torch import nn
import numpy as np
import torch.optim as optim
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

# This is used to transform the images to Tensor and normalize it
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
training = torchvision.datasets.MNIST(root='./target/data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(training, batch_size=4, shuffle=True, num_workers=2)
testing = torchvision.datasets.MNIST(root='./target/data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testing, batch_size=4, shuffle=False, num_workers=2)
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# create an iterator for train_loader
# get random training images
data_iterator = iter(train_loader)
images, labels = data_iterator.next()

# plot 4 images to visualize the data
rows = 2
columns = 2
fig = plt.figure()
for i in range(4):
    fig.add_subplot(rows, columns, i + 1)
    plt.title(classes[labels[i]])
    img = images[i] / 2 + 0.5  # this is for unnormalize the image
    img = torchvision.transforms.ToPILImage()(img)
    plt.imshow(img)
plt.show()


# flatten the tensor into
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# sequential based model
seq_model = nn.Sequential(
    nn.Conv2d(1, 10, kernel_size=5),
    nn.MaxPool2d(2),
    nn.ReLU(),
    nn.Dropout2d(),
    nn.Conv2d(10, 20, kernel_size=5),
    nn.MaxPool2d(2),
    nn.ReLU(),
    Flatten(),
    nn.Linear(320, 50),
    nn.ReLU(),
    nn.Linear(50, 10),
    nn.Softmax(),
)

net = seq_model
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
    # set the running loss at each epoch to zero
    running_loss = 0.0
    # we will enumerate the train loader with starting index of 0
    # for each iteration (i) and the data (tuple of input and labels)
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        # clear the gradient
        optimizer.zero_grad()
        # feed the input and acquire the output from network
        outputs = net(inputs)
        # calculating the predicted and the expected loss
        loss = criterion(outputs, labels)
        # compute the gradient
        loss.backward()
        # update the parameters
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 1000 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0

# make an iterator from test_loader
# Get a batch of training images
test_iterator = iter(test_loader)
images, labels = test_iterator.next()

results = net(images)
_, predicted = torch.max(results, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

fig2 = plt.figure()
for i in range(4):
    fig2.add_subplot(rows, columns, i + 1)
    plt.title('truth ' + classes[labels[i]] + ': predict ' + classes[predicted[i]])
    img = images[i] / 2 + 0.5  # this is to unnormalize the image
    img = torchvision.transforms.ToPILImage()(img)
    plt.imshow(img)
plt.show()
