import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

REBUILD_DATA = False  # set to true to recreate data


class LebarnVSTate():
    IMG_SIZE = 50
    LEBARN = "/Users/mali/dev/deep-learning-pytorch/src/lebarn"
    TATUM = "/Users/mali/dev/deep-learning-pytorch/src/tate"
    TESTING = "/Users/mali/dev/deep-learning-pytorch/src/testing"
    LABELS = {LEBARN: 0, TATUM: 1}
    training_data = []

    lebarncount = 0
    tatecount = 0

    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                if "jpg" in f:
                    try:
                        path = os.path.join(label, f)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                        # do something like print(np.eye(2)[1]), just makes one_hot
                        self.training_data.append(
                            [np.array(img), np.eye(2)[self.LABELS[label]]])
                        # print(np.eye(2)[self.LABELS[label]])

                        if label == self.LEBARN:
                            self.lebarncount += 1
                        elif label == self.TATUM:
                            self.tatecount += 1

                    except Exception as e:
                        pass
                        #print(label, f, str(e))

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print('Lebarn:', lebarnVtate.lebarncount)
        print('Tatum:', lebarnVtate.tatecount)


if REBUILD_DATA:
    lebarnVtate = LebarnVSTate()
    lebarnVtate.make_training_data()

training_data = np.load("./src/training_data.npy", allow_pickle=True)
print(len(training_data))


# Now we can split our training data into X and y, as well as convert it to a tensor:
X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])
print("Hello")
plt.imshow(X[0], cmap="gray")
plt.show()
print("END")


class Net(nn.Module):
    def __init__(self):
        super().__init__()  # just run the init of parent class (nn.Module)
        # input is 1 image, 32 output channels, 5x5 kernel / window
        self.conv1 = nn.Conv2d(1, 32, 5)
        # input is 32, bc the first layer output 32. Then we say the output will be 64 channels, 5x5 kernel / window
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50, 50).view(-1, 1, 50, 50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)  # flattening.
        # 512 in, 2 out bc we're doing 2 classes (lebarn vs tate).
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        # max pooling over 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        # .view is reshape ... this flattens X before
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # bc this is our output layer. No activation here.
        return F.softmax(x, dim=1)


net = Net()
print(net)
