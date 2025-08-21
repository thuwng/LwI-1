from torch import nn
import torch.nn.functional as F
import torch    

class LeNetArch(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding='same', bias=False)
        self.maxpool = nn.MaxPool2d(2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, 5, padding='same', bias=False)
        self.fc = nn.Linear(4096, out_features=num_classes, bias=False)
        self.head_var = 'fc'

    def forward(self, x):
        output = F.relu(self.conv1(x))
        output = self.maxpool(output)
        output = F.relu(self.conv2(output))
        output = self.maxpool(output)
        output = output.view(output.shape[0], -1)
        output = self.fc(output)
        return output

def LeNet(pretrained=False, **kwargs):
    if pretrained:
        raise NotImplementedError
    return LeNetArch(**kwargs)

