from torch import nn
import torch.nn.functional as F
import torch    
import os


class VGG(nn.Module):

  def __init__(self, classes=100):
    super(VGG, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,bias=False),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1,bias=False),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1,bias=False),
      nn.ReLU(inplace=True),
      nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1,bias=False),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,bias=False),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
    )
    
    self.fc = nn.Linear(256, classes,bias=False)
    self.head_var = 'fc'

  def forward(self, x):
    x = self.features(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)
    return x


def VggNet(num_out=100, pretrained=False):
    if pretrained:
        raise NotImplementedError
    return VGG(num_out)

vgg = VggNet()

# cfg = {
#     'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
#     'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
#     'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
#     'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
# }

# class VGG(nn.Module):

#     def __init__(self, features, num_class=100):
#         super().__init__()
#         self.features = features

#         self.classifier = nn.Sequential(
#             nn.Linear(512, 4096,bias=False),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Dropout()
#         )
#         self.fc = nn.Linear(4096, num_class, bias=False)
#         self.head_var = 'fc'

#     def forward(self, x):
#         output = self.features(x)
#         output = output.view(output.size()[0], -1)
#         output = self.classifier(output)
#         output = self.fc(output)

#         return output

# def make_layers(cfg, batch_norm=False):
#     layers = []

#     input_channel = 3
#     for l in cfg:
#         if l == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             continue

#         layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1,bias=False)]

#         if batch_norm:
#             layers += [nn.BatchNorm2d(l)]

#         layers += [nn.ReLU(inplace=True)]
#         input_channel = l

#     return nn.Sequential(*layers)


# def make_layers1(cfg, batch_norm=False):
#     layers = []

#     input_channel = 3
#     for l in cfg:
#         if l == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             continue

#         layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

#         if batch_norm:
#             layers += [nn.BatchNorm2d(l)]

#         layers += [nn.ReLU(inplace=True)]
#         input_channel = l

#     return nn.Sequential(*layers)


# def vgg11_bn(num_out=100):
#     return VGG(make_layers(cfg['A'], batch_norm=False),num_out)

# def vgg11_bn1(num_out=100):
#     return VGG(make_layers1(cfg['A'], batch_norm=False),num_out)

# def vgg13_bn():
#     return VGG(make_layers(cfg['B'], batch_norm=True))

# def vgg16_bn():
#     return VGG(make_layers(cfg['D'], batch_norm=True))

# def vgg19_bn():
#     return VGG(make_layers(cfg['E'], batch_norm=True))

# def vgg11(num_out=100):
#     return VGG(make_layers(cfg['A'], batch_norm=True),num_out)


# def VggNet(num_out=100, pretrained=False):
#     if pretrained:
#         raise NotImplementedError
#     return vgg11_bn1(num_out)

# def vggnet(num_out=100, pretrained=False):
#     if pretrained:
#         raise NotImplementedError
#     return vgg11(num_out)
