from torchvision.models import vgg19
from torch import nn
import config

class VGGloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features[:35].eval().to(config.DEVICE)
        for param in self.vgg.parameters:
            param.requires_grad = False
        self.loss = nn.MSELoss()

    def forward(self, input, target):
        input_features = self.vgg(input)
        target_features = self.vgg(target)
        return self.loss(input_features, target_features)