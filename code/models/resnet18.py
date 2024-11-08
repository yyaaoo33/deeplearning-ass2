import torch.nn as nn
import torchvision.models as models

def resnet18(num_classes=10, use_residual=True, is_pretrained=False):
    if is_pretrained:
        model = models.resnet18(pretrained=True)
    else:
        model = models.resnet18(pretrained=False)
    
    if not use_residual:
        model.layer1[0].downsample = None
        model.layer2[0].downsample = None
        model.layer3[0].downsample = None
        model.layer4[0].downsample = None

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
