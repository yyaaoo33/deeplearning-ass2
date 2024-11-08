import torch.nn as nn
import torchvision.models as models

def vgg16(num_classes=10, is_pretrained=False):
    if is_pretrained:
        model = models.vgg16(pretrained=True)
    else:
        model = models.vgg16(pretrained=False)
    
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model