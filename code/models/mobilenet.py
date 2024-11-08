import torch.nn as nn
import torchvision.models as models

def mobilenet(num_classes=10, is_pretrained=False):
    if is_pretrained:
        model = models.mobilenet_v2(pretrained=True)
    else:
        model = models.mobilenet_v2(pretrained=False)
    
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
