import torch.nn as nn
import torchvision.models as models

def efficientnet_b0(num_classes=10, is_pretrained=False):
    if is_pretrained:
        model = models.efficientnet_b0(pretrained=True)
    else:
        model = models.efficientnet_b0(pretrained=False)
    
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
