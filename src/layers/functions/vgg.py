import torch
from torchvision.models import vgg19
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn.functional as F

"""
VGG is very helpful in this task, we are passing our label and prediction images
through the VGG Model, which will evaluate the features of both at 
"""

def get_vgg_extractor(device):
    # get vgg model
    vgg = vgg19(pretrained=True)

    # which layer we want to extract features from
    return_nodes = {'features.16': 'vgg_features'}  # relu4_1
    extractor = create_feature_extractor(vgg, return_nodes=return_nodes)
    extractor.eval()

    for param in extractor.parameters():
        # Ignore the derivative of loss wrt the model weights
        # We only want the input grad
        param.requires_grad = False
    extractor.to(device)
    return extractor

vgg_extractor = get_vgg_extractor('cuda')

def vgg_loss(pred, label, vgg_extractor):
    # Ensure input is float and requires grad
    device = next(vgg_extractor.parameters()).device
    pred = pred.to(device).detach().clone().requires_grad_(True)
    label = label.to(device)

    # Normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    pred_norm = (pred - mean) / std
    label_norm = (label - mean) / std

    # Extract features
    pred_features = vgg_extractor(pred_norm)['vgg_features']
    with torch.no_grad():
        label_features = vgg_extractor(label_norm)['vgg_features']

    # Compute perceptual loss
    loss = F.mse_loss(pred_features, label_features)

    # Backprop to get dL/dpred
    loss.backward()
    grad = pred.grad.detach()

    return loss.detach(), grad

