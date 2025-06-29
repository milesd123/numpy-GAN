from vgg import vgg_loss, vgg_extractor
from mse import mse_loss_grad

def loss_grad_generator(pred, label, grad_alpha):
    #loss, grad
    loss_vgg, vgg_grad = vgg_loss(pred, label, vgg_extractor)
    mse_loss, mse_grad = mse_loss_grad(pred, label)
    return mse_loss + loss_vgg, mse_grad + vgg_grad*(grad_alpha)
