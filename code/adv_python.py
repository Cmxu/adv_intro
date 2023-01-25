import torch
import torch.nn.functional as F
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from data_utils import toDeviceDataLoader, load_cifar, to_device
from model_utils import VGG
from utils import asr, accuracy, show_attack, project_lp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# In[2]:

dataset_root = r"../data/"
cifar10_train, cifar10_val, cifar10_test = load_cifar(dataset_root,  download = False)
train_loader, val_loader, test_loader = toDeviceDataLoader(cifar10_train, cifar10_val, cifar10_test, device = device)


# ### Pretrained model VGG

# In[ ]:


mdl = to_device(VGG('VGG16'), device)
mdl.load_state_dict(torch.load('../models/torch_cifar_vgg.pth',map_location=torch.device('cpu')))
mdl = mdl.eval()


# ### FGSM

# In[ ]:


def fgsm(x, y, k, norm = np.inf, xi = 1e-1, step_size = 1e-1, device = device):
    
    #x, y = x.to(device), y.to(device)
    #x.requiresGrad = True
    x=torch.tensor(x,requires_grad = True)

    # calculate the loss
    output = mdl(x)
    mdl.zero_grad()
    loss = F.cross_entropy(output, y)

    # calculate the gradients wrt the input
    loss.backward(retain_graph=True)

    # calculate the sign of the gradients
    x_grad = torch.sign(x.grad)                
    
    
    # calculate the adversarial example
    x_adv = x + xi * x_grad ## using formula from paper

    #x_adv=torch.clamp(x_adv, 0, 1)

    # project the adversarial example back onto the l_norm ball
    #x_adv = torch.tensor(projected_gradient_descent(x_adv.cpu().numpy(), x.cpu().numpy(), k, norm), device=device)

    return x_adv


# ### PGD

# In[ ]:


def pgd(x, y, k, norm = np.inf, xi = 1e-1, step_size = 1e-2, epochs = 10, device = torch.device('cuda:0')):
    
    loss = nn.CrossEntropyLoss()
        
    x=torch.tensor(x,requires_grad = True)
        
    for i in range(epochs) :    
        x.requires_grad = True
        output = mdl(x)

        mdl.zero_grad()
        cost = loss(output, y)
        cost.backward()
        x_grad = torch.sign(x.grad) 
        adv_images = x + xi*x_grad
        eta = torch.clamp(adv_images - x, min=-xi, max=xi)
        images = torch.clamp(x + eta, min=0, max=1).detach_()
            
    return images


# In[ ]:


#Initial Test on Small Batch
x, y = next(iter(test_loader))
print(y.shape)
print('Base Accuracy {}'.format(accuracy(mdl(x), y))) # Varies with batch, mine ~ 0.875
print('FGSM Accuracy: {}'.format(accuracy(mdl(x + fgsm(x, y, mdl)), y))) # Varies with batch, mine ~ 0
print('PGD Accuracy: {}'.format(accuracy(mdl(x + pgd(x, y, mdl)), y))) # Varies with batch, mine ~ 0


# In[ ]:


v = fgsm(x, y, mdl)
show_attack(x, v, mdl)


# In[ ]:


#Test on Entire Dataset (this will take a few minutes depending on how many epochs of pgd you have)
print('Base Accuracy: {}'.format(1 - asr(test_loader, mdl))) # ~ 0.9171
print('FGSM Accuracy: {}'.format(1 - asr(test_loader, mdl, fgsm))) # ~ 0.0882
print('PGD Accuracy: {}'.format(1 - asr(test_loader, mdl, pgd))) # ~ 0.0001


# In[ ]:




