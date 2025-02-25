import torch
import torch.nn as nn
import torch.nn.functional as F

'''
from m3_learning.nn.Regularization
'''

class ContrastiveLoss(nn.Module):
    """Builds a contrastive loss function based on the cosine similarity between the latent vectors.

    $$L = \frac{cof1}{2N} \sum\limits_{i=1}^{N} \left[\left(\sum\limits_{j=1, j\neq i}^{N} \frac{latent_i \cdot latent_j}{\left\lVert latent_i \right\rVert \left\lVert latent_j \right\rVert}\right] - 1\right)$$

    Args:
        nn (nn.Module): Pytorch module
    """

    def __init__(self, cof1=1e-2):
        """Initializes the contrastive loss regularization

        Args:
            cof1 (float, optional): Regularization hyperparameter. Defaults to 1e-2.
        """
        super(ContrastiveLoss, self).__init__()
        self.cof1 = cof1

    def forward(self, latent):
        """Forward pass of the contrastive loss regularization

        Args:
            latent (Tensor): Activations of layer to apply the loss metric

        Returns:
            Tensor: Loss value
        """

        loss = 0
        beyond_0 = torch.where(torch.sum(latent, axis=1) != 0)[0]
        new_latent = latent[beyond_0]
        for i in beyond_0:
            loss += sum(F.cosine_similarity(
                latent[i].unsqueeze(0), new_latent)) - 1

        loss = self.cof1 * loss / (2.0 * latent.shape[0])

        return loss


class DivergenceLoss(nn.Module):
    def __init__(self, batch_size, cof1=1e-2):
        """Divergence regularization for the latent space.

        This regularization tries to make the latent vectors sparse and different from each other.

        Args:
            batch_size (Int): The batch size of each update
            cof1 (Tensor, optional): Hyperparameter. Defaults to 1e-2.
        """
        super(DivergenceLoss, self).__init__()
        self.batch_size = batch_size
        self.cof1 = cof1

    def forward(self, latent):
        """Forward pass of the divergence regularization

        Args:
            latent (Tensor): Activations of layer to apply the loss metric

        Returns:
            Tensor: Loss value
        """

        loss = 0

        for i in range(self.batch_size):
            no_zero = torch.where(latent[i].squeeze() != 0)[0]
            single = latent[i][no_zero]
            loss += self.cof1 * \
                torch.sum(abs(single.reshape(-1, 1) - single)) / 2.0

        loss = loss / self.batch_size

        return loss
    

'''
from Xinqiao's fork of m3_learning.nn.Regularization
'''
class Weighted_LN_loss(nn.Module):
    def __init__(self, ln_parm=2, coef=0.01, channels=1, ):
        """_summary_

        Args:
            ln_parm (int, optional): _description_. Defaults to 2.
            coef (float, optional): _description_. Defaults to 0.01.
            channels (int, optional): _description_. Defaults to 1.
        """        
        super(Weighted_LN_loss, self).__init__()        
        self.ln_parm = ln_parm
        self.coef = coef
        self.channels = channels
        
    def forward(self,x):
        weights = torch.linspace(0,1,self.channels).to(x.device) # penalize each channel using a different coefficient
        loss = x*weights.unsqueeze(0)
        loss = torch.norm(loss, self.ln_parm, dim=1).mean()
        # loss = (x**self.ln_parm).sum(dim=1)**(1/self.ln_parm)
        return loss.mean()*self.coef

class Sparse_Max_Loss(nn.Module): #TODO: break into channel-scaled coef loss and sparse max loss
    def __init__(self,min_threshold=3e-5,coef=0.01,channels=1,ln_parm=2):
        """_summary_

        Args:
            min_threshold (float, optional): if input is less than this value, it will not be penalized. Defaults to 3e-5. should not be too large
            coef (float, optional): scale this loss value. Defaults to 1.
        """        
        super(Sparse_Max_Loss, self).__init__()        
        self.coef = coef
        self.ln_parm=2
        self.threshold = min_threshold
        self.channels=channels
    def forward(self,x):
        ''' x (tensor): shape (batchsize, n). n is the number of channels '''
        # 1 
        # loss = torch.norm(x, self.ln_parm, dim=0).to(x.device) /x.shape[0] # take batchnorm. retain fit channels
        mask = torch.argwhere(x>self.threshold) # threshold
        loss = torch.norm(1-x[mask].sum(dim=1)/self.channels)/x.shape[1]
        return self.coef * loss.mean()