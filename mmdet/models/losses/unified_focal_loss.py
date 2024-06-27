import torch
import torch.nn as nn

from mmdet.registry import MODELS
from .utils import weight_reduce_loss

#https://github.com/oikosohn/compound-loss-pytorch/blob/main/unified_focal_loss_pytorch.py

def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5 : return [2,3,4]

    # Two dimensional
    elif len(shape) == 4 : return [2,3]
    
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')

class SymmetricFocalLoss(nn.Module):
    """
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
    epsilon : float, optional
        clip values to prevent division by zero error
    """
    def __init__(self, delta=0.7, gamma=2., epsilon=1e-03):
        super(SymmetricFocalLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):

        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        cross_entropy = -y_true * torch.log(y_pred)

        # Calculate losses separately for each class
        back_ce = torch.pow(1 - y_pred[:,0,:,:], self.gamma) * cross_entropy[:,0,:,:]
        back_ce =  (1 - self.delta) * back_ce

        fore_ce = torch.pow(1 - y_pred[:,1,:,:], self.gamma) * cross_entropy[:,1,:,:]
        fore_ce = self.delta * fore_ce

        loss = torch.mean(torch.sum(torch.stack([back_ce, fore_ce], axis=-1), axis=-1))

        return loss


class SymmetricFocalTverskyLoss(nn.Module):
    """This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    smooth : float, optional
        smooithing constant to prevent division by 0 errors, by default 0.000001
    epsilon : float, optional
        clip values to prevent division by zero error
    """
    def __init__(self, delta=0.7, gamma=0.75, epsilon=1e-03):
        super(SymmetricFocalTverskyLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):

        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)

        axis = identify_axis(y_true.size())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
        tp = torch.sum(y_true * y_pred, axis=axis)
        fn = torch.sum(y_true * (1-y_pred), axis=axis)
        fp = torch.sum((1-y_true) * y_pred, axis=axis)
        dice_class = (tp + self.epsilon)/(tp + self.delta*fn + (1-self.delta)*fp + self.epsilon)
        
        # Calculate losses separately for each class, enhancing both classes
        back_dice = (1-dice_class[:,0]) * torch.pow(1-dice_class[:,0], -self.gamma)
        fore_dice = (1-dice_class[:,1]) * torch.pow(1-dice_class[:,1], -self.gamma) 

        # Average class scores
        loss = torch.mean(torch.stack([back_dice,fore_dice], axis=-1))
        return loss




@MODELS.register_module()
class UnifiedFocalLoss(nn.Module):
    def __init__(self,
                 weight=0.5, 
                 delta=0.6, 
                 gamma=0.5,
                 eps=1e-3,
                 ):
        """
        """

        super(UnifiedFocalLoss, self).__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
        self.eps = eps
        self.flt_func = SymmetricFocalTverskyLoss(delta=self.delta, gamma=self.gamma)
        self.fl_func = SymmetricFocalLoss(delta=self.delta, gamma=self.gamma)

    def forward(self,
                y_pred,
                y_true,
                weight=None,
                reduction_override=None,
                avg_factor=None):
        """
        """

        num_rois = y_pred.size()[0]
        inds = torch.arange(0, num_rois, dtype=torch.long, device=y_pred.device)
        y_pred_slice = y_pred[inds, weight].squeeze(1)
        y_pred_slice = y_pred_slice.sigmoid()

        y_pred_slice = torch.stack([y_pred_slice, 1-y_pred_slice],dim=1)
        y_true = torch.stack([y_true, 1-y_true],dim=1)

        symmetric_ftl = self.flt_func(y_pred_slice, y_true)
        symmetric_fl = self.fl_func(y_pred_slice, y_true)
        if self.weight is not None:
            loss = (self.weight * symmetric_ftl) + ((1-self.weight) * symmetric_fl)  
            # print(loss)
            return loss
        else:
            return symmetric_ftl + symmetric_fl
