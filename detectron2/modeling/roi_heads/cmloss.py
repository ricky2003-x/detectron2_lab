import torch
import torch.nn as nn
import sys
from torch import Tensor
from torch.nn.modules.loss import _Loss

class ConfusionMatrixBasedLoss(torch.nn.Module):
    """
    Implements of Confusion Matrix Based Loss Function
    """
    
    def __init__(self, epsilon=10, 
                gamma_Sens = 1,
                 gamma_Spec = 1,
                 gamma_PPV = 1, 
                 gamma_NPV = 1,
                 gamma_Acc = 0,
                 my_name = 'ConfusionMatrixBasedLoss'):
        
        super(ConfusionMatrixBasedLoss, self).__init__()
        
        self.epsilon = epsilon
        # harmonic mean用の重み
        self.gamma_Sens = gamma_Sens
        self.gamma_Spec = gamma_Spec
        self.gamma_PPV = gamma_PPV
        self.gamma_NPV = gamma_NPV
        self.gamma_Acc = gamma_Acc
        self.__name__ = my_name


    def forward(self, pred: Tensor, observed: Tensor) -> Tensor:
        """
        Calculate Confuison Matrix Based Loss value.
        
        Input:
            pred (torch.Tensor): 0 or 1の予測値
            observed (torch.Tensor): 観測値

        Output:
            Loss (torch.Tensor): 損失

        Note:
            predは0～1の確率でないといけない logitではだめ
            obsは0 or 1の

        """
        # check input size
        # pred = pred[:, 0:1, :, :]
        if not (pred.size() == observed.size()):
            sys.exit(f'The dimensions of pred and observed are different.'\
            f'(pred dim: {pred.size()}, observed dim: {observed.size()})')

        # transform to Tensor with [N] shape
        _reshaped_pred = self._transform_to_vec(pred)
        _reshaped_observed = self._transform_to_vec(observed)

        # calculate loss value
        return self._cm_loss_core(_reshaped_pred, _reshaped_observed)


    def _cm_loss_core(self, pred: Tensor, observed: Tensor) -> Tensor:
        """

        Calculate Confuison Matrix Based Loss value.

        Input:
            pred (torch.Tensor): 0 or 1の予測値
            observed (torch.Tensor): 観測値

        Output:
            loss (torch.Tensor): 損失
        """

        _negative_proba = 1 - pred.detach()

        _positive_flag = observed.detach()
        _negative_flag = 1 - observed.detach()

        _correct_proba = pred * _positive_flag + _negative_proba * _negative_flag
        _incorrect_proba = pred * _negative_flag + _negative_proba * _positive_flag

        _d_kn = -1 * _correct_proba + _incorrect_proba

        _penalties = torch.sigmoid( self.epsilon * _d_kn )  # torch.sigmoid(x) = 1/(1+exp(-x))

        # print(f'd_kn: {d_kn}')
        # print(f'penalties: {penalties}')

        _delta_kn_2 = observed.detach()
        _delta_kn_1 = 1 - observed.detach()

        _N_TP_vect = ( 1 - _penalties ) * _delta_kn_2
        _N_TP = _N_TP_vect.sum()

        _N_FP_vect = _penalties * _delta_kn_1
        _N_FP = _N_FP_vect.sum()

        _N_TN_vect = ( 1 - _penalties ) * _delta_kn_1
        _N_TN = _N_TN_vect.sum()

        _N_FN_vect = _penalties * _delta_kn_2
        _N_FN = _N_FN_vect.sum()


        # wtf, we can calculate IoU
        # we want to raize IoU, so, loss mean -IoU
        return -1 * (_N_TP / (_N_TP + _N_FP + _N_FN))
        # end


        # ここで0が起きると困る warnは出すようにしたいな
        _Sens = _N_TP / (_N_TP + _N_FN)
        _Spec = _N_TN / (_N_TN + _N_FP)
        _PPV = _N_TP / (_N_TP + _N_FP)
        _NPV = _N_TN / (_N_TN + _N_FN)
        _Acc = (_N_TP + _N_TN) / (_N_TP + _N_FP + _N_TN + _N_FN)

        # 選択された平均を使いたい
        _S_gamma = self.gamma_Sens + self.gamma_Spec + self.gamma_PPV + self.gamma_NPV + self.gamma_Acc
        _HM = _S_gamma * (1/(self.gamma_Sens/_Sens + self.gamma_Spec/_Spec + self.gamma_PPV/_PPV + self.gamma_NPV/_NPV + self.gamma_Acc/_Acc))

        _loss = -1 * _HM
        # print(f'TN: {_N_TN}, FP: {_N_FP}, FN: {_N_FN}, TP: {_N_TP}, Sens: {_Sens}, Spec: {_Spec}, PPV: {_PPV}, NPV: {_NPV}')
        return _loss

        


    def _transform_to_vec(self, some_dims_tensor: Tensor) -> Tensor:
        """
        Convert a tensor with some dimensions to a vector.

        Input:
            some_dims_tensor (torch.Tensor): とある次元を持つTensor

        Output:
            reshaped_vect (torch.Tensor): Tensor with one dimension
        """
        return some_dims_tensor.reshape(-1)