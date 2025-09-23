import torch
import torch.nn as nn
from .OP import Bn_bin_conv_pool_block_baw, BinarizeF

class ECG_XNOR_Full_Bin_BinaryInput(nn.Module):
    '''
    Full Binary Model with Binary Inputs and Binary Weights
    '''
    def __init__(self, block1, block2, block3, block4, block5, block6, block7, device):
        super(ECG_XNOR_Full_Bin_BinaryInput, self).__init__()
        self.name = 'Full_Bin_BinaryInput_ECG'
        self.device = device

        self.block1 = Bn_bin_conv_pool_block_baw(*block1)
        self.block2 = Bn_bin_conv_pool_block_baw(*block2)
        self.block3 = Bn_bin_conv_pool_block_baw(*block3)
        self.block4 = Bn_bin_conv_pool_block_baw(*block4)

        self.is_block5 = False
        self.is_block6 = False
        self.is_block7 = False

        if block5 is not None:
            self.block5 = Bn_bin_conv_pool_block_baw(*block5)
            self.is_block5 = True
        if block6 is not None:
            self.block6 = Bn_bin_conv_pool_block_baw(*block6)
            self.is_block6 = True
        if block7 is not None:
            self.block7 = Bn_bin_conv_pool_block_baw(*block7)
            self.is_block7 = True

    def forward(self, batch_data):
        # Input preprocessing - Standardize
        batch_data = batch_data.clone().detach().requires_grad_(True).to(self.device)
        # Binarize inputs to {+1, -1}
        batch_data = BinarizeF.apply(batch_data)

        batch_data = self.block1(batch_data)
        batch_data = self.block2(batch_data)
        batch_data = self.block3(batch_data)
        batch_data = self.block4(batch_data)

        if self.is_block5:
            batch_data = self.block5(batch_data)
        if self.is_block6:
            batch_data = self.block6(batch_data)
        if self.is_block7:
            batch_data = self.block7(batch_data)

        batch_data = self.dropout0(batch_data)
        batch_data = batch_data.mean(dim=2) #GSP

        return batch_data

