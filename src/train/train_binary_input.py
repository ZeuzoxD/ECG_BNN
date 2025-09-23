import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os

from utils.model_binary_input import ECG_XNOR_FULL_BIN_BinaryInput
from utils.dataset_binary import LoaderBinaryInput 
from utils.engine import train, test_step
from utils.save_model import save_model

class BinarizeF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input.masked_fill_(input.ge(1) | input.le(-1), 0)
        return grad_input

class BinaryConv1d_baw(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, input):

        a = input
        w = self.weight
        ba = BinarizeF.apply(a)
        bw = BinarizeF.apply(w)
        
        out = torch.nn.functional.conv1d(input=ba, weight=bw, bias=None, stride=self.stride,
                       padding=self.padding, dilation=self.dilation, groups=self.groups)
        return out

class Bn_bin_conv_pool_block_baw(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, padding_value, pool_size, pool_stride):
        super().__init__()
        self.pad = nn.ConstantPad1d(padding=padding, value=padding_value)

        self.conv = BinaryConv1d_baw(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0, bias=False)
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)

        self.prelu = nn.PReLU()

        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, I):
        I = self.pad(I)
        I = self.conv(I)
        I = self.pool(I)
        I = self.prelu(I)
        I = self.bn(I)
        return I

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    classes_num = 5
    test_size = 0.2
    batch_size = 512
    lr = 0.01
    seed 169

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print("Loading data with binary input preprocessing")
    loader = LoaderBinaryInput(batch_size=batch_size, classes_num=classes_num, device=device, test_size=test_size)
    labels, train_loader, test_loader = loader.loader()
    print("Data split summary")
    loader.plot_train_test_splits()

    kernel_size, padding, pool_size = 7, 5, 7
    padding_value = -1

    A = [[1, 8, kernel_size, 2, padding, padding_value, poolsize, 2],
         [8, 16, kernel_size, 1, padding, padding_value, poolsize, 2],
         [16, 32, kernel_size, 1, padding, padding_value, poolsize, 2],
         [32, 32, kernel_size, 1, padding, padding_value, poolsize, 2],
         [32, 64, kernel_size, 1, padding, padding_value, poolsize, 2],
         [64, classes_num, kernel_size, 1, padding, padding_value, poolsize, 2],
    ]

    model = ECG_XNOR_Full_Bin_BinaryInput(block1=A[0], block2=A[1], block3=A[2], block4=A[3], block5=A[4], block6=A[5], block7=None, device=device).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_epochs = 1000
    print(f"Training for {num_epochs} epoches")
    print("-" * 60)

    def train_lp_model():
        best_test_acc = 0.0
        best_train_acc = 0.0

        for epoch in range(num_epochs):
            model.train()
            train_loss, train_acc = 0, 0
            correct, total = 0, 0

            for X, y in train_loader:
                X, y = X.to(device), y.to(device)

                optimizer.zero_grad()
                y_pred = model(X)
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(y_pred.data, dim=1)
                total += len(y)
                correct += (predicted == y).sum().cpu().item()

            train_loss = train_loss / len(train_loader)
            train_acc = correct / total

            model.eval()
            test_loss, test_acc = 0, 0
            correct, total = 0, 0

            with torch.no_grad():
                for X, y in test_loader:
                    X, y = X.to(device), y.to(device)
                    y_pred = model(X)
                    loss = loss_fn(y_pred, y)

                    test_loss += loss.item()
                    _, predicted = torch.max(y_pred.data, dim=1)
                    total += len(y)
                    correct += (predicted == y).sum().cpu().item()

            test_loss = test_loss / len(test_loader)
            test_acc = correct / total

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                if test_acc >= 0.9
                    acc_str = f"{test_acc*100:.2f}%"
                    model_name = f"LP_ECG_Net_for_5_{acc_str}.pth"

                    os.makedirs("models_LP", exist_ok=True)
                    torch.save(model.state_dict(), f"models_LP/{model_name}")
                    torch.save(model, f"model_LP/full_{model_name}")
                    print(f"Saved LP model: {model_name} (acc: {acc_str})")

            if train_acc > best_train_acc:
                best_train_acc = train_acc

            print(f"Epoch {epoch:4d}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}, train_loss={train_loss:.4f}, test_loss={test_loss:.4f}")
        
        print("="*60)
        print(f"Best test accuracy: {best_test_acc*100:.2f}%")
        print(f"Best train accuracy: {best_train_acc*100:.2f}%")

    train_lp_model()

if __name__ == "__main__":
    main()
