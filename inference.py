import torch
import numpy as np
import argparse
import random
from pathlib import Path

def pad_1d(x, pad, val=1):
    if pad == 0: return x
    B, C, L = x.shape
    left = torch.full((B, C, pad), val, dtype=x.dtype, device=x.device)
    right = torch.full((B, C, pad), val, dtype=x.dtype, device=x.device)
    return torch.cat([left, x, right], dim=2)

def binary_conv1d(x, w, stride=1, padding=0):
    B, Cin, L = x.shape
    Cout, _, K = w.shape
    if padding > 0:
        x = pad_1d(x, padding, val=1)
    x_unf = x.unfold(2, K, stride) # [B, Cin, Lout, K]
    x_unf = x_unf.permute(0, 2, 1, 3).unsqueeze(2)
    w_exp = w.unsqueeze(0).unsqueeze(0)
    xnor = (x_unf == w_exp).int()
    popcount = xnor.sum(dim=(3, 4))
    out = (2 * popcount - Cin * K).to(torch.int32)
    return out.permute(0, 2, 1)

def maxpool_int(x, k=7, s=2):
    B, C, L = x.shape
    if L < k:
        pad = torch.full((B, C, k - L), -2**30, dtype=x.dtype, device=x.device)
        x = torch.cat([x, pad], dim=2)
    return x.unfold(2, k, s).amax(dim=3)

def compute_thresholds(bn_w, bn_b, bn_m, bn_v, prelu_w, scale=2**12, eps=1e-5):
    a = prelu_w.expand_as(bn_m)
    s = torch.sqrt(bn_v + eps)
    gamma = bn_w.clone()
    gamma[gamma.abs() < 1e-6] = torch.sign(gamma[gamma.abs() < 1e-6]) * 1e-6
    delta_plus = bn_m - bn_b * s / gamma
    bn0 = gamma * (- bn_m) / s + bn_b
    delta_minus = torch.where(
        a.abs() >= 1e-12,
        delta_plus / a,
        torch.where(bn0 >= 0.0, torch.full_like(delta_plus, -1e9), torch.full_like(delta_plus, 1e9))
    )
    dp_int = (delta_plus * scale).round().int()
    dm_int = (delta_minus * scale).round().int()
    a_sign = torch.sign(a).int()
    return dp_int, dm_int, a_sign, scale

def threshold_activation(x, params, pool_k=7, pool_s=2):
    dp, dm, a_sign, scale = params

    x_pool = maxpool_int(x.to(torch.int32), pool_k, pool_s)
    x_scaled = x_pool.to(torch.int64) * int(scale)

    B, C, L = x_pool.shape
    dp = dp.view(1, C, 1).to(torch.int64).expand(B, C, L)
    dm = dm.view(1, C, 1).to(torch.int64).expand(B, C, L)
    a_s = a_sign.view(1, C, 1).expand(B, C, L)

    pos_mask = x_pool >= 0
    cond_pos = pos_mask & (x_scaled >= dp)

    neg_mask = ~pos_mask
    cond_neg = torch.where(
        a_s >= 0,
        neg_mask & (x_scaled >= dm),
        neg_mask & (x_scaled <= dm)
    )

    return torch.where(cond_pos | cond_neg,
                      torch.ones_like(x_pool, dtype=torch.int32),
                      -torch.ones_like(x_pool, dtype=torch.int32))

class ECG_BNN_MODEL:
    def __init__(self, pth_path, num_classes=5, labels=['N', 'S', 'V', 'F', 'Q'], scale=2**6):
        self.scale = scale
        self.LABELS = labels
        self.num_classes = num_classes
        self.blocks = [
            [1, 8, 7, 2, 5, 7, 2], 
            [8, 16, 7, 1, 5, 7, 2],
            [16, 32, 7, 1, 5, 7, 2],
            [32, 32, 7, 1, 5, 7, 2],
            [32, 64, 7, 1, 5, 7, 2],
            [64, num_classes, 7, 1, 5, 7, 2],
        ]
        self._load_params(pth_path)

    def _load_params(self, pth_path):
        sd = torch.load(pth_path, map_location='cpu')
        self.weights = [torch.sign(sd[f'block{i+1}.conv.weight']) for i in range(6)]
        self.thresholds = []
        for i in range(6):
            bn = f'block{i+1}'
            params = compute_thresholds(
                sd[f'{bn}.bn.weight'],
                sd[f'{bn}.bn.bias'],
                sd[f'{bn}.bn.running_mean'],
                sd[f'{bn}.bn.running_var'],
                sd[f'{bn}.prelu.weight'],
                self.scale
            )
            self.thresholds.append(params)

    def preprocess(self, sig):
        sig = torch.from_numpy(sig).float().unsqueeze(0).unsqueeze(0)
        sig = (sig - sig.mean(dim=-1, keepdim=True)) / (sig.std(dim=-1, keepdim=True) + 1e-8)
        return torch.sign(sig) # Binarize to {-1, +1}

    def forward(self, sig):
        x = self.preprocess(sig)
        
        for i in range(6):
            cfg = self.blocks[i]
            x = binary_conv1d(x, self.weights[i], stride=cfg[3], padding=cfg[4])
            x = threshold_activation(x, self.thresholds[i], cfg[5], cfg[6])

        # Global Sum Pooling (GSP) - sum over temporal dimension
        # Output shape: [B, num_classes]
        logits = x.sum(dim=2)
        pred_idx = torch.argmax(logits, dim=1).item()
        return pred_idx, logits

    def predict(self, sig):
        idx, logits = self.forward(sig)
        return idx, self.LABELS[idx], logits

def load_ecg(path, classes_num=5):
    if classes_num == 17:
        import scipy.io as scio
        data_train = scio.loadmat(path)
        data_arr = data_train.get('val')
        return np.asarray(data_arr).squeeze()
    else:
        arr = np.load(path, allow_pickle=True)
        if arr.dtype == object:
            arr = np.array(arr.tolist())
        return np.asarray(arr).squeeze()

def sample_files(root, n=50, classes_num=5):
    files = []
    file_ext = '*.npy' if classes_num == 5 else '*.mat'
    for d in sorted(Path(root).glob('*')):
        if d.is_dir():
            files.extend(d.glob(file_ext))
    random.seed(33)
    return random.sample(files, min(n, len(files)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=int, choices=[5, 17], default=5)
    parser.add_argument('--n', type=int, default=100)
    args = parser.parse_args()

    if args.c == 17:
        pth_path = "models/LP_ECG_Net_for_5_90.50%.pth"
        data_path = "ECG_Dataset/ECG-17"
        labels = ['NSR', 'APB', 'AFL', 'AFIB', 'SVTA', 'WPW', 'PVC', 'Bigeminy',
                  'Trigeminy', 'VT', 'IVR', 'VFL', 'Fusion', 'LBBBB', 'RBBBB', 'SDHB', 'PR']
    else:
        pth_path = "models/LP_ECG_Net_for_5_90.37%.pth"
        data_path = "ECG_Dataset/ECG-5"
        labels = ['N', 'S', 'V', 'F', 'Q']

    model = ECG_BNN_MODEL(pth_path=pth_path, num_classes=args.c, labels=labels)
    files = sample_files(data_path, args.n, args.c)
    correct = 0

    for i, f in enumerate(files, 1):
        sig = load_ecg(f, args.c)
        gt = int(Path(f).parent.name.split()[0]) - 1
        idx, lbl, logits = model.predict(sig)
        if idx == gt:
            correct += 1
        print(f"{i:3d}: GT={labels[gt]} Pred={lbl} {'✓' if idx == gt else '✗'} | Logits: {logits.squeeze().tolist()}")

    print(f"\nAccuracy: {correct/len(files)*100:.2f}% ({correct}/{len(files)})")

if __name__ == "__main__":
    main()
