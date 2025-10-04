import torch
import numpy as np
import argparse
import random
from pathlib import Path

def pad_1d(x, pad, val=1):
    if pad == 0: return x
    C, L = x.shape
    left = torch.full((C, pad), val, dtype=x.dtype, device=x.device)
    right = torch.full((C, pad), val, dtype=x.dtype, device=x.device)
    return torch.cat([left, x, right], dim=1)

def binary_conv1d(x, w, stride=1, padding=0):
    Cin, L = x.shape
    Cout, _, K = w.shape

    x = pad_1d(x, padding, val=1)
    x_unf = x.unfold(1, K, stride).permute(1, 0, 2)
    xnor = (x_unf.unsqueeze(1) == w.unsqueeze(0))
    popcount = xnor.sum(dim=(2, 3))
    out = (2 * popcount - Cin * K).T
    return out

def maxpool_int(x, k=7, s=2):
    out = x.unfold(1, k, s).amax(dim=2)
    return out

def compute_thresholds(bn_w, bn_b, bn_m, bn_v, prelu_w, eps=1e-5):
    a = prelu_w.expand_as(bn_m)
    s = torch.sqrt(bn_v + eps)

    gamma = bn_w.clone()
    gamma[gamma.abs() < 1e-6] = torch.sign(gamma[gamma.abs() < 1e-6]) * 1e-6

    delta_plus = bn_m - bn_b * s / gamma
    bn0 = gamma * (-bn_m) / s + bn_b
    delta_minus = torch.where(
        a.abs() >= 1e-12,
        delta_plus / a,
        torch.where(bn0 >= 0.0,
                    torch.full_like(delta_plus, -1e9),
                    torch.full_like(delta_plus, 1e9))
    )
    dp_int = (delta_plus * 20).round().int()
    dm_int = (delta_minus * 20).round().int()
    a_sign = torch.sign(a).int()

    return dp_int, dm_int, a_sign

def threshold_activation(x, params, pool_k=7, pool_s=2):
    dp, dm, a_sign = params

    x_pool = maxpool_int(x, pool_k, pool_s) * 20
    C, L = x_pool.shape

    dp_expanded = dp.view(C, 1).expand(C, L)
    dm_expanded = dm.view(C, 1).expand(C, L)
    a_s_expanded = a_sign.view(C, 1).expand(C, L)

    pos_mask = x_pool >= 0
    cond_pos = pos_mask & (x_pool >= dp_expanded)

    neg_mask = ~pos_mask
    cond_neg = torch.where(
        a_s_expanded >= 0,
        neg_mask & (x_pool >= dm_expanded),
        neg_mask & (x_pool <= dm_expanded)
    )

    out = torch.where(cond_pos | cond_neg,
                      torch.ones_like(x_pool),
                      -torch.ones_like(x_pool))
    return out

class ECG_BNN_MODEL:
    def __init__(self, pth_path, num_classes=5, labels=['N', 'S', 'V', 'F', 'Q']):
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
        sd = torch.load(pth_path, map_location="cpu")
        self.weights = [torch.sign(sd[f'block{i+1}.conv.weight']) for i in range(6)]
        self.thresholds = []
        for i in range(6):
            bn = f'block{i+1}'
            params = compute_thresholds(
                sd[f'{bn}.bn.weight'],
                sd[f'{bn}.bn.bias'],
                sd[f'{bn}.bn.running_mean'],
                sd[f'{bn}.bn.running_var'],
                sd[f'{bn}.bn.prelu.weight'],
            )
            self.thresholds.append(params)

    def preprocess(self, sig):
        sig = torch.from_numpy(sig).float()
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)
        return torch.sign(sig).unsqueeze(0)

    def forward(self, sig):
        x = self.preprocess(sig)
        for i in range(6):
            cfg = self.blocks[i]
            in_ch, out_ch, conv_k, conv_s, conv_p, pool_k, pool_s = cfg
            x = binary_conv1d(x, self.weights[i], stride=conv_s, padding=conv_p)
            x = threshold_activation(x, self.thresholds[i], pool_k, pool_s)

        logits = x.sum(dim=1)
        pred_idx = torch.argmax(logits).item()
        return pred_idx, logits

    def predict(self, sig, debug=False):
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
    random.seed(62)
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
        print(f"{i:3d}: GT={labels[gt]} Pred={lbl} {'✓' if idx == gt else '✗'} | Logits: {logits.tolist()}")
    print(f"\nAccuracy: {correct/len(files)*100:.2f}% ({correct}/{len(files)})")

if __name__ == "__main__":
    main()
