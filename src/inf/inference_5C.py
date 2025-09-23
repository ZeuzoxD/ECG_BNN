import torch
import torch.nn.functional as F
import numpy as np

import argparse
import os
import random
from pathlib import Path

LABELS = ['N', 'S', 'V', 'F', 'Q']

def preprocess_to_binary(signal, scale_factor=2**10):
    signal = torch.from_numpy(signal).float()
    signal_mean = torch.mean(signal)
    signal_std = torch.std(signal)
    signal_normalized = (signal - signal_mean) / (signal_std + 1e-8)

    signal_binary = torch.sign(signal_normalized)
    return signal_binary.unsqueeze(0).unsqueeze(0)

def pad_1d_binary(x, pad, val=1):
    if pad==0:
        return x
    B, C, L = x.shape
    left = torch.full((B, C, pad), val, dtype=x.dtype, device=x.device)
    right = torch.full((B, C, pad), val, dtype=x.dtype, device=x.device)
    return torch.cat([left, x, right], dim=2)

def xnor_popcount_conv1d(input_binary, weight_binary, stride=1, padding=0):
    B, C_in, L = input_binary.shape
    C_out, _, K = weight_binary.shape

    if padding > 0:
        input_binary = pad_1d_binary(input_binary, padding, val=1)

    input_unfolded = input_binary.unfold(2, K, stride)
    B, C_in, L_out, K = input_unfolded.shape

    input_flat = input_unfolded.permute(0, 2, 1, 3)
    input_flat = input_flat.unsqueeze(2)

    weight_flat = weight_binary.unsqueeze(0).unsqueeze(0)
    xnor_result = (input_flat == weight_flat)

    popcount = xnor_result.sum(dim=(3,4), dtype=torch.int32)

    total_bits = C_in * K
    output = 2*popcount - total_bits

    return output.permute(0, 2, 1).to(torch.int32)

def maxpool_int(x_int, kernel_size=7, stride=2):
    B, C, L = x_int.shape
    if L < kernel_size:
        pad_size = kernel_size - L
        pad_tensor = torch.full((B, C, pad_size), -2**30, dtype=torch.int32, device=x_int.device)
        x_int = torch.cat([x_int, pad_tensor], dim=2)
        L = x_int.shape[2]

    L_out = (L - kernel_size) // stride + 1
    if L_out <= 0:
        return torch.full((B, C, 1), -2**30, dtype=torch.int32, device=x_int.device)

    # Unfold and take maximum
    unfolded = x_int.unfold(2, kernel_size, stride)  # [B, C, L_out, kernel_size]
    return unfolded.max(dim=3)[0]  # [B, C, L_out]

def threshold_activation(x, threshold_params, pool_kernel=7, pool_stride=2):
    threshold_pos_int, threshold_neg_int, a_sign, scale_factor = threshold_params

    x_pooled = maxpool_int(x, pool_kernel, pool_stride)
    x_scaled = x_pooled.to(torch.int64) * int(scale_factor)

    B, C, L = x_pooled.shape
    thresh_pos = threshold_pos_int.view(1, C, 1).to(torch.int64).expand(B, C, L)
    thresh_neg = threshold_neg_int.view(1, C, 1).to(torch.int64).expand(B, C, L)
    a_s = a_sign.view(1, C, 1).expand(B, C, L)

    pos_mask = x_pooled >= 0
    neg_mask = ~pos_mask

    cond_pos = pos_mask & (x_scaled >= thresh_pos)

    cond_neg_a_pos = neg_mask & (x_scaled >= thresh_neg)
    cond_neg_a_neg = neg_mask & (x_scaled <= thresh_neg)
    cond_neg = torch.where(a_s >= 0, cond_neg_a_pos, cond_neg_a_neg)

    output = torch.where(
        cond_pos | cond_neg,
        torch.ones_like(x_pooled, dtype=torch.int8),
        -torch.ones_like(x_pooled, dtype=torch.int8)
    )
    return output.float()  

def final_layer(x, final_params, pool_kernel=7, pool_stride=2, scale_factor=2**12):
    alpha_pos_int, alpha_neg_int, beta_int = final_params
    x_pooled = maxpool_int(x, pool_kernel, pool_stride)
    x_scaled = x_pooled.to(torch.int64) * int(scale_factor)

    B, C, L = x_pooled.shape
    alpha_p = alpha_pos_int.view(1, C, 1).to(torch.int64).expand(B, C, L)
    alpha_n = alpha_neg_int.view(1, C, 1).to(torch.int64).expand(B, C, L)
    beta = beta_int.view(1, C, 1).to(torch.int64).expand(B, C, L)

    pos_mask = x_pooled >= 0
    output_pos = alpha_p * x_scaled + beta * int(scale_factor)
    output_neg = alpha_n * x_scaled + beta * int(scale_factor)

    output = torch.where(pos_mask, output_pos, output_neg)
    output_scaled = output.float() / (scale_factor * scale_factor)

    return output_scaled

class HardwareOptimizedBlock:
    def __init__(self, weight_binary_int8, threshold_params, pool_kernel=7, pool_stride=2, stride=1, padding=5):
        self.weight_binary = weight_binary_int8  # Stored as int8 {-1, +1}
        self.threshold_params = threshold_params
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        if x.dtype != torch.int8:
            x_binary = torch.sign(x).to(torch.int8)
        else:
            x_binary = x

        # XNOR + POPCOUNT convolution
        conv_out = xnor_popcount_conv1d(
            x_binary.float(), self.weight_binary.float(),
            stride=self.stride, padding=self.padding
        )
        # Threshold activation
        output = threshold_activation(
            conv_out, self.threshold_params,
            self.pool_kernel, self.pool_stride
        )
        return output

class HardwareFinalBlock:
    def __init__(self, weight_binary_int8, final_params, pool_kernel=7, pool_stride=2, stride=1, padding=5):
        self.weight_binary = weight_binary_int8
        self.final_params = final_params
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        if x.dtype != torch.int8:
            x_binary = torch.sign(x).to(torch.int8)
        else:

            x_binary = x

        conv_out = xnor_popcount_conv1d(
            x_binary.float(), self.weight_binary.float(),
            stride=self.stride, padding=self.padding

        )
        output = final_layer(
            conv_out, self.final_params,
            self.pool_kernel, self.pool_stride
        )
        return output

class ECG_BNN_MODEL:
    def __init__(self, num_classes=5):
        self.num_classes = num_classes
        self.scale_factor = 2**12

        self.block_configs = [
            [1, 8, 7, 2, 5, 7, 2],      # Block 1
            [8, 16, 7, 1, 5, 7, 2],     # Block 2
            [16, 32, 7, 1, 5, 7, 2],    # Block 3
            [32, 32, 7, 1, 5, 7, 2],    # Block 4
            [32, 64, 7, 1, 5, 7, 2],    # Block 5
            [64, 5, 7, 1, 5, 7, 2],     # Block 6
        ]

        self.blocks = []
        self.total_parameters = 0

    def load_from_pytorch_model(self, state_dict_path):
        print(f"Loading and converting model: {state_dict_path}")
        state_dict = torch.load(state_dict_path, map_location='cpu')

        self.blocks = []

        # Process blocks 1-5 (with threshold activation)
        for i in range(5):
            block_name = f'block{i+1}'
            print(f"Converting {block_name}...")

            # Extract and binarize weights
            conv_weight = state_dict[f'{block_name}.conv.weight']
            weight_binary = torch.sign(conv_weight).to(torch.int8)
            self.total_parameters += weight_binary.numel()

            # Compute threshold parameters
            bn_w = state_dict[f'{block_name}.bn.weight'].float()
            bn_b = state_dict[f'{block_name}.bn.bias'].float()
            bn_m = state_dict[f'{block_name}.bn.running_mean'].float()
            bn_v = state_dict[f'{block_name}.bn.running_var'].float()
            prelu_w = state_dict[f'{block_name}.prelu.weight'].float()

            threshold_params = self._compute_thresholds(bn_w, bn_b, bn_m, bn_v, prelu_w)

            # Create hardware block
            cfg = self.block_configs[i]
            block = HardwareOptimizedBlock(
                weight_binary, threshold_params,
                cfg[5], cfg[6], cfg[3], cfg[4]
            )
            self.blocks.append(block)

        # Process block 6 (final block)
        print("Converting block6 (final)...")
        conv_weight = state_dict['block6.conv.weight']

        weight_binary = torch.sign(conv_weight).to(torch.int8)
        self.total_parameters += weight_binary.numel()


        # Compute final affine parameters
        bn_w = state_dict['block6.bn.weight'].float()
        bn_b = state_dict['block6.bn.bias'].float()
        bn_m = state_dict['block6.bn.running_mean'].float()
        bn_v = state_dict['block6.bn.running_var'].float()
        prelu_w = state_dict['block6.prelu.weight'].float()

        final_params = self._compute_final_params(bn_w, bn_b, bn_m, bn_v, prelu_w)

        cfg = self.block_configs[5]
        final_block = HardwareFinalBlock(
            weight_binary, final_params,
            cfg[5], cfg[6], cfg[3], cfg[4]
        )

        self.blocks.append(final_block)


        # Calculate storage requirements
        storage_bits = self.total_parameters  # 1 bit per parameter
        storage_bytes = storage_bits // 8 + (1 if storage_bits % 8 else 0)
        storage_kb = storage_bytes / 1024


        print(f"Conversion complete!")
        print(f"Total parameters: {self.total_parameters:,}")
        print(f"Storage requirement: {storage_kb:.2f} KB (1-bit weights)")
        print(f"Paper claim: 3.76 KB")

        return storage_kb


    def _compute_thresholds(self, bn_w, bn_b, bn_m, bn_v, prelu_w, eps=1e-5):
        if prelu_w.numel() == 1:
            a = prelu_w.expand_as(bn_m)
        else:
            a = prelu_w.view(-1)

        scale = bn_w / torch.sqrt(bn_v + eps)
        shift = bn_b - scale * bn_m

        threshold_pos = -shift / scale
        threshold_neg = -shift / (scale * a)
        threshold_pos_int = (threshold_pos * self.scale_factor).round().to(torch.int32)
        threshold_neg_int = (threshold_neg * self.scale_factor).round().to(torch.int32)
        a_sign = torch.sign(a).to(torch.int32)

        return threshold_pos_int, threshold_neg_int, a_sign, self.scale_factor


    def _compute_final_params(self, bn_w, bn_b, bn_m, bn_v, prelu_w, eps=1e-5):
        if prelu_w.numel() == 1:
            a = prelu_w.expand_as(bn_m)
        else:
            a = prelu_w.view(-1)

        scale = bn_w / torch.sqrt(bn_v + eps)
        shift = bn_b - scale * bn_m

        alpha_pos = scale
        alpha_neg = scale * a

        alpha_pos_int = (alpha_pos * self.scale_factor).round().to(torch.int32)
        alpha_neg_int = (alpha_neg * self.scale_factor).round().to(torch.int32)
        beta_int = (shift * self.scale_factor).round().to(torch.int32)

        return alpha_pos_int, alpha_neg_int, beta_int

    def forward(self, signal):
        # Preprocess signal to binary
        x = preprocess_to_binary(signal)

        # Forward through all blocks
        for i, block in enumerate(self.blocks):
            x = block.forward(x)
            # Convert intermediate outputs to binary for next layer (except last)
            if i < len(self.blocks) - 1:
                x = torch.sign(x)

        # Global Sum Pooling (equivalent to mean with fixed scaling)
        x = x.sum(dim=2)  # Sum instead of mean for integer compatibility
        return x


    def predict(self, signal):
        with torch.no_grad():
            logits = self.forward(signal)
            pred_idx = torch.argmax(logits, dim=1).item()
            return pred_idx, LABELS[pred_idx], logits.squeeze().cpu().numpy()

def load_ecg(path):
    arr = np.load(path, allow_pickle=True)
    if arr.dtype == object:
        arr = np.array(arr.tolist())
    return np.asarray(arr).squeeze()

def sample_files(root, n=50):
    files = []
    for d in sorted(Path(root).glob('*')):
        if d.is_dir():

            files.extend(d.glob('*.npy'))
    random.seed(33)
    return random.sample(files, min(n, len(files)))


def main():
    parser = argparse.ArgumentParser(description="ECG BNN - Stage 4: Complete Integer Pipeline")
    parser.add_argument("--model", type=str, default="modelsFBI/LP_ECG_Net_for_5_90.37%.pth")
    parser.add_argument("--data", type=str, default="ECG_Dataset/ECG-5")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--benchmark", action="store_true", help="Run timing benchmark")

    args = parser.parse_args()

    model = ECG_BNN_MODEL()
    storage_kb = model.load_from_pytorch_model(args.model)

    files = sample_files(args.data, args.n)
    test_signals = []

    for f in files:
        sig = load_ecg(f)
        gt = int(Path(f).parent.name.split()[0]) - 1
        test_signals.append((sig, gt))

    correct = 0
    if args.benchmark:
        import time
        times = []
    for i, (signal, gt) in enumerate(test_signals, 1):
        if args.benchmark:
            start_time = time.time()
        pred_idx, pred_label, logits = model.predict(signal)
        if args.benchmark:
            end_time = time.time()
            times.append(end_time - start_time)
        if pred_idx == gt:
            correct += 1

        status = '✓' if pred_idx == gt else '✗'
        print(f"{i:3d}: GT={LABELS[gt]} Pred={pred_label} {status}")

    accuracy = correct / len(test_signals) * 100
    print("=" * 60)

    print("FINAL RESULTS:")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{len(test_signals)})")
    print(f"Storage Size: {storage_kb:.2f} KB")

    if args.benchmark:
        avg_time = np.mean(times) * 1000  # Convert to ms
        print(f"Average inference time: {avg_time:.2f} ms per sample")

if __name__ == "__main__":
    main()
