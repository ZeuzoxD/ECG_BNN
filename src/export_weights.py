"""
Weight Extractor - extracts weights and computes thresholds from PyTorch .pth files
Outputs: .bin and .h
"""

import torch
import struct
import numpy as np
import os
from pathlib import Path

class WeightExtractor:
    def __init__(self, pth_path, output_dir="extracted_weights", scale=20):
        self.pth_path = pth_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.scale = scale
        self.state_dict = None
        self.weights = []
        self.thresholds = []

    def load_model(self):
        print(f"\n Loading Model: {self.pth_path}")
        self.state_dict = torch.load(self.pth_path, map_location='cpu')
        print(f" Loaded {len(self.state_dict.keys())} parameters")

    def extract_weights(self):
        print(f"\n Extracting Weights ")
        for i in range(6):
            block_name = f'block{i+1}'
            weight_key = f'{block_name}.conv.weight'

            weight = self.state_dict[weight_key]
            weight_binary = torch.sign(weight).to(torch.int8)

            shape = tuple(weight_binary.shape)
            num_params = weight_binary.numel()

            print(f"    Block {i+1}: shape={shape}, params={num_params:,}")
            self.weights.append({
                'shape': shape,
                'data': weight_binary.flatten().numpy()
            })

        total_params = sum(len(w['data']) for w in self.weights)
        print(f"    ✓ Extracted {len(self.weights)} blocks, {total_params:,} total weight parameters")

    def compute_thresholds(self):
        print(f"\n Computing Thresholds")
        for i in range(6):
            block_name = f'block{i+1}'

            bn_weight = self.state_dict[f'{block_name}.bn.weight']
            bn_bias = self.state_dict[f'{block_name}.bn.bias']
            bn_mean = self.state_dict[f'{block_name}.bn.running_mean']
            bn_var = self.state_dict[f'{block_name}.bn.running_var']
            prelu_weight = self.state_dict[f'{block_name}.prelu.weight']

            eps = 1e-5
            num_channels = len(bn_weight)

            if prelu_weight.numel() == 1:
                a = prelu_weight.expand(num_channels)
            else:
                a = prelu_weight

            s = torch.sqrt(bn_var + eps)
            gamma = bn_weight.clone()
            gamma[gamma.abs() < 1e-6] = torch.sign(gamma[gamma.abs() < 1e-6]) * 1e-6
            delta_plus = bn_mean - bn_bias * s / gamma
            bn0 = gamma * (-bn_mean) / s + bn_bias

            delta_minus = torch.where(
                a.abs() >= 1e-12,
                delta_plus / a,
                torch.where(bn0 >= 0.0,
                            torch.full_like(delta_plus, -1e9),
                            torch.full_like(delta_plus, 1e9))
            )

            dp_int = (delta_plus * self.scale).round().to(torch.int32)
            dm_int = (delta_minus * self.scale).round().to(torch.int32)
            a_sign = torch.sign(a).to(torch.int8)

            print(f"    Block {i+1}: {num_channels} channels, "
                  f"del+ range=[{dp_int.min()}, {dp_int.max()}], "
                  f"del- range=[{dm_int.min()}, {dm_int.max()}]")

            self.thresholds.append({
                'num_channels': num_channels,
                'delta_plus': dp_int.numpy(),
                'delta_minus': dm_int.numpy(),
                'a_sign': a_sign.numpy()
            })

        total_threshold_params = sum(
            t['num_channels'] * 3 for t in self.thresholds
        )
        print(f"    ✓ Computed {len(self.thresholds)} blocks, {total_threshold_params:,} total threshold parameters")

    def save_binary(self, filename):
        output_path = self.output_dir / filename

        with open(output_path, 'wb') as f:
            # Header
            f.write(b'ECGBNN') # Magic Identifier
            f.write(struct.pack('H', 1)) # Version
            f.write(struct.pack('H', 6)) # Num blocks
            f.write(struct.pack('I', self.scale)) # Scale Factor

            for i in range(6):
                w = self.weights[i]
                t = self.thresholds[i]

                cout, cin, k = w['shape']
                f.write(struct.pack('HHH', cout, cin, k))
                f.write(w['data'].tobytes())
                
                num_ch = t['num_channels']
                f.write(struct.pack('H', num_ch))

                f.write(t['delta_plus'].tobytes())
                f.write(t['delta_minus'].tobytes())
                f.write(t['a_sign'].tobytes())
        
        size = output_path.stat().st_size
        print(f"    ✓ Binary: {output_path.name} ({size:,} bytes = {size/1024:.2f} KB)")
        return size

    
    def save_c_header(self, filename):
        output_path = self.output_dir / filename

        with open (output_path, 'w') as f:
            guard = filename.upper().replace('.', '_').replace('-', '_')
            f.write(f"/* Auto-generated ECG BNN weights - DO NOT EDIT */\n")
            f.write(f"#ifndef {guard}\n")
            f.write(f"#define {guard}\n\n")
            f.write(f"#include <stdint.h>\n\n")

            # Configuration
            f.write(f"#define NUM_BLOCKS 6\n")
            f.write(f"#define THRESHOLD_SCALE {self.scale}\n\n")

            # Block dimensions
            f.write("/* Block dimensions: [Cout, Cin, K] */\n")
            f.write("const uint16_t block_dims[NUM_BLOCKS][3] = {\n")
            for i, w in enumerate(self.weights):
                cout, cin, k = w['shape']
                f.write(f"    {{{cout:3d}, {cin:3d}, {k}}}")
                if i < len(self.weights) - 1:
                    f.write(",")
                f.write(f"  /* Block {i+1} */\n")
            f.write("};\n\n")

            # Total weight counts for memory allocation
            f.write("/* Total weights per block */\n")
            f.write("const uint32_t block_weight_counts[NUM_BLOCKS] = {\n    ")
            counts = [len(w['data']) for w in self.weights]
            f.write(", ".join(str(c) for c in counts))
            f.write("\n};\n\n")

            # Write weights for each block (as flat arrays)
            for i in range(6):
                w = self.weights[i]
                cout, cin, k = w['shape']
                data = w['data']
                f.write(f"/* Block {i+1} weights: [{cout}, {cin}, {k}] = {len(data)} bytes */\n")
                f.write(f"const int8_t block{i+1}_weights[{len(data)}] = {{\n")

                for idx in range(0, len(data), 16):
                    chunk = data[idx:idx+16]
                    f.write("    ")
                    f.write(", ".join(f"{val:2d}" for val in chunk))
                    if idx + 16 < len(data):
                        f.write(",")
                    f.write("\n")
                f.write("};\n\n")

            # Thresholds
            for i in range(6):
                t = self.thresholds[i]
                num_ch = t['num_channels']

                f.write(f"/* Block {i+1} thresholds: {num_ch} channels */\n")

                # Delta plus
                f.write(f"const int32_t block{i+1}_delta_plus[{num_ch}] = {{\n    ")
                f.write(", ".join(str(x) for x in t['delta_plus']))
                f.write("\n};\n")

                # Delta minus
                f.write(f"const int32_t block{i+1}_delta_minus[{num_ch}] = {{\n    ")
                f.write(", ".join(str(x) for x in t['delta_minus']))
                f.write("\n};\n")

                # A sign
                f.write(f"const int8_t block{i+1}_a_sign[{num_ch}] = {{\n    ")
                f.write(", ".join(str(x) for x in t['a_sign']))
                f.write("\n};\n\n")


            # Pointer arrays for easy access
            f.write("/* Pointer arrays for convenient access */\n")
            f.write("const int8_t* const block_weights[NUM_BLOCKS] = {\n")
            for i in range(6):
                f.write(f"    block{i+1}_weights")
                if i < 5:
                    f.write(",")
                f.write("\n")
            f.write("};\n\n")

            f.write("const int32_t* const block_delta_plus[NUM_BLOCKS] = {\n")
            for i in range(6):
                f.write(f"    block{i+1}_delta_plus")
                if i < 5:
                    f.write(",")
                f.write("\n")
            f.write("};\n\n")

            f.write("const int32_t* const block_delta_minus[NUM_BLOCKS] = {\n")
            for i in range(6):
                f.write(f"    block{i+1}_delta_minus")
                if i < 5:
                    f.write(",")
                f.write("\n")
            f.write("};\n\n")

            f.write("const int8_t* const block_a_sign[NUM_BLOCKS] = {\n")
            for i in range(6):
                f.write(f"    block{i+1}_a_sign")
                if i < 5:
                    f.write(",")
                f.write("\n")
            f.write("};\n\n")

            f.write(f"#endif /* {guard} */\n")

        size = output_path.stat().st_size
        print(f"    ✓ Header: {output_path.name} ({size:,} bytes = {size/1024:.2f} KB)")
        return size

    def print_summary(self):
        total_weights = sum(len(w['data']) for w in self.weights)
        total_thresholds = sum(
            t['num_channels'] * 3 for t in self.thresholds
        )
        
        weight_storage = total_weights * 1 # int8
        threshold_storage = sum(
            t['num_channels'] * 4 + # delta_plus int32
            t['num_channels'] * 4 + # delta_minus int32
            t['num_channels'] * 1   # a_sign int8
            for t in self.thresholds
        )
        header_storage = 14 # Magic(6) + version(2) + num_blocks(2) + scale(4)
        block_headers = 6 * (2 + 2 + 2 + 2) # Each block: dims(6) + num_ch(2)
        total_storage = header_storage + block_headers + weight_storage + threshold_storage

        print(f"\n Extraction Summary")
        print(f"   {'─' * 50}")
        print(f"   Model: {Path(self.pth_path).name}")
        print(f"   Blocks: {len(self.weights)}")
        print(f"   Scale factor: {self.scale}")
        print(f"\n   Parameters:")
        print(f"     Weights:    {total_weights:6,} × 1 byte  = {weight_storage:6,} bytes")
        print(f"     Thresholds: {total_thresholds:6,} params   = {threshold_storage:6,} bytes")
        print(f"     Overhead:   {'':6} {'':7} = {header_storage + block_headers:6,} bytes")
        print(f"     {'─' * 50}")
        print(f"     TOTAL:      {'':6} {'':7} = {total_storage:6,} bytes ({total_storage/1024:.2f} KB)")
        print(f"\n   Output: {self.output_dir.absolute()}")
 
def main():
    models = [
        {
            'name': '5-class',
            'pth': 'models/LP_ECG_Net_for_5_90.37%.pth',
            'output_dir': 'extracted_weights/ecg_5class',
            'prefix': 'ecg_5class'
        },
        {
            'name': '17-class',
            'pth': 'models/LP_ECG_Net_for_5_90.50%.pth',
            'output_dir': 'extracted_weights/ecg_17class',
            'prefix': 'ecg_17class'
        }
    ]

    for model_info in models:
        print(f"\n{'═' * 60}")
        print(f" {model_info['name'].upper()} MODEL ".center(60, '═'))
        print(f"{'═' * 60}")

        if not os.path.exists(model_info['pth']):
            print(f"   ⚠️  Model file not found: {model_info['pth']}")
            print(f"   Skipping...")
            continue

        try:
            # Create extractor (scale=20 to match inference.py)
            extractor = WeightExtractor(
                pth_path=model_info['pth'],
                output_dir=model_info['output_dir'],
                scale=20
            )

            # Extract
            extractor.load_model()
            extractor.extract_weights()
            extractor.compute_thresholds()


            # Save files
            print(f"\n💾 Saving files...")
            extractor.save_binary(f"{model_info['prefix']}_weights.bin")
            extractor.save_c_header(f"{model_info['prefix']}_weights.h")

            # Summary
            extractor.print_summary()

        except Exception as e:
            print(f"\n   ❌ Error processing {model_info['name']} model:")
            print(f"   {type(e).__name__}: {e}")
            continue

    print(f"\n{'═' * 60}")
    print("✅ Extraction Complete".center(60))
    print(f"{'═' * 60}\n")

if __name__ == "__main__":
    main()
