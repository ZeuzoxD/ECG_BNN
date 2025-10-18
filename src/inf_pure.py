import struct
import random
import glob
import os
import math

class BinaryWeightLoader:
    def __init__(self, bin_path):
        self.bin_path = bin_path
        self.scale = None
        self.weights = []
        self.thresholds = []

    def load(self):
        with open(self.bin_path, 'rb') as f:
            magic = f.read(6)
            if magic != b'ECGBNN':
                raise ValueError(f"Invalid binary file format. Expected 'ECGBNN', got {magic}")

            version = struct.unpack('H', f.read(2))[0]
            num_blocks = struct.unpack('H', f.read(2))[0]
            self.scale = struct.unpack('I', f.read(4))[0]
            print(f"Loading binary weights: version={version}, blocks={num_blocks}, scale={self.scale}")

            for block_idx in range(num_blocks):
                cout = struct.unpack('H', f.read(2))[0]
                cin = struct.unpack('H', f.read(2))[0]
                k = struct.unpack('H', f.read(2))[0]

                num_weights = cout * cin * k
                weight_data = struct.unpack(f'{num_weights}b', f.read(num_weights))

                weights = []
                idx = 0
                for co in range(cout):
                    cout_list = []
                    for ci in range(cin):
                        cin_list = []
                        for ki in range(k):
                            cin_list.append(weight_data[idx])
                            idx += 1
                        cout_list.append(cin_list)
                    weights.append(cout_list)

                num_channels = struct.unpack('H', f.read(2))[0]

                delta_plus = list(struct.unpack(f'{num_channels}i', f.read(num_channels * 4)))
                delta_minus = list(struct.unpack(f'{num_channels}i', f.read(num_channels * 4)))
                a_sign = list(struct.unpack(f'{num_channels}b', f.read(num_channels)))

                self.weights.append(weights)
                self.thresholds.append((delta_plus, delta_minus, a_sign))

                print(f"  Block {block_idx + 1}: weights=[{cout}, {cin}, {k}], channels={num_channels}")

            print(f"Loaded {len(self.weights)} blocks successfully\n")
            return self.weights, self.thresholds, self.scale

class ECG_BNN_INFERENCE:
    def __init__(self, bin_path, num_classes=5, labels=None):
        self.num_classes = num_classes
        self.labels = labels 

        self.blocks = [
            [1, 8, 7, 2, 5, 7, 2],
            [8, 16, 7, 1, 5, 7, 2],
            [16, 32, 7, 1, 5, 7, 2],
            [32, 32, 7, 1, 5, 7, 2],
            [32, 64, 7, 1, 5, 7, 2],
            [64, num_classes, 7, 1, 5, 7, 2],
        ]

        loader = BinaryWeightLoader(bin_path)
        self.weights, self.thresholds, self.scale = loader.load()

    def pad_1d(self, x, pad, val=1):
        if pad == 0:
            return x
        C = len(x)
        padded = []
        for c in range(C):
            row = [val] * pad + x[c] + [val] * pad
            padded.append(row)
        return padded

    def binary_conv1d(self, x, w, stride=1, padding=0):
        Cin = len(x)
        L = len(x[0])
        Cout = len(w)
        K = len(w[0][0])

        x = self.pad1d(x, padding, val=1)
        L_padded = len(x[0])

        L_out = (L_padded - K) // stride + 1
        output = [[0 for _ in range(L_out)] for _ in range(Cout)]

        for i in range(L_out):
            start = i * stride
            for co in range(Cout):
                popcount = 0
                for ci in range(Cin):
                    for ki in range(K):
                        if x[ci][start + ki] == w[co][ci][ki]:
                            popcount += 1
                output[co][i] = 2 * popcount - Cin * K

        return output

    def maxpool_int(self, x, k=7, s=2):
        C = len(x)
        L = len(x[0])
        L_out = (L - K) // s + 1
        output = [[0 for _ in range(L_out)] for _ in range(C)]

        for c in range(C):
            for i in range(L_out):
                start = i * s
                max_val = x[c][start]
                for j in range(1, k):
                    if x[c][start + j] > max_val:
                        max_val = x[c][start + j]
                output[c][i] = max_val
        return output

    def threshold_activation(self, x, params, pool_k=7, pool_s=2):
        delta_plus, delta_minus, a_sign = params

        x_pool = self.maxpool_int(x, pool_k, pool_s)
        C = len(x_pool)
        L = len(x_pool[0])

        for c in range(C):
            for l in range(L):
                x_pool[c][l] *= self.scale

        output  = [[0 for _ in range(L)] for _ in range(C)]
        for c in range(C):
            dp = delta_plus[c]
            dm = delta_minus[c]
            a_s = a_sign[c]

            for l in range(L):
                val = x_pool[c][l]
                
                if val >= 0:
                    if val >= dp:
                        output[c][l] = 1
                    else:
                        output[c][l] = -1
                else:
                    if a_s >= 0:
                        if val >= dm:
                            output[c][l] = 1
                        else:
                            output[c][l] = -1
                    else:
                        if val <= dm:
                            output[c][l] = 1
                        else:
                            output[c][l] = -1

        return output

    def preprocess(self, signal):
        mean = sum(signal) / len(signal)
        variance = sum((x - mean) ** 2 for x in signal) / len(signal)
        std = math.sqrt(variance)
        signal_std = [(x - mean) / (std + 1e-8) for x in signal]
        signal_bin = []

        for x in signal_std:
            if x >= 0:
                signal_bin.append(1)
            else:
                signal_bin.append(-1)

        return[signal_bin]

    def forward(self, signal):
        x = self.preprocess(signal)
        for i in range(6):
            cin, cout, conv_k, conv_s, conv_p, pool_k, pool_s = self.blocks[i]
            x = self.binary_conv1d(x, self.weights[i], stride=conv_s, padding=conv_p)
            x = self.threshold_activation(x, self.thresholds[i], pool_k, pool_s)

        logits = [sum(channel) for channel in x]
        pred_idx = logits.index(max(logits))
        return pred_idx, logits

    def predict(self, signal):
        pred_idx, logits = self.forward(signal)
        pred_label = self.labels[pred_idx]
        return pred_idx, pred_label, logits

def load_ecg_numpy(path):
    with open(path, 'rb') as f:
        magic = f.read(6)
        if magic[:6] != b'\x93NUMPY':
            raise ValueError("Not a valid NPY file")

        major = struct.unpack('B', f.read(1))[0]
        minor = struct.unpack('B', f.read(1))[0]

        if major == 1:
            header_len = struct.unpack('<H', f.read(2))[0]
        else:
            header_len = struct.unpack('<I', f.read(4))[0]

        header = f.read(header_len).decode('latin1')

        data = []
        try: 
            while True:
                bytes_data = f.read(8)
                if not bytes_data:
                    break
                value = struct.unpack('<d', bytes_data)[0]
                data.append(value)
        except:
            pass

        return data

def load_ecg_mat(path):
    try:
        import scipy.io as scio
        data_train = scio.loadmat(path)
        data_arr = data_train.get('val')
        if hasattr(data_arr, 'flatten'):
            return data_arr.flatten().tolist()
        return list(data_arr)
    except ImportError:
        print("Warning: scipy not available, cannot load .mat files")
        return None

def load_ecg(path, classes_num=5):
    if classes_num == 17:
        return load_ecg_mat(path)
    else:
        return load_ecg_numpy(path)

def sample_files(root, n=50, classes_num=5):
    files = []
    file_ext = '.npy' if classes_num == 5 else '.mat'
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(file_ext):
                files.append(os.path.join(dirpath, filename))
    random.seed(62)
    return random.sample(files, min(n, len(files)))

def main():
    NUM_CLASSES = 5
    NUM_SAMPLES = 10

    if NUM_CLASSES == 17:
        bin_path = "extracted_weights/ecg_17class/ecg_17class_weights.bin"
        data_path = "ECG_Dataset/ECG-17"
        labels = ['NSR', 'APB', 'AFL', 'AFIB', 'SVTA', 'WPW', 'PVC', 'Bigeminy',
                  'Trigeminy', 'VT', 'IVR', 'VFL', 'Fusion', 'LBBBB', 'RBBBB', 'SDHB', 'PR']
    else:
        bin_path = "extracted_weights/ecg_5class/ecg_5class_weights.bin"
        data_path = "ECG_Dataset/ECG-5"
        labels = ['N', 'S', 'V', 'F', 'Q']

    model = ECG_BNN_INFERENCE(bin_path, NUM_CLASSES, labels)
    files = sample_files(data_path, NUM_SAMPLES, NUM_CLASSES)
    correct = 0

    for i, f in enumerate(files, 1):
        sig = load_ecg(f, NUM_CLASSES)
        dir_name = os.path.basename(os.path.dirname(f))
        gt_idx = int(dir_name.split()[0]) - 1
        gt_label = labels[gt_idx]
        pred_idx, pred_label, logits = model.predict(sig)
        is_correct = (pred_idx == gt_idx)
        if is_correct:
            correct += 1
        status = "✓" if is_correct else "✗"
        logits_str = "[" + ", ".join(f"{x:4d}" for x in logits) + "]"
        print(f"{i:3d}: GT={gt_label:8s} Pred={pred_label:8s} {status} | Logits: {logits_str}")

    accuracy = correct / len(files) * 100 if files else 0
    print("\nResults:")
    print(f"  Correct: {correct}/{len(files)}")
    print(f"  Accuracy: {accuracy:.2f}%")
