import os
import re
import numpy as np
import scipy.io as sio
from config import SAMPLE_LENGTH, MAX_SAMPLES_PER_CLASS

def load_data_from_dir(data_dir, fault_types, samples_per_class_dict=None):
    X, Y = [], []
    for file in os.listdir(data_dir):
        if not file.endswith('.mat'):
            continue
        match = re.search(r'\d+\s+([A-Z]+)_', file)
        if not match:
            continue
        fault = match.group(1)
        if fault not in fault_types:
            continue
        path = os.path.join(data_dir, file)
        try:
            data = sio.loadmat(path)['Data'][:, 2]
        except:
            continue
        sample_count = samples_per_class_dict.get(fault, MAX_SAMPLES_PER_CLASS) if samples_per_class_dict else MAX_SAMPLES_PER_CLASS
        for _ in range(sample_count):
            start = np.random.randint(0, len(data) - SAMPLE_LENGTH)
            segment = data[start:start + SAMPLE_LENGTH]
            fft_segment = np.abs(np.fft.fft(segment))[:SAMPLE_LENGTH // 2]
            fft_segment = (fft_segment - np.mean(fft_segment)) / (np.std(fft_segment) + 1e-8)
            X.append(fft_segment)
            Y.append(fault)
    return np.array(X), np.array(Y)

def load_data_per_fault_class(config_dict):
    X, Y = [], []
    for fault, conf in config_dict.items():
        data_dir = conf['dir']
        sample_count = int(conf['num'])
        loaded = 0
        for file in os.listdir(data_dir):
            if not file.endswith('.mat'):
                continue
            match = re.search(r'\d+\s+([A-Z]+)_', file)
            if not match or match.group(1) != fault:
                continue
            path = os.path.join(data_dir, file)
            try:
                data = sio.loadmat(path)['Data'][:, 2]
            except:
                continue
            for _ in range(sample_count - loaded):
                if len(data) < SAMPLE_LENGTH:
                    continue
                start = np.random.randint(0, len(data) - SAMPLE_LENGTH)
                segment = data[start:start + SAMPLE_LENGTH]
                fft_segment = np.abs(np.fft.fft(segment))[:SAMPLE_LENGTH // 2]
                fft_segment = (fft_segment - np.mean(fft_segment)) / (np.std(fft_segment) + 1e-8)
                X.append(fft_segment)
                Y.append(fault)
                loaded += 1
                if loaded >= sample_count:
                    break
            if loaded >= sample_count:
                break
    return np.array(X), np.array(Y)

def load_few_shot_composite_samples_by_type(composite_config, sample_length=2000):
    X, Y = [], []
    for fault, config in composite_config.items():
        data_dir = config['dir']
        max_num = config['num']
        count = 0
        for file in os.listdir(data_dir):
            if not file.endswith('.mat'):
                continue
            match = re.search(r'\d+\s+([A-Z]+)_', file)
            if not match or match.group(1) != fault:
                continue
            path = os.path.join(data_dir, file)
            try:
                data = sio.loadmat(path)['Data'][:, 2]
            except:
                continue
            for _ in range(max_num - count):
                if len(data) < sample_length:
                    continue
                start = np.random.randint(0, len(data) - sample_length)
                segment = data[start:start + sample_length]
                fft_segment = np.abs(np.fft.fft(segment))[:sample_length // 2]
                fft_segment = (fft_segment - np.mean(fft_segment)) / (np.std(fft_segment) + 1e-8)
                X.append(fft_segment)
                Y.append(fault)
                count += 1
                if count >= max_num:
                    break
            if count >= max_num:
                break
    return np.array(X), np.array(Y)
