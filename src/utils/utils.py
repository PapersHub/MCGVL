import string
import numpy as np

import torch

def tokenize(txt, translator=None):
    if not translator:
        translator = str.maketrans("", "", string.punctuation)
    tokens = str(txt).lower().translate(translator).strip().split()
    return tokens

def label2string(itow, label, start_idx=2, end_idx=3):
    if isinstance(label, torch.Tensor):
        label = label.detach().cpu().numpy().squeeze()
    if label.ndim == 0:
        if label == end_idx:
            return itow[str(end_idx)]
        else:
            return itow[str(label)]
    assert label.ndim == 1, "{}".format(label.ndim)
    txt = []
    for l in label:
        if l == start_idx: continue
        if l == end_idx: break
        else: txt.append(itow[str(l)])
    return " ".join(txt).strip()

def string2label(itow, txt):
    if len(txt) == 0:
        return None
    else:
        label = []
        for w in txt:
            label.append(itow[w])
        return np.asarray(label)

def get_filename_from_path(file_path, delimiter="/"):
    filename = file_path.split(delimiter)[-1]
    return filename.split(".")[0]

def timestamp_to_featstamp(timestamp, nfeats, duration):
    start, end = timestamp
    start = min(int(round(start / duration * nfeats)), nfeats - 1)
    end = max(int(round(end / duration * nfeats)), start + 1)
    return start, end