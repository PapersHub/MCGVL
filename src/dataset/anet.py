from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pdb
import time
import json
import h5py
import string
import numpy as np
np.set_printoptions(precision=4)
from tqdm import tqdm

import torch
import torch.utils.data as data

from src.dataset.abstract_dataset import AbstractDataset
from src.utils import utils, io_utils

def create_loaders(split, loader_configs, num_workers):
    dsets, L = {}, {}
    for di,dt in enumerate(split):
        shuffle = True if dt == "train" else False
        drop_last = True if dt == "train" else False
        dsets[dt] = ActivityNetCaptionsDataset(loader_configs[di])
        L[dt] = data.DataLoader(
            dsets[dt],
            batch_size = loader_configs[di]["batch_size"],
            num_workers = num_workers,
            shuffle = shuffle,
            collate_fn = dsets[dt].collate_fn,
            drop_last= drop_last
        )
    return dsets, L


class ActivityNetCaptionsDataset(AbstractDataset):
    def __init__(self, config):
        super(self.__class__, self).__init__(config)

        self.S = config.get("num_segment", 128)
        self.split = config.get("split", "train")
        self.data_dir = config.get("data_dir", "")
        self.feature_type = config.get("feature_type", "C3D")
        self.in_memory = config.get("in_memory", False)
        self.feat_hdf5 = config.get("video_feature_path",
                "data/ActivityNet/feats/sub_activitynet_v1-3.c3d.hdf5")

        paths = self._get_data_path(config)

        ann_path = config.get("annotation_path",
                "data/ActivityNet/captions/annotations/train.json")
        self.anns, self.qids, self.vids  = self._load_annotation(ann_path)
        if not self._exist_data(paths):
            self.generate_labels(config)

        if self.in_memory:
            self.feats = {}
            h = io_utils.load_hdf5(self.feat_hdf5, verbose=False)
            for k in tqdm(self.vids, desc="In-Memory: vid_feat"):
                self.feats[k] = h[k]["c3d_features"][:]

            self.s_pos, self.e_pos, self.att_mask = {}, {}, {}
            grd_info = io_utils.load_hdf5(self.paths["grounding_info"], False)
            for k in tqdm(self.qids, desc="In-Memory: grounding"):
                self.s_pos[k] = grd_info["start_pos/"+k][()]
                self.e_pos[k] = grd_info["end_pos/"+k][()]
                self.att_mask[k] = grd_info["att_mask/"+k][()]

            self.query_labels = {}
            query_labels = h5py.File(self.paths["query_labels"], "r")
            for k in tqdm(self.qids, desc="In-Memory: query"):
                self.query_labels[k]= query_labels[k][:]

        query_info = io_utils.load_json(self.paths["query_info"])
        self.wtoi = query_info["wtoi"]
        self.itow = query_info["itow"]
        self.query_lengths = query_info["query_lengths"]

        self.batch_size = config.get("batch_size", 64)
        self.num_instances = len(self.qids)

    def __getitem__(self, idx):
        qid = str(self.qids[idx])
        vid = self.anns[qid]["video_id"]
        timestamp = self.anns[qid]["timestamps"]
        duration = self.anns[qid]["duration"]

        if self.in_memory:
            q_label = self.query_labels[qid]
        else:
            query_labels = h5py.File(self.paths["query_labels"], "r")
            q_label = query_labels[qid][:]
        q_leng = self.query_lengths[qid]

        if self.in_memory:
            start_pos = self.s_pos[qid]
            end_pos = self.e_pos[qid]
        else:
            grd_info = io_utils.load_hdf5(self.paths["grounding_info"], False)
            start_pos = grd_info["start_pos/"+qid][()]
            end_pos = grd_info["end_pos/"+qid][()]

        if self.in_memory:
            vid_feat_all = self.feats[vid]
        else:
            vid_feat_all = io_utils.load_hdf5(self.feat_hdf5, verbose=False)[vid]["c3d_features"]
        vid_feat, nfeats, start_index, end_index = self.get_fixed_length_feat(
                vid_feat_all, self.S, start_pos, end_pos)

        vid_mask = np.zeros((self.S, 1))
        vid_mask[:nfeats] = 1

        if self.in_memory:
            att_mask = self.att_mask[qid]
        else:
            att_mask = grd_info["att_mask/"+qid][:]
        instance = {
            "vids": vid,
            "qids": qid,
            "timestamps": timestamp,
            "duration": duration,
            "query_lengths": q_leng,
            "query_labels": torch.LongTensor(q_label).unsqueeze(0),
            "query_masks": (torch.FloatTensor(q_label)>0).unsqueeze(0),
            "grounding_start_pos": torch.FloatTensor([start_pos]),
            "grounding_end_pos": torch.FloatTensor([end_pos]),
            "grounding_att_masks": torch.FloatTensor(att_mask),
            "nfeats": torch.FloatTensor([nfeats]),
            "video_feats": torch.FloatTensor(vid_feat),
            "video_masks": torch.ByteTensor(vid_mask),
        }

        return instance

    def collate_fn(self, data):
        seq_items = ["video_feats", "video_masks", "grounding_att_masks"]
        tensor_items = [
            "query_labels", "query_masks", "nfeats",
            "grounding_start_pos", "grounding_end_pos",
        ]
        batch = {k: [d[k] for d in data] for k in data[0].keys()}

        if len(data) == 1:
            for k,v in batch.items():
                if k in tensor_items:
                    batch[k] = torch.cat(batch[k], 0)
                elif k in seq_items:
                    batch[k] = torch.nn.utils.rnn.pad_sequence(
                            batch[k], batch_first=True)
                else:
                    batch[k] = batch[k][0]

        else:
            for k in tensor_items:
                batch[k] = torch.cat(batch[k], 0)
            for k in seq_items:
                batch[k] = torch.nn.utils.rnn.pad_sequence(
                    batch[k], batch_first=True)

        return batch

    def get_vocab_size(self):
        return len(self.wtoi)

    def get_wtoi(self):
        return self.wtoi

    def get_itow(self):
        return self.itow

    def _get_data_path(self, config):

        split = config.get("split", "train")
        L = config.get("max_length", 20)
        F = config.get("frequency_threshold", 1)
        S = config.get("num_segment", 128)
        FT = config.get("feature_type", "C3D")

        root_dir = os.path.join(config.get("data_dir", ""), "preprocess")
        grounding_info_path = os.path.join(root_dir,
                "grounding_info", "{}_labels_S{}_{}.hdf5".format(split, S, FT))
        query_info_path = os.path.join(root_dir,
                "query_info", "{}_info_F{}_L{}_{}.json".format(split, F, L, FT))
        query_label_path = os.path.join(root_dir,
                "query_info", "{}_label_F{}_L{}_{}.hdf5".format(split, F, L, FT))
        caption_label_path = os.path.join(root_dir,
                "query_info", "{}_caption_label_F{}_L{}_{}.hdf5".format(split, F, L, FT))

        io_utils.check_and_create_dir(os.path.join(root_dir, "grounding_info"))
        io_utils.check_and_create_dir(os.path.join(root_dir, "query_info"))

        self.paths = {
            "grounding_info": grounding_info_path,
            "query_labels": query_label_path,
            "query_info": query_info_path,
        }
        return self.paths

    def _preprocessing(self, anns, new_anns, qid, vids):
        translator = str.maketrans("", "", string.punctuation)
        for vid in anns.keys():
            ann = anns[vid]
            duration = ann["duration"]
            for ts,q in zip(ann["timestamps"], ann["sentences"]):
                new_anns[str(qid)] = {
                    "timestamps": ts,
                    "query": q,
                    "tokens": utils.tokenize(q.lower(), translator),
                    "duration": duration,
                    "video_id": vid
                }
                qid += 1
        vids.extend(list(anns.keys()))
        return new_anns, qid, list(set(vids))

    def _load_annotation(self, ann_path):
        qid = 0
        new_anns = {}
        vids = []
        if isinstance(ann_path, list):
            for ap in ann_path:
                anno = io_utils.load_json(ap)
                new_anns, qid, vids = self._preprocessing(anno, new_anns, qid, vids)
        else:
            anno = io_utils.load_json(ann_path)
            new_anns, qid, vids = self._preprocessing(anno, new_anns, qid, vids)

        return new_anns, list(new_anns.keys()), vids

    def generate_labels(self, config):
        if not os.path.exists(self.paths["query_labels"]):
            train_ann_path = "data/ActivityNet/captions/annotations/train.json"
            train_anns, _, _ = self._load_annotation(train_ann_path)
            wtoi = self._build_vocab(train_anns)
            itow = {v:k for k,v in wtoi.items()}

            L = config.get("max_length", 25)
            encoded = self._encode_query(self.anns, wtoi, L)
            query_labels = io_utils.open_hdf5( self.paths["query_labels"], "w")
            for qid in tqdm(encoded["query_lengths"].keys(), desc="Saving query"):
                _ = query_labels.create_dataset(str(qid), data=encoded["query_labels"][qid])
            query_labels.close()

            query_info = {
                "wtoi": wtoi,
                "itow": itow,
                "query_lengths": encoded["query_lengths"],
            }
            io_utils.write_json(self.paths["query_info"], query_info)

        if not os.path.exists(self.paths["grounding_info"]):
            if self.feature_type == "C3D":
                features = io_utils.load_hdf5(self.feat_hdf5)
            grd_dataset = io_utils.open_hdf5(self.paths["grounding_info"], "w")
            start_pos = grd_dataset.create_group("start_pos")
            end_pos = grd_dataset.create_group("end_pos")
            att_masks = grd_dataset.create_group("att_mask")

            for qid,ann in tqdm(self.anns.items(), desc="Gen. Grd. Labels"):
                ts = ann["timestamps"]
                vid_d = ann["duration"]
                start = ts[0] / vid_d
                end = ts[1] / vid_d

                vid = ann["video_id"]
                nfeats = features[vid]["c3d_features"][:].shape[0]
                nfeats = min(nfeats, self.S)

                fs = utils.timestamp_to_featstamp(ts, nfeats, vid_d)
                att_mask = np.zeros((self.S))
                att_mask[fs[0]:fs[1]+1] = 1

                _ = start_pos.create_dataset(qid, data=start, dtype="float")
                _ = end_pos.create_dataset(qid, data=end, dtype="float")
                _ = att_masks.create_dataset(qid, data=att_mask, dtype="float")

            grd_dataset.close()



