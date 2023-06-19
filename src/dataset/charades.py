from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
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
        dsets[dt] = CharadesDataset(loader_configs[di])
        L[dt] = data.DataLoader(
            dsets[dt],
            batch_size = loader_configs[di]["batch_size"],
            num_workers = num_workers,
            shuffle = shuffle,
            collate_fn = dsets[dt].collate_fn,
            drop_last= drop_last
        )
    return dsets, L


class CharadesDataset(AbstractDataset):
    def __init__(self, config):
        super(self.__class__, self).__init__(config)

        self.S = config.get("num_segment", 128)
        self.split = config.get("split", "train")
        self.data_dir = config.get("data_dir", "data/charades")
        self.feature_type = config.get("feature_type", "I3D")
        self.in_memory = config.get("in_memory", False)
        if self.feature_type == "I3D":
            self.feat_path = config.get(
                "video_feature_path",
                "data/charades/features/i3d_finetuned/{}.npy"
            )
        else:
            raise ValueError("Wrong feature_type")

        paths = self._get_data_path(config)

        ann_path = "../../data/charades/annotations/charades_sta_{}.txt".format(self.split)
        aux_ann_path = "../../data/charades/annotations/Charades_v1_{}.csv".format(self.split)
        self.anns, self.qids, self.vids = self._load_annotation(ann_path, aux_ann_path)
        if not self._exist_data(paths):
            self.generate_labels(config)

        if self.in_memory:
            self.feats = {}
            for vid in tqdm(self.vids, desc="In-Memory: vid_feat"):
                self.feats[vid] = np.load(self.feat_path.format(vid)).squeeze()

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

        if self.feature_type == "I3D":
            if self.in_memory:
                vid_feat_all = self.feats[vid]
            else:
                vid_feat_all = np.load(self.feat_path.format(vid)).squeeze()
            vid_feat, nfeats, start_index, end_index = self.get_fixed_length_feat(
                vid_feat_all, self.S, start_pos, end_pos)
        else:
            raise ValueError("Wrong feature_type")

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
                batch[k] = torch.nn.utils.rnn.pad_sequence(batch[k], batch_first=True)

        return batch

    def get_vocab_size(self):
        return len(self.wtoi)

    def get_wtoi(self):
        return self.wtoi

    def get_itow(self):
        return self.itow

    def _get_data_path(self, config):

        split = config.get("split", "train")
        L = config.get("max_length", 10)
        F = config.get("frequency_threshold", 1)
        S = config.get("num_segment", 128)
        FT = config.get("feature_type", "I3D")

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

    def _preprocessing(self, anns, aux_ann_path):
        aux_anns = io_utils.load_csv(aux_ann_path)
        vid2len = {ann["id"]: ann["length"] for ann in aux_anns}
        vids = []

        new_anns = dict()
        translator = str.maketrans("", "", string.punctuation)
        for qid,ann in enumerate(anns):
            info, query = ann.split("##")
            vid, spos, epos = info.split(" ")
            duration = vid2len[vid]
            new_anns[str(qid)] = {
                "timestamps": [float(spos), float(epos)],
                "query": query,
                "tokens": utils.tokenize(query.lower(), translator),
                "duration": float(duration),
                "video_id": vid
            }
            vids.append(vid)
        return new_anns, list(set(vids))

    def _load_annotation(self, ann_path, aux_path):
        anns = io_utils.load_lines_from(ann_path)
        new_anns, vids = self._preprocessing(anns, aux_path)
        return new_anns, list(new_anns.keys()), vids

    def generate_labels(self, config):
        if not os.path.exists(self.paths["query_labels"]):
            train_ann_path = "../../data/charades/annotations/charades_sta_train.txt"
            train_aux_path = "../../data/charades/annotations/Charades_v1_train.csv"
            train_anns, _, _ = self._load_annotation(train_ann_path, train_aux_path)
            wtoi = self._build_vocab(train_anns)
            itow = {v:k for k,v in wtoi.items()}

            L = config.get("max_length", 20)
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
                if self.feature_type == "I3D":
                    nfeats = np.load(self.feat_path.format(vid)).shape[0]
                else:
                    raise NotImplementedError()

                nfeats = min(nfeats, self.S)

                fs = utils.timestamp_to_featstamp(ts, nfeats, vid_d)
                att_mask = np.zeros((self.S))
                att_mask[fs[0]:fs[1]+1] = 1

                _ = start_pos.create_dataset(qid, data=start, dtype="float")
                _ = end_pos.create_dataset(qid, data=end, dtype="float")
                _ = att_masks.create_dataset(qid, data=att_mask, dtype="float")

            grd_dataset.close()