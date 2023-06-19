import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from collections import OrderedDict
import random
from src.utils import accumulator


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()



def compute_tiou(pred, gt):
    intersection = max(0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])
    return float(intersection) / union

class TALLEvaluator(object):
    def __init__(self):
        self.tiou_threshold = [0.1, 0.3, 0.5, 0.7]
        self.metrics = ["R1-0.1", "R1-0.3", "R1-0.5", "R1-0.7", "mIoU"]
        self.duration = None

    def get_metric(self):
        return "R1-0.5"

    def set_duration(self, duration=[]):
        if len(duration) == 0:
            self.duration = None
        else:
            self.duration = duration

    def eval_instance(self, pred, gt, topk):
        correct = {str(tiou):0 for tiou in self.tiou_threshold}
        find = {str(tiou):False for tiou in self.tiou_threshold}
        if len(pred) == 0:
            return correct

        if len(pred) > topk:
            pred = pred[:topk]

        best_tiou = 0
        for loc in pred:
            cur_tiou = compute_tiou(loc, gt)

            if cur_tiou > best_tiou:
                best_tiou = cur_tiou

            for tiou in self.tiou_threshold:
                if (not find[str(tiou)]) and (cur_tiou >= tiou):
                    correct[str(tiou)] = 1
                    find[str(tiou)] = True

        return correct, best_tiou

    def eval(self, preds, gts):
        num_instances = float(len(preds))
        all_rank1 = {"R1-"+str(tiou):0 for tiou in self.tiou_threshold}
        all_rank5 = {"R5-"+str(tiou):0 for tiou in self.tiou_threshold}
        miou = 0

        ii = 0
        pt_idx = random.randint(0, len(gts)-1)
        for pred,gt in zip(preds, gts):
            if ii == pt_idx:
                if self.duration is not None:
                    print("pred: {}\tgt: {}\ttIoU: {:.4f}".format(
                        str(np.array(pred)/self.duration[ii]),
                        str(np.array(gt)/self.duration[ii]),
                        compute_tiou(np.array(pred).squeeze()/self.duration[ii],
                                   np.array(gt).squeeze()/self.duration[ii])
                    ))
                else:
                    print("pred: {}\tgt: {}\ttIoU: {}".format(
                            str(pred), str(gt), compute_tiou(np.array(pred).squeeze(), gt)))

            correct, _ = self.eval_instance(pred, gt, topk=1)
            for tiou in self.tiou_threshold:
                all_rank1["R1-"+str(tiou)] += correct[str(tiou)]

            correct, iou = self.eval_instance(pred, gt, topk=5)
            miou += iou
            for tiou in self.tiou_threshold:
                all_rank5["R5-"+str(tiou)] += correct[str(tiou)]

            ii += 1

        return all_rank1, all_rank5, miou

def istensor(data):
    return isinstance(data, torch.Tensor)

class HadamardProduct(nn.Module):
    def __init__(self):
        super(HadamardProduct, self).__init__()

        idim_1 = 512
        idim_2 = 512
        hdim = 512

        self.fc_1 = nn.Linear(idim_1, hdim)
        self.fc_2 = nn.Linear(idim_2, hdim)
        self.fc_3 = nn.Linear(hdim, hdim)

    def forward(self, inp):

        x1, x2 = inp[0], inp[1]
        return torch.sigmoid(self.fc_3(torch.relu(self.fc_1(x1)) * torch.relu(self.fc_2(x2))))

class NonLocalBlock(nn.Module):
    def __init__(self, prefix=""):
        super(NonLocalBlock, self).__init__()
        name = prefix if prefix is "" else prefix+"_"
        print("Non-Local Block - ", name)

        self.idim = 512
        self.odim = 512
        self.nheads = 4

        self.use_bias = True
        self.use_local_mask = False

        self.c_lin = nn.Linear(self.idim, self.odim*2, bias=self.use_bias)
        self.v_lin = nn.Linear(self.idim, self.odim, bias=self.use_bias)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(0.0)

    def forward(self, m_feats, mask):

        mask = mask.float()
        B, nseg = mask.size()


        m_k = self.v_lin(self.drop(m_feats))
        m_trans = self.c_lin(self.drop(m_feats))
        m_q, m_v = torch.split(m_trans, m_trans.size(2) // 2, dim=2)

        new_mq = m_q
        new_mk = m_k

        w_list = []
        mk_set = torch.split(new_mk, new_mk.size(2) // self.nheads, dim=2)
        mq_set = torch.split(new_mq, new_mq.size(2) // self.nheads, dim=2)
        mv_set = torch.split(m_v, m_v.size(2) // self.nheads, dim=2)
        for i in range(self.nheads):
            mk_slice, mq_slice, mv_slice = mk_set[i], mq_set[i], mv_set[i]

            m2m = mk_slice @ mq_slice.transpose(1,2) / ((self.odim // self.nheads) ** 0.5)
            if self.use_local_mask:
                local_mask = mask.new_tensor(self._get_mask(nseg, self.ksize, self.dilation))
                m2m = m2m.masked_fill(local_mask.unsqueeze(0).eq(0), -1e9)
            m2m = m2m.masked_fill(mask.unsqueeze(1).eq(0), -1e9)
            m2m_w = F.softmax(m2m, dim=2)
            w_list.append(m2m_w)

            r = m2m_w @ mv_slice if (i==0) else torch.cat((r, m2m_w @ mv_slice), dim=2)

        updated_m = self.drop(m_feats + r)
        return updated_m, torch.stack(w_list, dim=1)

    def _get_mask(self, N, ksize, d):
        if self.local_mask is not None: return self.local_mask
        self.local_mask = np.eye(N)
        K = ksize // 2
        for i in range(1, K+1):
            self.local_mask += np.eye(N, k=d+(i-1)*d)
            self.local_mask += np.eye(N, k=-(d+(i-1)*d))
        return self.local_mask


class MCGVL(nn.Module):
    def __init__(self, config):
        super(MCGVL, self).__init__()

        self.device = "cuda"
        self.evaluator = TALLEvaluator()
        self._create_counters()

        emb_idim = config['model'].get("query_enc_emb_idim")
        self.embedding = nn.Embedding(emb_idim, 300)
        self.rnn = nn.LSTM(300, 256, 2, batch_first=True, dropout=0.5, bidirectional=True)
        self.rnn_mlp = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU()
        )
        self.q_w = nn.Linear(512, 256)
        self.w_w = nn.Linear(512, 256)
        self.to_alpha = nn.Linear(256, 1)

        self.Z_ZZ = nn.Linear(1024, 512, bias=False)
        self.Z_Q = nn.Linear(1024, 512, bias=False)
        self.ZZ_RELU = nn.ReLU()
        self.ZZ_SIGMOID = nn.Sigmoid()
        self.wemb_linear = nn.Linear(300,512)
        self.max_pooling = nn.AdaptiveMaxPool1d(1)


        vemb_idim = config['model'].get("video_enc_vemb_idim")
        self.vid_emb_fn = nn.Sequential(*[
            nn.Linear(vemb_idim, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        ])

        self.pos_emb_fn = nn.Sequential(*[
            nn.Embedding(128, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        ])

        self.fusion_fn = HadamardProduct()

        self.gru = nn.GRU(512, 256, batch_first=True, bidirectional=True)

        self.fea2att = nn.Linear(512, 256, bias=False)
        self.to_alpha1 = nn.Linear(256, 1, bias=False)

        self.n_global_nl = 2
        self.global_fn = self._make_modulelist(
            NonLocalBlock(config, "lgi_global"), self.n_global_nl)

        self.fea2att1 = nn.Linear(512, 256, bias=False)
        self.to_alpha2 = nn.Linear(256, 1, bias=False)
        self.fc = nn.Linear(512, 512, bias=False)

        self.MLP = nn.Sequential(
            nn.Linear(in_features=512, out_features=512, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=2, bias=True),
            nn.ReLU()
        )


    def forward(self, net_inps):
        word_labels = net_inps["query_labels"].cuda()
        word_masks = net_inps["query_masks"].cuda()
        c3d_feats = net_inps["video_feats"].cuda()
        seg_masks = net_inps["video_masks"].squeeze(2).cuda()
        B, nseg, _ = c3d_feats.size()

        w_feat, q_feat, wemb = self.QuerySequenceEncoder(word_labels, word_masks)

        q_feat = q_feat.unsqueeze(1).expand_as(w_feat)
        Z = torch.cat((w_feat, q_feat), dim=2)
        ZZ = self.ZZ_RELU(self.Z_ZZ(Z))
        Q = self.ZZ_SIGMOID(self.Z_Q(Z))
        wemb = self.wemb_linear(wemb)
        e = ZZ * Q + wemb * (1-Q)
        e = e.transpose(2,1)
        e = self.max_pooling(e)
        e = e.transpose(2,1)

        v_feat = self.VideoEmbeddingWithPosition(c3d_feats, seg_masks)

        q4s_feat = e[:, 0, :].unsqueeze(1).expand(v_feat.size(0), v_feat.size(1), -1)
        m0 = self.fusion_fn([v_feat, q4s_feat])
        m = m0*v_feat


        M, _ = self.gru(m)


        attn_f = self.fea2att(e)
        dot1 = torch.tanh(attn_f)
        alpha1 = self.to_alpha1(dot1)
        attw = F.softmax(alpha1.transpose(1, 2), dim=2)

        a_feats = attw*M
        for s in range(self.n_global_nl):
            a_feats, _ = self.global_fn[s](a_feats, seg_masks)

        attn_f = self.fea2att1(a_feats)
        dot2 = torch.tanh(attn_f)
        alpha2 = self.to_alpha2(dot2)
        alpha2 = alpha2.masked_fill(seg_masks.float().unsqueeze(2).eq(0), -1e9)
        attw1 = F.softmax(alpha2.transpose(1, 2), dim=2)
        att_feats = attw1 @ a_feats
        att_feats = att_feats.squeeze(1)
        attw1 = attw1.squeeze(1)
        att_feats = self.fc(att_feats)

        loc = self.MLP(att_feats)

        outs = OrderedDict()
        outs["grounding_loc"] = loc
        outs["tag_attw"] = attw1
        return outs

    def _create_counters(self):
        self.counters = OrderedDict()
        for k in self.evaluator.metrics:
            self.counters[k] = accumulator.Accumulator(k)

    def _make_modulelist(self, net, n):
        assert n > 0
        new_net_list = nn.ModuleList()
        new_net_list.append(net)
        if n > 1:
            for i in range(n-1):
                new_net_list.append(copy.deepcopy(net))
        return new_net_list


    def QuerySequenceEncoder(self, onehot, mask):

        wemb = self.embedding(onehot)

        max_len = onehot.size(1)
        length = mask.sum(1)
        pack_wemb = nn.utils.rnn.pack_padded_sequence(
                wemb, length.cpu(), batch_first=True, enforce_sorted=False)
        w_feats, _ = self.rnn(pack_wemb)
        w_feats, max_ = nn.utils.rnn.pad_packed_sequence(
                w_feats, batch_first=True, total_length=max_len)
        w_feats = w_feats.contiguous()

        B, L, H = w_feats.size()
        idx = (length-1).long()
        idx = idx.view(B, 1, 1).expand(B, 1, H//2)
        fLSTM = w_feats[:,:,:H//2]
        bLSTM = w_feats[:,:,H//2:]

        s_feats = torch.mean((fLSTM+bLSTM)/2, dim=1)
        s_feats = self.rnn_mlp(s_feats)


        return w_feats, s_feats, wemb

    def VideoEmbeddingWithPosition(self, seg_feats, seg_masks):

        seg_emb = self.vid_emb_fn(seg_feats) * seg_masks.float().unsqueeze(2)

        pos = torch.arange(0, seg_masks.size(1)).type_as(seg_masks).unsqueeze(0).long()
        pos_emb = self.pos_emb_fn(pos)
        B, nseg, pdim = pos_emb.size()

        return seg_emb

    def prepare_batch(self, batch):
        self.gt_list = ["vids", "qids", "timestamps", "duration",
                   "grounding_start_pos", "grounding_end_pos",
                   "grounding_att_masks", "nfeats"]
        self.both_list = ["grounding_att_masks"]

        net_inps, gts = {}, {}
        for k in batch.keys():
            item = batch[k].to(self.device) \
                if istensor(batch[k]) else batch[k]

            if k in self.gt_list: gts[k] = item
            else: net_inps[k] = item

            if k in self.both_list: net_inps[k] = item

        gts["tag_att_masks"] = gts["grounding_att_masks"]
        return net_inps, gts

    def reset_status(self):
        self.results = {"predictions": [], "gts": [],
                        "durations": [], "vids": [], "qids": []}

    def compute_status(self, net_outs, gts):

        loc = net_outs["grounding_loc"].detach()
        B = loc.size(0)
        gt_ts = gts["timestamps"]
        vid_d = gts["duration"]

        for ii in range(B):
            pred = [[float(loc[ii,0])*vid_d[ii], float(loc[ii,1])*vid_d[ii]]]
            self.results["predictions"].append(pred)
            self.results["gts"].append(gt_ts[ii])
            self.results["durations"].append(vid_d[ii])
            self.results["vids"].append(gts["vids"][ii])
            self.results["qids"].append(gts["qids"][ii])

    def performances(self):
        nb = float(len(self.results["gts"]))
        self.evaluator.set_duration(self.results["durations"])
        rank1, rank5, miou = self.evaluator.eval(
            self.results["predictions"], self.results["gts"])
        for k, v in rank1.items():
            self.counters[k].add(v / nb, 1)
        self.counters["mIoU"].add(miou / nb, 1)

    def reset_counters(self):
        for k, v in self.counters.items():
            v.reset()

    def print_performances(self, epoch):
        val_list = self.evaluator.metrics
        txt = "epoch: {}".format(epoch)
        for k in val_list:
            v = self.counters[k]
            txt += ", {} = {:.4f}".format(v.get_name(), v.get_average())
        print(txt)


