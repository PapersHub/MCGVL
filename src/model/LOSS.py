import torch
import torch.nn as nn
import pdb

class TGRegressionCriterion(nn.Module):
    def __init__(self):
        super(TGRegressionCriterion, self).__init__()

        self.regloss1 = nn.SmoothL1Loss()
        self.regloss2 = nn.SmoothL1Loss()

    def forward(self, net_outs, gts):
        loc  = net_outs["grounding_loc"]
        s_gt = gts["grounding_start_pos"].cuda()
        e_gt = gts["grounding_end_pos"].cuda()


        total_loss = self.regloss1(loc[:,0], s_gt) + self.regloss2(loc[:,1], e_gt)

        return total_loss

class TAGLoss(nn.Module):
    def __init__(self):
        super(TAGLoss, self).__init__()

        self.w = 1.0

    def forward(self, net_outs, gts):
        mask = gts["tag_att_masks"]
        w = net_outs["tag_attw"]

        ac_loss = (-mask*torch.log(w+1e-8)).sum(1) / mask.sum(1)
        ac_loss = (self.w * ac_loss.mean(0))

        return ac_loss

class LOSS(nn.Module):
    def __init__(self):
        super(LOSS, self).__init__()
        self.names = ['grounding', 'tag']
        self.crits = nn.ModuleList([TGRegressionCriterion(), TAGLoss()])

    def forward(self, crit_inp, gts):
        self.loss = {}
        self.loss["total_loss"] = 0
        for name,crit in iter(zip(self.names, self.crits)):
            self.loss[name] = crit(crit_inp, gts)
            self.loss["total_loss"] += self.loss[name]
        return self.loss["total_loss"]