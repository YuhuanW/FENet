# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn
import numpy as np

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel
import torch.nn.functional as F

def ngaussian_wasserstein_distance(box1, box2, sigma=1.0):
    b1_cx, b1_cy, b1_w, b1_h = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_cx, b2_cy, b2_w, b2_h = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    cx_L2Norm = torch.pow((b1_cx - b2_cx), 2)
    cy_L2Norm = torch.pow((b1_cy - b2_cy), 2)
    p1 = cx_L2Norm + cy_L2Norm

    w_FroNorm = torch.pow((b1_w - b2_w) / 2, 2)
    h_FroNorm = torch.pow((b1_h - b2_h) / 2, 2)
    p2 = w_FroNorm + h_FroNorm

    return p1 + p2

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class GRL(nn.Module):
    def __init__(self, loss_fcn):
        super(GRL, self).__init__()
        self.loss_fcn = loss_fcn
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply SL to each element

    def forward(self, pred, true, auto_iou=0.5):
        sigma = 0.1
        loss = self.loss_fcn(pred, true)
        if auto_iou < 0.2:
            auto_iou = 0.2
        b1 = true <= auto_iou - sigma
        a1 = torch.exp(-((true - auto_iou) ** 2) / (2 * sigma ** 2)) + 1
        b2 = (true > (auto_iou - 0.1)) & (true < auto_iou + sigma)
        a2 = torch.exp(-((true - auto_iou) ** 2) / (2 * sigma ** 2)) + 1  # a2 = math.exp(1.0 - auto_iou) ccc
        b3 = true >= auto_iou + sigma
        a3 = torch.exp(-((true - auto_iou) ** 2) / (2 * sigma ** 2)) + 1
        modulating_weight = a1 * b1 + a2 * b2 + a3 * b3
        loss *= modulating_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

def wasserstein_loss(pred, target, eps=1e-7, constant=12.8, reg_weight=0.1):#0.01
    # Calculate anchor area
    anchor_area = pred[:, 2] * pred[:, 3]

    # Get indices of anchors with area less than 32*32
    valid_indices = anchor_area < 32 * 32

    if torch.any(valid_indices):
        # Filter out anchors with area greater than or equal to 32*32
        pred = pred[valid_indices]
        target = target[valid_indices]

        pred_mean = pred[:, :2]
        target_mean = target[:, :2]

        pred_covariance = torch.diag_embed(pred[:, 2:4].float(), dim1=-2, dim2=-1)
        target_covariance = torch.diag_embed(target[:, 2:4].float(), dim1=-2, dim2=-1)

        pred_distribution = torch.distributions.MultivariateNormal(pred_mean, pred_covariance)
        target_distribution = torch.distributions.MultivariateNormal(target_mean, target_covariance)

        wasserstein_distance = torch.sqrt(torch.sum((pred_distribution.mean - target_distribution.mean) ** 2)) + \
                               torch.sum(torch.sqrt(torch.linalg.det(pred_distribution.covariance_matrix + eps))) + \
                               torch.sum(torch.sqrt(torch.linalg.det(target_distribution.covariance_matrix + eps)))

        # Regularization term added
        reg_term = reg_weight * (torch.sum(torch.abs(pred[:, 2:4])) + torch.sum(torch.abs(target[:, 2:4])))

        return torch.exp(- (wasserstein_distance + reg_term) / constant)
    else:
        # Return None if no anchors have area less than 32*32
        return None

def IoG(gt_box, pre_box):
    inter_xmin = torch.max(gt_box[:, 0], pre_box[:, 0])
    inter_ymin = torch.max(gt_box[:, 1], pre_box[:, 1])
    inter_xmax = torch.min(gt_box[:, 2], pre_box[:, 2])
    inter_ymax = torch.min(gt_box[:, 3], pre_box[:, 3])
    Iw = torch.clamp(inter_xmax - inter_xmin, min=0)
    Ih = torch.clamp(inter_ymax - inter_ymin, min=0)
    I = Iw * Ih
    G = ((gt_box[:, 2] - gt_box[:, 0]) * (gt_box[:, 3] - gt_box[:, 1])).clamp(1e-6)
    return I / G

def Ngaussian_wasserstein_distance(box1, box2, sigma=1.0):
    # # 计算框架的中心点坐标和宽度高度
    # b1_cx, b1_cy, b1_w, b1_h = (box1[:, 0] + box1[:, 2]) / 2, (box1[:, 1] + box1[:, 3]) / 2, box1[:, 2] - box1[:, 0], box1[:, 3] - box1[:, 1]
    # b2_cx, b2_cy, b2_w, b2_h = (box2[:, 0] + box2[:, 2]) / 2, (box2[:, 1] + box2[:, 3]) / 2, box2[:, 2] - box2[:, 0], box2[:, 3] - box2[:, 1]
    # 直接找到中心点坐标和WH
    b1_cx, b1_cy, b1_w, b1_h = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_cx, b2_cy, b2_w, b2_h = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # 计算距离矩阵的平方
    cx_L2Norm = torch.pow((b1_cx - b2_cx), 2)
    cy_L2Norm = torch.pow((b1_cy - b2_cy), 2)
    p1 = cx_L2Norm + cy_L2Norm

    w_FroNorm = torch.pow((b1_w - b2_w) / 2, 2)
    h_FroNorm = torch.pow((b1_h - b2_h) / 2, 2)
    p2 = w_FroNorm + h_FroNorm

    # 计算高斯 Wasserstein 距离
    distance = torch.sqrt(p1 + p2) / sigma

    # 对距离进行归一化
    normalized_distance = F.softmax(distance, dim=0)

    return normalized_distance

#ccc 2 * (x - t) / (1 - x ** 2)---->2 * (x - t) / (1 - t ** 2)
def smooth_ln(x, t=0.5):
    return torch.where(
        torch.le(x, t),
        torch.log((1 + x) / (1 - x)),
        2 * (x - t) / (1 - t ** 2) + np.log((1 + t) / (1 - t))
    )

def Wasserstein(pbox, gtbox, x1y1x2y2=True):
    """
    计算预测框和真实框之间的 Wasserstein 距离（用于 PGIoU）。

    参数:
        pbox: Tensor[N, 4]，预测框，[x1, y1, x2, y2] 或 [cx, cy, w, h]
        gtbox: Tensor[N, 4]，真实框，[x1, y1, x2, y2] 或 [cx, cy, w, h]
        x1y1x2y2: 布尔值，表示输入框的格式。如果为 True，则为 [x1, y1, x2, y2]

    返回:
        Tensor[N]，每对框的 Wasserstein 距离
    """
    if x1y1x2y2:
        # 将 [x1, y1, x2, y2] 转换为 [cx, cy, w, h]
        p_cx = (pbox[:, 0] + pbox[:, 2]) / 2
        p_cy = (pbox[:, 1] + pbox[:, 3]) / 2
        p_w = pbox[:, 2] - pbox[:, 0]
        p_h = pbox[:, 3] - pbox[:, 1]

        g_cx = (gtbox[:, 0] + gtbox[:, 2]) / 2
        g_cy = (gtbox[:, 1] + gtbox[:, 3]) / 2
        g_w = gtbox[:, 2] - gtbox[:, 0]
        g_h = gtbox[:, 3] - gtbox[:, 1]
    else:
        # 已经是中心点格式 [cx, cy, w, h]
        p_cx, p_cy, p_w, p_h = pbox[:, 0], pbox[:, 1], pbox[:, 2], pbox[:, 3]
        g_cx, g_cy, g_w, g_h = gtbox[:, 0], gtbox[:, 1], gtbox[:, 2], gtbox[:, 3]

    # Wasserstein 距离：中心点距离的平方 + 尺寸差异的平方
    center_dist = (p_cx - g_cx) ** 2 + (p_cy - g_cy) ** 2
    size_dist = (p_w - g_w) ** 2 + (p_h - g_h) ** 2

    return center_dist + size_dist

def repulsion_loss_torch(pbox, gtbox, deta=0.5, pnms=0.1, gtnms=0.1, x1x2y1y2=False):#x1x2y1y2=False
    #print('this is the real nwd_rep_loss')
    repgt_loss = 0.0
    repbox_loss = 0.0
    pbox = pbox.detach()
    gtbox = gtbox.detach()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gtbox_cpu = gtbox.cuda().data.cpu().numpy()
    # pgiou = box_iou_v5(pbox, gtbox, x1y1x2y2=x1x2y1y2)
    pgiou = Wasserstein(pbox, gtbox, x1y1x2y2=x1x2y1y2)
    pgiou = pgiou.cuda().data.cpu().numpy()
    # ppiou = box_iou_v5(pbox, pbox, x1y1x2y2=x1x2y1y2)
    ppiou = Wasserstein(pbox, pbox, x1y1x2y2=x1x2y1y2)
    ppiou = ppiou.cuda().data.cpu().numpy()
    # t1 = time.time()
    len = pgiou.shape[0]


    for j in range(len):
        for z in range(j, len):
            ppiou[j, z] = 0
            # if int(torch.sum(gtbox[j] == gtbox[z])) == 4:
            # if int(torch.sum(gtbox_cpu[j] == gtbox_cpu[z])) == 4:
            # if int(np.sum(gtbox_numpy[j] == gtbox_numpy[z])) == 4:
            if (gtbox_cpu[j][0]==gtbox_cpu[z][0]) and (gtbox_cpu[j][1]==gtbox_cpu[z][1]) and (gtbox_cpu[j][2]==gtbox_cpu[z][2]) and (gtbox_cpu[j][3]==gtbox_cpu[z][3]):
                pgiou[j, z] = 0
                pgiou[z, j] = 0
                ppiou[z, j] = 0

    # t2 = time.time()
    # print("for cycle cost time is: ", t2 - t1, "s")
    pgiou = torch.from_numpy(pgiou).cuda().detach()
    ppiou = torch.from_numpy(ppiou).cuda().detach()
    # repgt
    max_iou, argmax_iou = torch.max(pgiou, 1)
    pg_mask = torch.gt(max_iou, gtnms)
    num_repgt = pg_mask.sum()
    if num_repgt > 0:
        iou_pos = pgiou[pg_mask, :]
        max_iou_sec, argmax_iou_sec = torch.max(iou_pos, 1)
        pbox_sec = pbox[pg_mask, :]
        gtbox_sec = gtbox[argmax_iou_sec, :]
        # CCC修改
        #IOG = IoG(gtbox_sec, pbox_sec)
        IOG = Ngaussian_wasserstein_distance(gtbox_sec, pbox_sec)
        repgt_loss = smooth_ln(IOG, deta)
        repgt_loss = repgt_loss.mean()

    # repbox
    pp_mask = torch.gt(ppiou, pnms)
    num_pbox = pp_mask.sum()
    if num_pbox > 0:
        repbox_loss = smooth_ln(ppiou, deta)
        repbox_loss = repbox_loss.mean()
    # mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
    # print(mem)
    torch.cuda.empty_cache()

    return repgt_loss, repbox_loss

class ComputeLoss(nn.Module):
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False, weather_module=None):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = GRL(nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device)))
        BCEobj = GRL(nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device)))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device
        self.weather_module = weather_module
        
        # self.log_sigma_iou = nn.Parameter(torch.tensor(0.0))
        # self.log_sigma_nwd = nn.Parameter(torch.tensor(0.0))
        # self.register_parameter("log_sigma_iou", self.log_sigma_iou)
        # self.register_parameter("log_sigma_nwd", self.log_sigma_nwd)

    def __call__(self, p, targets, images=None):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lbox_iou = torch.zeros(1, device=self.device)  # box loss
        lbox_wd = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Compute weather_scores for each image in the batch
        if self.weather_module is not None and images is not None:
            weather_scores = self.weather_module(images)  # [batch_size, 1], each image has its own score
        else:
            weather_scores = None

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                # sigma_iou_sq = torch.exp(self.log_sigma_iou * 2)
                # sigma_nwd_sq = torch.exp(self.log_sigma_nwd * 2)
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                
                if weather_scores is not None:
                    # Use weather_score for each target based on its image index (b)
                    target_weather_scores = weather_scores[b].squeeze(-1)  # [n] - weather_score for each target
                    wd_loss = wasserstein_loss(pbox, tbox[i])
                    if wd_loss is not None:
                        # Weighted loss: each target uses its own weather_score
                        iou_loss = (1.0 - iou)
                        wd_loss_val = (1.0 - wd_loss)
                        weighted_iou = (1 + 4.5 * target_weather_scores) * iou_loss
                        weighted_wd = (1 - target_weather_scores) * wd_loss_val
                        lbox += (weighted_iou + weighted_wd).mean()
                    else:
                        # If wasserstein_loss returns None, only use IoU loss
                        iou_loss = (1.0 - iou)
                        weighted_iou = (1 + 4.5 * target_weather_scores) * iou_loss
                        lbox += weighted_iou.mean()
                else:
                    # No weather_module, use standard IoU loss
                    lbox += (1.0 - iou).mean()
                # lbox_wd += (1.0 - wasserstein_loss(pbox, tbox[i])).mean()
                # lbox += (1 / (2 * sigma_iou_sq)) * (1.0 - iou).mean() + (1 / (2 * sigma_nwd_sq)) * (1.0 - wasserstein_loss(pbox, tbox[i])).mean() + self.log_sigma_iou + self.log_sigma_nwd

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

