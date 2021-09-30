import paddle
import numpy
import paddle.nn as nn
import paddle.nn.functional as F
from utils.box_utils import match, log_sum_exp
from data import cfg_mnet
GPU = cfg_mnet['gpu_train']

class MultiBoxLoss(nn.Layer):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

        # self.smooth_l1_loss1 = nn.SmoothL1Loss(reduction='sum')
        # self.smooth_l1_loss2 = nn.SmoothL1Loss(reduction='sum')
        # self.cross_entropy_loss1 = nn.CrossEntropyLoss(reduction='sum')


    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: paddle.shape(batch_size,num_priors,num_classes)
                loc shape: paddle.shape(batch_size,num_priors,4)
                priors shape: paddle.shape(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        loc_data, conf_data, landm_data = predictions
        num = loc_data.shape[0]
        num_priors = (priors.shape[0])

        # match priors (default boxes) and ground truth boxes
        loc_t = paddle.ones([num, num_priors, 4])
        landm_t = paddle.ones([num, num_priors, 10])
        conf_t = paddle.ones([num, num_priors], dtype='int32')

        for idx in range(num):
            truths = targets[idx][:, :4].detach()
            labels = targets[idx][:, -1].detach()
            landms = targets[idx][:, 4:14].detach()
            defaults = priors.detach()
            match(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx)
        if GPU:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            landm_t = landm_t.cuda()

        zeros = paddle.to_tensor(0, dtype='int32', stop_gradient=True).cuda()
        # landm Loss (Smooth L1)
        # Shape: [batch,num_priors,10]
        pos1 = conf_t > zeros
        num_pos_landm = pos1.astype('int32').sum(1, keepdim=True)
        N1 = max(num_pos_landm.detach().sum().astype('float32'), 1)
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
        landm_p = paddle.masked_select(landm_data, pos_idx1).reshape([-1, 10])
        # print(landm_p)
        landm_t = paddle.masked_select(landm_t, pos_idx1).reshape([-1, 10])
        # print(landm_t)
        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')
        # print(loss_landm)


        pos = conf_t != zeros
        conf_t[pos] = 1

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = paddle.masked_select(loc_data, pos_idx).reshape([-1, 4])
        # print(loc_p)
        loc_t = paddle.masked_select(loc_t, pos_idx).reshape([-1, 4])
        # print(loc_t)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')
        # print(loss_l)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.reshape([-1, self.num_classes])
        new_conf_t = conf_t.reshape([-1, 1])
        add_conf_t = numpy.ones((new_conf_t.shape[0], 1), dtype=numpy.int32)
        for i in range(len(add_conf_t)):
            add_conf_t[i] = i
        add_conf_t = paddle.to_tensor(add_conf_t, stop_gradient=True)
        new_conf_t = paddle.concat([add_conf_t, new_conf_t], axis=1)
        new_batch_conf = paddle.gather_nd(batch_conf, new_conf_t)
        new_batch_conf = new_batch_conf.reshape([-1, 1])
        # batch_conf = paddle.gather(batch_conf, index=conf_t, axis=1)
        loss_c = log_sum_exp(batch_conf) - new_batch_conf
        # print(loss_c)

        # Hard Negative Mining
        pos = pos.numpy()
        loss_c = loss_c.numpy()
        loss_c[pos.reshape([-1, 1])] = 0 # filter out pos boxes for now
        pos = paddle.to_tensor(pos, stop_gradient=True)
        loss_c = paddle.to_tensor(loss_c, stop_gradient=True)
        loss_c = loss_c.reshape([num, -1])
        # print(loss_c)
        loss_idx = paddle.argsort(loss_c, 1, descending=True)
        idx_rank = paddle.argsort(loss_idx, 1)
        num_pos = pos.astype('int32').sum(1, keepdim=True)
        num_neg = paddle.clip(self.negpos_ratio*num_pos, max=pos.shape[1]-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        new_idx = paddle.to_tensor((pos_idx.numpy()+neg_idx.numpy()).astype('int32'), stop_gradient=True)
        new_idx = new_idx.greater_than(paddle.zeros_like(new_idx, dtype='int32'))
        conf_data = paddle.masked_select(conf_data, new_idx)
        conf_p = conf_data.reshape([-1,self.num_classes])
        pos_neg = paddle.to_tensor((pos.numpy()+neg.numpy()).astype('int32'), stop_gradient=True)
        pos_neg = pos_neg.greater_than(paddle.zeros_like(pos_neg, dtype='int32'))
        targets_weighted = paddle.masked_select(conf_t, pos_neg)
        loss_c = F.cross_entropy(conf_p, targets_weighted.astype('int64'), reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.detach().sum().astype('float32'), 1)
        loss_l /= N
        # print(loss_l)
        loss_c /= N
        # print(loss_c)
        loss_landm /= N1
        # print(loss_landm)
        
        return loss_l, loss_c, loss_landm
