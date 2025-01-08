
import scipy.io
import torch
import numpy as np
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='Ranking')
parser.add_argument('--mat_saved_path', default='pytorch_result.mat', type=str, help='save mat path')
opt = parser.parse_args()
#######################################################################
# Evaluate

def evaluate(qf,ql,qc,gf,gl,gc):
    query = qf.view(-1,1)

    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()

    # predict index
    index = np.argsort(score)  #from small to large
    # index = [20,332,42,1,345....]
    index = index[::-1]
    # good index
    # ql=1 gl=[1,1,1,1,2,2,2,3,3,3,......]  query_index=0,1,2,3
    query_index = np.argwhere(gl==ql)
    # same camera
    # qc=0 gc=222222 # no overlap
    camera_index = np.argwhere(gc==qc)

    # in query_index but not in camera_index, and sorted
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    
    CMC_tmp = compute_mAP(index, qc, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, qc, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    ranked_camera = gallery_cam[index]
    mask = np.in1d(index, junk_index, invert=True)
    mask2 = np.in1d(index, np.append(good_index,junk_index), invert=True)
    index = index[mask]
    ranked_camera = ranked_camera[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    # find the location of the rank
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc



def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        pos_idx = np.where(orig_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx]/ (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)

    return all_cmc, mAP, mINP

def eval_func_cloth(distmat, q_pids, g_pids, q_camids, g_camids, q_cloids, g_cloids, max_rank=50):
    """
        Evaluation with market1501 metric, Plus cloth-remove
        Key: for each query identity, its gallery images from the same camera view and with same cloth are discarded.
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid and cloid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        q_cloid = q_cloids[q_idx]

        # remove gallery samples that have the same pid and camid and cloid with query
        order = indices[q_idx]

        # True or False
        remove_cam = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        remove_cloth = (g_pids[order] == q_pid) & (g_cloids[order] == q_cloid)

        remove = remove_cam | remove_cloth
        # reverse
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        pos_idx = np.where(orig_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx]/ (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)

    return all_cmc, mAP, mINP


######################################################################
result = scipy.io.loadmat(opt.mat_saved_path + '/pytorch_result.mat')
query_feature = torch.FloatTensor(result['query_f'])
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

print(query_feature.shape)


# print("------------origin standard settings----------------")

# alpha = [0, 0.5, -1]
# for j in range(len(alpha)):
#     CMC = torch.IntTensor(len(gallery_label)).zero_()
#     ap = 0.0
#     for i in range(len(query_label)):
#         qf = query_feature[i].clone()
#         if alpha[j] == -1:
#             qf[0:512] *= 0
#         else:
#             qf[512:1024] *= alpha[j]
#         ap_tmp, CMC_tmp = evaluate(qf,query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
#         if CMC_tmp[0]==-1:
#             continue
#         CMC = CMC + CMC_tmp
#         ap += ap_tmp
#
#     CMC = CMC.float()
#     CMC = CMC/len(query_label) #average CMC
#     print('Alpha:%.2f Rank@1:%.4f Rank@5:%.4f Rank@10:%.4f mAP:%.4f'%(alpha[j], CMC[0],CMC[4],CMC[9],ap/len(query_label)))

print("------------AGW standard settings----------------")
alpha = [0, 0.5, 0.6, 0.7, 0.8, -1]
for j in range(len(alpha)):

    qf = query_feature.clone()
    gf = gallery_feature.clone()

    if alpha[j] == -1:
        qf[:, 0:512] *= 0
    else:
        qf[:, 512:1024] *= alpha[j]

    m, n = qf.shape[0], gf.shape[0]
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.cpu().numpy()
    cmc, mAP, mINP = eval_func(distmat, query_label, gallery_label, query_cam, gallery_cam)
    print('Alpha:%.2f Rank@1:%.4f Rank@5:%.4f Rank@10:%.4f mAP:%.4f mINP:%.4f'%(alpha[j], cmc[0], cmc[4], cmc[9], mAP, mINP))
