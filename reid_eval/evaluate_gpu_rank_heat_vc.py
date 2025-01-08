# coding=utf-8

import scipy.io
import torch
import numpy as np
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import json
import argparse

parser = argparse.ArgumentParser(description='Ranking')
parser.add_argument('--name', default='test', type=str, help='save model path')
parser.add_argument('--cloth', action='store_true', help='standard setting or cloth-change setting')
parser.add_argument('--mix_cloth', action='store_true', help='standard setting + cloth-change setting')
parser.add_argument('--mat_saved_path', default='pytorch_result.mat', type=str, help='save mat path')

opt = parser.parse_args()

#######################################################################
# Evaluate

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def evaluate(count, qn, gn, qf, ql, qc, gf, gl, gc):
    query = qf.view(-1, 1)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # print("The scores:", score)
    # predict index
    index = np.argsort(score)  # from small to large
    # 得到最相近的index （最不相近--最相近）
    # 得到最相近的index （最相近--最不相近）
    index = index[::-1]
    # print("The predict index range is：", index)

    # good index
    query_index = np.argwhere(gl == ql)
    #     print("gallery laebl=",gl)
    #     print("query laebl=",ql)
    # query_index.shape = (23, 1)
    #     print("query_index=",query_index, query_index.shape)
    # same camera
    camera_index = np.argwhere(gc == qc)
    #     print("gallery cam=",gc)
    #     print("query cam=",qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index = np.argwhere(gl == -1)
    # junk_index2 = np.intersect1d(query_index, camera_index)
    # junk_index = np.append(junk_index2, junk_index1)  # .flatten())

    #     print("good_index=", good_index)
    #     print("camera_index=", camera_index)
    #     print("junk_index2=", junk_index2)

    CMC_tmp = compute_mAP(count, ql, gl, qn, gn, index, qc, query_index, junk_index)
    return CMC_tmp


def evaluate_cloth(count, qn, gn, qf, ql, qc, qcloth, gf, gl, gc, gcloth):
    query = qf.view(-1, 1)
    # print(query.shape)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]

    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)
    # print("query_index", query_index)
    camera_index = np.argwhere(gc == qc)

    cloth_index = np.argwhere(gcloth == qcloth)
    # print("cloth_index", cloth_index)




    # 在ar1中但不在ar2中的已排序的唯一值。
    # good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    # print("good_index", good_index)
    good_index_cloth = np.setdiff1d(query_index, cloth_index, assume_unique=True)
    # print("good_index_cloth", good_index_cloth)


    junk_index1 = np.argwhere(gl == -1)
    # junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index3 = np.intersect1d(query_index, cloth_index)

    # junk_index_temp = np.append(junk_index2, junk_index1)  # .flatten())
    junk_index = np.append(junk_index1, junk_index3)  # .flatten())

    # CMC_tmp = compute_mAP(index, good_index_cloth, junk_index)
    CMC_tmp = compute_mAP(count, ql, gl, qn, gn, index, qc, good_index_cloth, junk_index)
    return CMC_tmp


def evaluate_cloth_new(count, qn, gn, qf, ql, qc, qcloth, gf, gl, gc, gcloth):
    query = qf.view(-1, 1)
    # print(query.shape)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]

    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)
    print("query_index", query_index)
    camera_index = np.argwhere(gc == qc)
    cloth_index = np.argwhere(gcloth == qcloth)
    print("cloth_index", cloth_index)

    # 在ar1中但不在ar2中的已排序的唯一值。
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    print("good_index", good_index)

    # good_index_cloth = np.setdiff1d(good_index, cloth_index, assume_unique=True)
    # print("good_index_cloth", good_index_cloth)

    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index3 = np.intersect1d(query_index, cloth_index)

    junk_index_temp = np.append(junk_index2, junk_index1)  # .flatten())
    junk_index = np.append(junk_index_temp, junk_index3)  # .flatten())

    # CMC_tmp = compute_mAP(index, good_index_cloth, junk_index)
    CMC_tmp = compute_mAP(count, ql, gl, qn, gn, index, qc, good_index, junk_index)
    return CMC_tmp


# def compute_mAP(index, qc, good_index, junk_index):
#     ap = 0
#     cmc = torch.IntTensor(len(index)).zero_()
#     if good_index.size==0:   # if empty
#         cmc[0] = -1
#         return ap,cmc
#
#     # remove junk_index
#     ranked_camera = gallery_cam[index]
#     mask = np.in1d(index, junk_index, invert=True)
#     mask2 = np.in1d(index, np.append(good_index,junk_index), invert=True)
#     index = index[mask]
#     ranked_camera = ranked_camera[mask]
#
#     # find good_index index
#     ngood = len(good_index)
#     mask = np.in1d(index, good_index)
#     # find the location of the rank
#     rows_good = np.argwhere(mask==True)
#     rows_good = rows_good.flatten()
#
#     cmc[rows_good[0]:] = 1
#     for i in range(ngood):
#         d_recall = 1.0/ngood
#         precision = (i+1)*1.0/(rows_good[i]+1)
#         if rows_good[i]!=0:
#             old_precision = i*1.0/rows_good[i]
#         else:
#             old_precision=1.0
#         ap = ap + d_recall*(old_precision + precision)/2
#
#     return ap, cmc


def compute_mAP(count, query_label, gallery_label, query_name, gallery_name, index, qc, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()

    result = dict()

    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc, result

    # remove junk_index
    ranked_camera = gallery_cam[index]
    mask = np.in1d(index, junk_index, invert=True)
    mask2 = np.in1d(index, np.append(good_index, junk_index), invert=True)
    index = index[mask]
    ranked_camera = ranked_camera[mask]

    result['id'] = count
    result['label'] = query_label
    result['image_name'] = query_name[count]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    #     print("index=",index,len(index))
    #     print("good_index=", good_index)
    #     print("mask=",mask[:10],mask.shape)
    rows_good = np.argwhere(mask == True)
    #     print("rows_good", rows_good)
    rows_good = rows_good.flatten()
    #     print("rows_good", rows_good)

    result['top5'] = list(index[:5])
    result['top1'] = index[0]
    result['top10'] = list(index[:10])

    result['rank_name'] = list(gallery_name[index[:10]])
    result['rank_label'] = list(gallery_label[index[:10]])

    min_rank = np.min(rows_good)

    if min_rank < 1:
        result['is_top1'] = 1
    else:
        result['is_top1'] = 0
    if min_rank < 5:
        result['is_top5'] = 1
    else:
        result['is_top5'] = 0
    if min_rank < 10:
        result['is_top10'] = 1
    else:
        result['is_top10'] = 0

    cmc[rows_good[0]:] = 1
    #     print(cmc)
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc, result

######################################################################
result = scipy.io.loadmat(opt.mat_saved_path + '/pytorch_result.mat')
query_feature = torch.FloatTensor(result['query_f'])
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
query_cloth = result['query_cloth'][0]
query_name = result['query_name']

gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]
gallery_cloth = result['gallery_cloth'][0]
gallery_name = result['gallery_name']

# multi = os.path.isfile('multi_query.mat')
# if multi:
#     m_result = scipy.io.loadmat('multi_query.mat')
#     mquery_feature = torch.FloatTensor(m_result['mquery_f'])
#     mquery_cam = m_result['mquery_cam'][0]
#     mquery_label = m_result['mquery_label'][0]
#     mquery_feature = mquery_feature.cuda()
query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()
print(query_feature.shape)

# print("------------standard settings----------------")
# alpha = [0, 0.5, -1]
# for j in range(len(alpha)):
#     CMC = torch.IntTensor(len(gallery_label)).zero_()
#     ap = 0.0
#     results = []
#     for i in range(len(query_label)):
#         qf = query_feature[i].clone()
#         if alpha[j] == -1:
#             qf[0:512] *= 0
#         else:
#             qf[512:1024] *= alpha[j]
#         # ap_tmp, CMC_tmp = evaluate(qf,query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
#         ap_tmp, CMC_tmp, result_tmp = evaluate(i, query_name, gallery_name, qf, query_label[i], query_cam[i],
#                                                gallery_feature, gallery_label, gallery_cam)
#         if CMC_tmp[0]==-1:
#             continue
#         CMC = CMC + CMC_tmp
#         ap += ap_tmp
#         results.append(result_tmp)
#
#     if j == 1:
#         dist_file = opt.mat_saved_path + '/Standard_result_rank.json'
#         results_dict = dict()
#         results_dict["rank_results"] = results
#         # results_dict["cloth_change"] = cloth_change
#
#         with open(dist_file, 'w') as f:
#             print(dist_file, "open success!")
#             json.dump(results_dict, f, cls=NpEncoder)
#             print(dist_file, "dump success!")
#         # print("Results in Alpha", alpha[j], " is save success")
#
#
#
#     CMC = CMC.float()
#     CMC = CMC/len(query_label) #average CMC
#     print('Alpha:%.2f Rank@1:%.4f Rank@5:%.4f Rank@10:%.4f mAP:%.4f'%(alpha[j], CMC[0],CMC[4],CMC[9],ap/len(query_label)))
#
# print("------------cloth changing settings----------------")
# alpha = [0, 0.5, -1]
# for j in range(len(alpha)):
#     CMC = torch.IntTensor(len(gallery_label)).zero_()
#     ap = 0.0
#     results = []
#     for i in range(len(query_label)):
#         qf = query_feature[i].clone()
#         if alpha[j] == -1:
#             qf[0:512] *= 0
#         else:
#             qf[512:1024] *= alpha[j]
#         # ap_tmp, CMC_tmp = evaluate(qf,query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
#         # (count, qn, gn, qf, ql, qc, qcloth, gf, gl, gc, gcloth):
#         ap_tmp, CMC_tmp, result_tmp = evaluate_cloth(i, query_name, gallery_name, qf, query_label[i], query_cam[i], query_cloth[i], gallery_feature, gallery_label, gallery_cam, gallery_cloth)
#         if CMC_tmp[0]==-1:
#             continue
#         CMC = CMC + CMC_tmp
#         ap += ap_tmp
#         results.append(result_tmp)
#
#     if j == 1:
#         dist_file = opt.mat_saved_path + '/Change_result_rank.json'
#         results_dict = dict()
#         results_dict["rank_results"] = results
#         # results_dict["cloth_change"] = cloth_change
#
#         with open(dist_file, 'w') as f:
#             print(dist_file, "open success!")
#             json.dump(results_dict, f, cls=NpEncoder)
#             print(dist_file, "dump success!")
#         # print("Results in Alpha", alpha[j], " is save success")
#
#
#
#     CMC = CMC.float()
#     CMC = CMC/len(query_label) #average CMC
#     print('Alpha:%.2f Rank@1:%.4f Rank@5:%.4f Rank@10:%.4f mAP:%.4f'%(alpha[j], CMC[0],CMC[4],CMC[9],ap/len(query_label)))
#

def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_cloths, g_cloths, max_rank=50):
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
        q_cloth = q_cloths[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        remove_dif_Cloth = (g_pids[order] == q_pid) & (g_cloths[order] != q_cloth)

        remove = remove | remove_dif_Cloth
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

print("------------agw standard settings----------------")
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
    cmc, mAP, mINP = eval_func(distmat, query_label, gallery_label, query_cam, gallery_cam, query_cloth, gallery_cloth)
    print('Alpha:%.2f Rank@1:%.4f Rank@5:%.4f Rank@10:%.4f mAP:%.4f mINP:%.4f'%(alpha[j], cmc[0], cmc[4], cmc[9], mAP, mINP))


print("------------agw cloth changing settings----------------")
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
    cmc, mAP, mINP = eval_func_cloth(distmat, query_label, gallery_label, query_cam, gallery_cam, query_cloth, gallery_cloth)
    print('Alpha:%.2f Rank@1:%.4f Rank@5:%.4f Rank@10:%.4f mAP:%.4f mINP:%.4f'%(alpha[j], cmc[0], cmc[4], cmc[9], mAP, mINP))

