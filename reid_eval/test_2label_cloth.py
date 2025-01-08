

from __future__ import print_function, division

import sys
sys.path.append('..')
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
from reIDmodel import ACID
import glob
import shutil
from shutil import copyfile
import random
import cv2


######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default=90000, type=int, help='80000')
parser.add_argument('--test_dir',default='../../Market/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='test', type=str, help='save model path')
parser.add_argument('--batchsize', default=80, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', action='store_true', help='use PCB' )
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--cloth', action='store_true', help='standard setting or cloth-change setting' )
parser.add_argument('--mix_cloth', action='store_true', help='standard setting + cloth-change setting' )
parser.add_argument('--rerank', action='store_true', help='use re-ranking' )
parser.add_argument('--single_shot', action='store_true', help='single shot matching setting')
parser.add_argument('--ibn', action='store_true', help='use ibn.' )
parser.add_argument('--ltcc', action='store_true', help='use ltcc dataset.')
parser.add_argument('--vc', action='store_true', help='use vc-cloth dataset.')
parser.add_argument('--celeb', action='store_true', help='use celeb_reid dataset.')
parser.add_argument('--celebL', action='store_true', help='use celeb_light_reid dataset.')
parser.add_argument('--seed', default=1, type=int, help='random seed')

opt = parser.parse_args()

str_ids = opt.gpu_ids.split(',')
which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])

###load config###
root_path = "../"
if opt.ltcc:
    model_weight_path = root_path + 'ltcc_outputs/outputs'
elif opt.vc:
    model_weight_path = root_path + 'vc_outputs/outputs'
elif opt.celeb:
    model_weight_path = root_path + 'celeb_outputs/outputs'
elif opt.celebL:
    model_weight_path = root_path + 'celebL_outputs/outputs'
else:
    model_weight_path = root_path + 'outputs'

###load config###

config_path = os.path.join(model_weight_path, name, 'config.yaml')


with open(config_path, 'r') as stream:
    config = yaml.load(stream)

if "dataset" not in config:
    config['dataset'] = "prcc"

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if opt.PCB:
    data_transforms = transforms.Compose([
        transforms.Resize((384,192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])

def get_single_shot_folder(gallery_path, seed):

    print("-----Using Seed", seed, "-------------")
    random.seed(seed)
    gallery_save_path = gallery_path[:-1] + '/Single_A'

    if not os.path.isdir(gallery_save_path):
        os.mkdir(gallery_save_path)
    else:
        print("Re-build Single Folder")
        shutil.rmtree(gallery_save_path)
        os.mkdir(gallery_save_path)


    gallery_img_path = glob.glob(gallery_path+'/*/*.jpg')
    random.shuffle(gallery_img_path)

    keeped_label = []
    for img_path in gallery_img_path:
        filename = img_path.split('/')[-1]
        label = img_path.split('/')[-2]

        if label not in keeped_label:

            keeped_label.append(label)
            dst_path = gallery_save_path + '/' + label

            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)

            copyfile(img_path, dst_path + '/' + filename)
        else:
            continue
    print("Single-Shot Folder Build Success")
    return gallery_save_path



data_dir = test_dir
image_datasets = {}
if opt.cloth:
    if config['dataset'] == "prcc":
        print("--------PRCC Cloth_Change Setting--------")
        image_datasets['query'] = datasets.ImageFolder(os.path.join(data_dir, 'test', 'C'), data_transforms)
        image_datasets['gallery'] = datasets.ImageFolder(os.path.join(data_dir, 'test', 'A'), data_transforms)
    elif config['dataset'] == "vc" or config['dataset'] == "ltcc":
        print("------------", config['dataset'], "Setting--------")
        image_datasets['query'] = datasets.ImageFolder(os.path.join(data_dir, 'query_new'), data_transforms)
        image_datasets['gallery'] = datasets.ImageFolder(os.path.join(data_dir, 'gallery_new'), data_transforms)
    else:
        print("------------", config['dataset'], "Setting--------")
        image_datasets['query'] = datasets.ImageFolder(os.path.join(data_dir, 'query'), data_transforms)
        image_datasets['gallery'] = datasets.ImageFolder(os.path.join(data_dir, 'gallery'), data_transforms)
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                  shuffle=False, num_workers=16) for x in ['gallery', 'query']}
elif opt.mix_cloth:
    print("--------Mix_Cloth_Change Setting--------")
    image_datasets['query'] = datasets.ImageFolder(os.path.join(data_dir, 'test', 'BC'), data_transforms)
    image_datasets['gallery'] = datasets.ImageFolder(os.path.join(data_dir, 'test', 'A'), data_transforms)
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                  shuffle=False, num_workers=16) for x in ['gallery', 'query']}
elif opt.single_shot:
    print("--------Single-Shot Setting--------")
    gallery_path_new = get_single_shot_folder(os.path.join(data_dir, 'test', 'A'), opt.seed)
    image_datasets['query'] = datasets.ImageFolder(os.path.join(data_dir, 'test', 'C'), data_transforms)
    image_datasets['gallery'] = datasets.ImageFolder(gallery_path_new, data_transforms)
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery', 'query']}

else:
    if config['dataset'] == "vc" or config['dataset'] == "ltcc":
        print("--------", config['dataset'], "Standard Setting--------")
        image_datasets['query'] = datasets.ImageFolder(os.path.join(data_dir, 'query'), data_transforms)
        image_datasets['gallery'] = datasets.ImageFolder(os.path.join(data_dir, 'gallery'), data_transforms)
    else:
        print("--------PRCC Standard Setting--------")
        image_datasets['query'] = datasets.ImageFolder(os.path.join(data_dir, 'test', 'B'), data_transforms)
        image_datasets['gallery'] = datasets.ImageFolder(os.path.join(data_dir, 'test', 'A'), data_transforms)
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery', 'query']}

######################################################################
# Load model
#---------------------------
def load_network(network):

    save_path = os.path.join(model_weight_path, name, 'checkpoints/id_%08d.pt' % opt.which_epoch)
    state_dict = torch.load(save_path)
    network.load_state_dict(state_dict['a'], strict=False)

    print ("Using Model -------", name, "-----------------")
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def norm(f):
    f = f.squeeze()
    fnorm = torch.norm(f, p=2, dim=1, keepdim=True)
    f = f.div(fnorm.expand_as(f))
    return f

def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        if opt.use_dense:
            ff = torch.FloatTensor(n,1024).zero_()
        else:
            ff = torch.FloatTensor(n,1024).zero_()
        if opt.PCB:
            ff = torch.FloatTensor(n, 2048, 6).zero_() # we have six parts
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            if config['ID_style'] == 'ACID':
                f, x, x1, x2 = model(input_img, label, False)
            else:
                f, x = model(input_img)
            '''
            f.shape= torch.Size([80, 8192])
            x1.shape= torch.Size([80, 512])
            x2.shape= torch.Size([80, 512])
            new_f.shape= torch.Size([80, 1024])
            '''
            # print("f.shape=", f.shape)
            # print("x1.shape=", x[0].shape)
            # print("x2.shape=", x[1].shape)
            x[0] = norm(x[0])
            x[1] = norm(x[1])
            f = torch.cat((x[0], x[1]), dim=1) #use 512-dim feature
            # print("new_f.shape=", f.shape)
            f = f.data.cpu()
            ff = ff+f

        # print("ff.shape=", ff.shape)
        ff[:, 0:512] = norm(ff[:, 0:512])
        ff[:, 512:1024] = norm(ff[:, 512:1024])

        # norm feature
        if opt.PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6) 
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)

        features = torch.cat((features,ff), 0)
    return features

def get_id(img_path):
    camera_id = []
    labels = []
    cam2label = {'A': 0, 'B': 1, 'C': 2}
    filenames = []

    for path, v in img_path:
        filename = path.split('/')[-1]
        # filename = os.path.basename(path)
        label = path.split('/')[-2]

        camera = cam2label[filename[0]]

        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))

        camera_id.append(int(camera))
        filenames.append(filename)

    return camera_id, labels, filenames

def get_id_ltcc(img_path):
    camera_id = []
    labels = []
    # cam2label = {'A': 0, 'B': 1, 'C': 2}
    filenames = []
    cloth_id = []

    for path, v in img_path:
        filename = path.split('/')[-1]
        # filename = 1_c11_015833.png
        label = path.split('/')[-2]
        cloth = filename.split('_')[0]
        camera = filename.split('_')[1][1:]

        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))

        camera_id.append(int(camera))
        cloth_id.append(int(cloth))
        filenames.append(filename)

    return cloth_id, camera_id, labels, filenames

def get_id_celeb(img_path):
    camera_id = []
    labels = []
    # cam2label = {'A': 0, 'B': 1, 'C': 2}
    filenames = []
    cloth_id = []

    for path, v in img_path:
        filename = path.split('/')[-1]
        # filename = 1_c11_015833.png
        label = path.split('/')[-2]
        cloth = filename.split('_')[0]
        camera = cloth

        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))

        camera_id.append(int(camera))
        cloth_id.append(int(cloth))
        filenames.append(filename)

    return cloth_id, camera_id, labels, filenames

# def get_id(img_path, dataset):
#     camera_id = []
#     labels = []
#     # cam2label = {'A': 0, 'B': 1, 'C': 2}
#     filenames = []
#     cloth_id = []
#
#     for path, v in img_path:
#         filename = path.split('/')[-1]
#         # filename = os.path.basename(path)
#         label = path.split('/')[-2]
#
#         camera = filename.split('_')[1][1:]
#         cloth = filename[0]
#
#         if label[0:2]=='-1':
#             labels.append(-1)
#         else:
#             labels.append(int(label))
#
#         camera_id.append(int(camera))
#         filenames.append(filename)
#         cloth_id.append(int(cloth))
#
#     return camera_id, labels, filenames, cloth_id

def heatmap_rainbow(img, heatmap, path):

    img_new = cv2.imread(img)

    print("heatmap.shape = ", heatmap.shape)
    # shape= (16,8)
    heatmap = cv2.resize(heatmap, (img_new.shape[1], img_new.shape[0]), interpolation=cv2.INTER_LINEAR)

    # print("heatmap_new.shape = ", heatmap.shape)
    # shape= (16,8)

    heatmap = heatmap/np.max(heatmap)

    # must convert to type unit8
    heatmap = np.uint8(255 * heatmap)
    print("heatmap_new.shape = ", heatmap.shape)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)


    # print("img.shape =", img_new.shape)

    superimposed_img = heatmap*0.4+img_new

    imgs = np.hstack([img_new, superimposed_img])

    cv2.imwrite(path+".jpg", superimposed_img)


gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs
# mquery_path = image_datasets['multi-query'].imgs

if config['dataset'] == "prcc":
    gallery_cam, gallery_label, gallery_name = get_id(gallery_path)
    query_cam, query_label, query_name = get_id(query_path)
elif config['dataset'] == "ltcc" or config['dataset'] == "vc":
    gallery_cloth, gallery_cam, gallery_label, gallery_name = get_id_ltcc(gallery_path)
    query_cloth, query_cam, query_label, query_name = get_id_ltcc(query_path)
elif config['dataset'] == "celeb" or config['dataset'] == "celeb_light":
    gallery_cloth, gallery_cam, gallery_label, gallery_name = get_id_celeb(gallery_path)
    query_cloth, query_cam, query_label, query_name = get_id_celeb(query_path)
else:
    print("Undefined", config['dataset'], ", please checke the CONFIG file")
    raise RuntimeError


class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()

######################################################################
# Load Collected data Trained model
print('-------test-----------')
if config['ID_style'] == 'ACID':
    global_att = True
else:
    raise RuntimeError ("Not Supported ID Style" + config['ID_style'])

print("-------Using AICD model-------")
model_structure = ACID(config['ID_class'], norm=config['norm_id'], stride=config['ID_stride'],
                           pool=config['pool'], ibn=opt.ibn, ca_layer=config['CA_layer'], using_global=global_att)

model = load_network(model_structure)

# Remove the final fc layer and classifier layer
model.model.fc = nn.Sequential()
model.classifier1.classifier = nn.Sequential()
model.classifier2.classifier = nn.Sequential()

if config['ID_style'] == 'ACID':
    model.classifier = nn.Sequential()
    model.classifier_bap = nn.Sequential()
    model.classifierCA.classifier = nn.Sequential()


# print(model)


# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()

# Extract feature
since = time.time()
with torch.no_grad():
    gallery_feature = extract_feature(model, dataloaders['gallery'])
    query_feature = extract_feature(model, dataloaders['query'])
    time_elapsed = time.time() - since
    print('Extract features complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # if opt.multi:
    #     mquery_feature = extract_feature(model,dataloaders['multi-query'])
    
# Save to Matlab for check
if config['dataset'] == "prcc":
    result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(), 'query_label':query_label, 'query_cam':query_cam, 'query_name':query_name, 'gallery_name':gallery_name}
# elif config['dataset'] == "ltcc":
else:
    result = {'gallery_cloth': gallery_cloth, 'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label, 'gallery_cam': gallery_cam, 'query_cloth': query_cloth, 'query_f': query_feature.numpy(), 'query_label': query_label, 'query_cam': query_cam, 'query_name': query_name, 'gallery_name': gallery_name}

save_path = os.path.join(model_weight_path, name)
scipy.io.savemat(save_path + "/pytorch_result.mat", result)


# count = 0
# for data in dataloaders['query']:
#     img, label = data
#     with torch.no_grad():
#         x = model.model.conv1(img.cuda())
#         x = model.model.bn1(x)
#         x = model.model.relu(x)
#         x = model.model.maxpool(x)
#
#         output_0 = model.model.layer1(x)
#
#         output_1 = model.model.layer2(output_0)
#
#         output_2 = model.model.layer3(output_1)
#
#         output_3 = model.model.layer4(output_2)
#
#     # (1,1024,16,8)
#     # output.shape= torch.Size([1, 512, 32, 16])
#     # heatmap.shape= (32, 16)
#
#     print("output0.shape=", output_0.shape)
#
#     heatmap_0 = output_0.squeeze().sum(dim=0).cpu().numpy()
#     heatmap_1 = output_1.squeeze().sum(dim=0).cpu().numpy()
#     heatmap_2 = output_2.squeeze().sum(dim=0).cpu().numpy()
#     heatmap_3 = output_3.squeeze().sum(dim=0).cpu().numpy()
#
#     #(16,8)
#     # print("heatmap.shape=", heatmap_1.shape)
#
#     # Result is saved tas `heatmap.png`
#
#     saved_path = "../heat_result/prcc_map_test/C/" + query_path[count][0].split('/')[-2]
#     if not os.path.exists(saved_path):
#         os.makedirs(saved_path)
#
#     saved_path = saved_path + '/' + query_path[count][0].split('/')[-1][:-4]
#
#     heatmap_rainbow(query_path[count][0], heatmap_0, saved_path+'0')
#     # heatmap_rainbow(query_path[count][0], heatmap_1, saved_path)
#     heatmap_rainbow(query_path[count][0], heatmap_1, saved_path+'1')
#     heatmap_rainbow(query_path[count][0], heatmap_2, saved_path+'2')
#     heatmap_rainbow(query_path[count][0], heatmap_3, saved_path+'3')
#     count = count+1
#
#     if count == 3:
#         break
# print("Heatmaps draw success")

if opt.rerank:
    print("---------Use Re-ranking---------")
    os.system('python evaluate_rerank.py')
elif opt.single_shot:
    os.system('python evaluate_gpu.py --mat_saved_path ' + save_path)
elif opt.cloth:
    if config['dataset'] == "prcc":
        command = 'python evaluate_gpu_rank_heat.py --cloth --name ' + opt.name + ' --mat_saved_path ' + save_path
    # elif config['dataset'] == "ltcc":
    elif config['dataset'] == "vc":
        command = 'python evaluate_gpu_rank_heat_vc.py --cloth --name ' + opt.name + ' --mat_saved_path ' + save_path
    else: # evaluate_agwsetting_ltcc  # evaluate_gpu_rank_heat_ltcc # evaluate_gpu_rank_heat_for_cloth
        command = 'python evaluate_agwsetting_ltcc.py --cloth --name ' + opt.name + ' --mat_saved_path ' + save_path
    os.system(command)
elif opt.mix_cloth:
    command = 'python evaluate_gpu_rank_heat.py --mix_cloth --name ' + opt.name + ' --mat_saved_path ' + save_path
    os.system(command)
else:
    if config['dataset'] == "vc":
        command = 'python evaluate_gpu_rank_heat_vc.py --cloth --name ' + opt.name + ' --mat_saved_path ' + save_path
    elif config['dataset'] == "ltcc":
        command = 'python evaluate_gpu_rank_heat_ltcc.py --cloth --name ' + opt.name + ' --mat_saved_path ' + save_path
    else:
        # standard
        command = 'python evaluate_gpu_rank_heat.py --name ' + opt.name + ' --mat_saved_path ' + save_path
    os.system(command)
