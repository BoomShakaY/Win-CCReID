# coding=utf-8

import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F

import math

import torch
from non_local import Non_local

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('InstanceNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier_new(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim # We remove the input_dim

        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck, affine=True)]

        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]

        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            # feature, score
            return [f, x]
            # return x
        else:
            x = self.classifier(x)
            return x

# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num, norm=False, pool='avg', stride=2):
        super(ft_net, self).__init__()
        if norm:
            self.norm = True
        else:
            self.norm = False
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        self.part = 4
        if pool=='max':
            model_ft.partpool = nn.AdaptiveMaxPool2d((self.part,1)) 
            model_ft.avgpool = nn.AdaptiveMaxPool2d((1,1))
        else:
            model_ft.partpool = nn.AdaptiveAvgPool2d((self.part,1)) 
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # remove the final downsample
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)

        self.model = model_ft   
        self.classifier = ClassBlock(2048, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)  # -> 512 32*16
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        f = self.model.partpool(x) # 8 * 2048 4*1
        x = self.model.avgpool(x)  # 8 * 2048 1*1
        
        x = x.view(x.size(0), x.size(1))
        f = f.view(f.size(0), f.size(1)*self.part)
        if self.norm:
            fnorm = torch.norm(f, p=2, dim=1, keepdim=True) + 1e-8
            f = f.div(fnorm.expand_as(f))
        x = self.classifier(x)
        return f, x

# Define the AB Model
class ft_netAB(nn.Module):

    def __init__(self, class_num, norm=False, stride=2, droprate=0.5, pool='avg', circle=False, ibn=False):
        super(ft_netAB, self).__init__()

        if ibn == True:
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
        else:
            model_ft = models.resnet50(pretrained=True)

        self.part = 4
        if pool=='max':
            model_ft.partpool = nn.AdaptiveMaxPool2d((self.part,1))
            model_ft.avgpool = nn.AdaptiveMaxPool2d((1,1))
        else:
            model_ft.partpool = nn.AdaptiveAvgPool2d((self.part,1))
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.model = model_ft
        self.circle = circle

        if stride == 1:
            self.model.layer4[0].downsample[0].stride = (1,1)
            self.model.layer4[0].conv2.stride = (1,1)

        self.classifier1 = ClassBlock(2048, class_num, 0.5, return_f = circle)
        self.classifier2 = ClassBlock(2048, class_num, 0.75, return_f = circle)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        f = self.model.partpool(x)
        f = f.view(f.size(0),f.size(1)*self.part)
        f = f.detach() # no gradient 
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x1 = self.classifier1(x) 
        x2 = self.classifier2(x)  
        x=[]
        x.append(x1)
        x.append(x2)

        # feature, [score1[feature, score], score2[feature, score]]
        return f, x


# Define the DenseNet121-based Model
class ft_net_dense(nn.Module):

    def __init__(self, class_num ):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 1024 
        self.classifier = ClassBlock(1024, class_num)

    def forward(self, x):
        x = self.model.features(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x
    
# Define the ResNet50-based Model (Middle-Concat)
# In the spirit of "The Devil is in the Middle: Exploiting Mid-level Representations for Cross-Domain Instance Matching." Yu, Qian, et al. arXiv:1711.08106 (2017).
class ft_net_middle(nn.Module):

    def __init__(self, class_num ):
        super(ft_net_middle, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048+1024, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        # x0  n*1024*1*1
        x0 = self.model.avgpool(x)
        x = self.model.layer4(x)
        # x1  n*2048*1*1
        x1 = self.model.avgpool(x)
        x = torch.cat((x0,x1),1)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x

# Part Model proposed in Yifan Sun etal. (2018)
class PCB(nn.Module):
    def __init__(self, class_num ):
        super(PCB, self).__init__()

        self.part = 4 # We cut the pool5 to 4 parts
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        self.softmax = nn.Softmax(dim=1)
        # define 4 classifiers
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(2048, class_num, True, False, 256))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        f = x
        f = f.view(f.size(0),f.size(1)*self.part)
        x = self.dropout(x)
        part = {}
        predict = {}
        # get part feature batchsize*2048*4
        for i in range(self.part):
            part[i] = x[:,:,i].contiguous()
            part[i] = part[i].view(x.size(0), x.size(1))
            name = 'classifier'+str(i)
            c = getattr(self,name)
            predict[i] = c(part[i])

        y=[]
        for i in range(self.part):
            y.append(predict[i])

        return f, y

class PCB_test(nn.Module):
    def __init__(self,model):
        super(PCB_test,self).__init__()
        self.part = 6
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        # remove the final downsample
        self.model.layer3[0].downsample[0].stride = (1,1)
        self.model.layer3[0].conv2.stride = (1,1)

        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        y = x.view(x.size(0),x.size(1),x.size(2))
        return y




class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return  y

EPSILON = 1e-12

class BAP(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions, counterfactual=False):

        random_uniform = True
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)

        # sign-sqrt
        feature_matrix_raw = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix_raw, dim=-1)

        if counterfactual:
            if self.training:
                if random_uniform == True:
                    fake_att = torch.zeros_like(attentions).uniform_(-1, 1)
                elif random_uniform == False:
                    assert attentions.dim() == 4
                    fake_att = torch.normal(mean=0., std=1., size=(attentions.size(0),attentions.size(1),attentions.size(2),attentions.size(3)))
                    fake_att = fake_att.cuda()
            else:
                fake_att = torch.ones_like(attentions)
            # mean_feature = features.mean(3).mean(2).view(B, 1, C)
            # counterfactual_feature = mean_feature.expand(B, M, C).contiguous().view(B, -1)
            counterfactual_feature = (torch.einsum('imjk,injk->imn', (fake_att, features)) / float(H * W)).view(B, -1)

            counterfactual_feature = torch.sign(counterfactual_feature) * torch.sqrt(torch.abs(counterfactual_feature) + EPSILON)

            counterfactual_feature = F.normalize(counterfactual_feature, dim=-1)
            return feature_matrix, counterfactual_feature
        else:
            return feature_matrix


class MultiHeadAtt(nn.Module):
    """
    Extend the channel attention into MultiHeadAtt.
    It is modified from "Zhang H, Wu C, Zhang Z, et al. Resnest: Split-attention networks."
    """

    def __init__(self, in_channels, channels,
                 radix=4, reduction_factor=4,
                 rectify=False, norm_layer=nn.BatchNorm2d):
        super(MultiHeadAtt, self).__init__()

        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.channels = channels

        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups=1)
        self.bn1 = norm_layer(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, channels * radix, 1, groups=1)

    def forward(self, x):
        batch, channel = x.shape[:2]
        splited = torch.split(x, channel // self.radix, dim=1)
        gap = sum(splited)
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap).view((batch, self.radix, self.channels))
        atten = F.softmax(atten, dim=1).view(batch, -1, 1, 1)
        atten = torch.split(atten, channel // self.radix, dim=1)

        out = torch.cat([att * split for (att, split) in zip(atten, splited)], 1)
        return out.contiguous()


class BN2d(nn.Module):
    def __init__(self, planes):
        super(BN2d, self).__init__()
        self.bottleneck2 = nn.BatchNorm2d(planes)
        self.bottleneck2.bias.requires_grad_(False)  # no shift
        self.bottleneck2.apply(weights_init_kaiming)

    def forward(self, x):
        return self.bottleneck2(x)

# Define the ACID Model
class ACID(nn.Module):

    def __init__(self, class_num, norm=False, stride=2, droprate=0.5, pool='avg', ca_layer=0, circle=False, ibn=False,
                 soft=False, th=0, using_global=True):
        super(ACID, self).__init__()

        self.using_global = using_global
        self.CA_layer = ca_layer
        self.class_num = class_num
        self.gt = []
        self.count_iter = 0
        self.soft = soft
        self.th = th
        self.radix = 2
        self.in_planes = 2048

        if ibn == True:
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
        else:
            model_ft = models.resnet50(pretrained=True)

        self.part = 4
        if pool == 'max':
            model_ft.partpool = nn.AdaptiveMaxPool2d((self.part, 1))
            model_ft.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            model_ft.partpool = nn.AdaptiveAvgPool2d((self.part, 1))
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        """
        x.shape= torch.Size([8, 3, 384, 192])
        base_1.shape= torch.Size([8, 64, 96, 48])
        att_s1.shape= torch.Size([8, 64, 96, 48])
        BN1.shape= torch.Size([8, 64, 96, 48])
        att1.shape= torch.Size([8, 64, 1, 1])
        x_att1.shape= torch.Size([8, 64, 96, 48])
        """
        self.base_1 = nn.Sequential(*list(model_ft.children())[0:3])
        self.BN1 = BN2d(64)
        self.att1 = SELayer(64, 8)
        self.att_s1 = MultiHeadAtt(64, int(64 / self.radix), radix=self.radix)

        self.base_2 = nn.Sequential(*list(model_ft.children())[3:4])
        self.BN2 = BN2d(256)
        self.att2 = SELayer(256,32)
        self.att_s2 = MultiHeadAtt(256,int(256/self.radix),radix=self.radix)

        self.base_3 = nn.Sequential(*list(model_ft.children())[4:5])
        self.BN3 = BN2d(512)
        self.att3 = SELayer(512,64)
        self.att_s3 = MultiHeadAtt(512,int(512/self.radix),radix=self.radix)

        self.base_4 = nn.Sequential(*list(model_ft.children())[5:6])
        self.BN4 = BN2d(1024)
        self.att4 = SELayer(1024,128)
        self.att_s4 = MultiHeadAtt(1024,int(1024/self.radix),radix=self.radix)

        self.base_5 = nn.Sequential(*list(model_ft.children())[6:])
        self.BN5 = BN2d(2048)
        self.att5 = SELayer(2048,256)
        self.att_s5 = MultiHeadAtt(2048,int(2048/self.radix),radix=self.radix)

        self.M = 8
        self.attentions = BasicConv2d(2048, self.M, kernel_size=1)
        self.bap = BAP(pool='GAP')

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)

        self.classifier = nn.Linear(self.in_planes, self.class_num, bias=False)
        self.classifier_bap = nn.Linear(self.in_planes * self.M, self.in_planes, bias=False)

        self.classifier.apply(weights_init_classifier_new)
        self.classifier_bap.apply(weights_init_classifier_new)

        self.model = model_ft
        self.circle = circle

        if stride == 1:
            self.model.layer4[0].downsample[0].stride = (1, 1)
            self.model.layer4[0].conv2.stride = (1, 1)

        self.classifier1 = ClassBlock(2048, class_num, 0.5, return_f=circle)
        self.classifier2 = ClassBlock(2048, class_num, 0.75, return_f=circle)

        if self.CA_layer == 1:
            input_dim = 256
        elif self.CA_layer == 2:
            input_dim = 512
        elif self.CA_layer == 3:
            input_dim = 1024
        elif self.CA_layer == 4:
            input_dim = 2048
        else:
            print("Wrong CA_layer:", self.CA_layer, "which is out of range")
            raise RuntimeError

        self.classifierCA = ClassBlock(input_dim, class_num, 0.5, return_f=circle)

    def competitive_attention(self, x):
        # input= ([8, 2048, 16, 8])
        if self.training:
            self.eval()

            self.count_iter = self.count_iter + 1
            if self.count_iter < 82000 * 8:
                th_l = 64
                th_s = 16
                fg_drop_per = 2
                bg_drop_per = fg_drop_per * 3
            else:
                if self.count_iter == 82000 * 8:
                    print("------Change to bigger thresholds------")
                th_l = 75
                th_s = 16
                fg_drop_per = 2
                bg_drop_per = fg_drop_per * 3

            bbox_feats_new = x
            bbox_feats_new = Variable(bbox_feats_new.data, requires_grad=True)

            bbox_feats_tmp = self.model.avgpool(bbox_feats_new)
            bbox_feats_tmp = bbox_feats_tmp.view(bbox_feats_tmp.size(0), bbox_feats_tmp.size(1))  # (8,2048)

            # (batch_size)
            num_batch = bbox_feats_new.shape[0]
            num_channel = bbox_feats_new.shape[1]
            width = bbox_feats_new.shape[2]
            height = bbox_feats_new.shape[3]

            output_score = self.classifierCA(bbox_feats_tmp)
            # shape = （8,150）
            index = self.gt

            one_hot = torch.zeros((1), dtype=torch.float32).cuda()
            one_hot = Variable(one_hot, requires_grad=False)

            sp_i = torch.ones([2, num_batch]).long()
            sp_i[0, :] = torch.arange(num_batch)
            sp_i[1, :] = index
            sp_v = torch.ones([num_batch])

            one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v,
                                                      torch.Size([num_batch, self.class_num])).to_dense().cuda()
            one_hot_sparse = Variable(one_hot_sparse, requires_grad=False)  # 8,150

            one_hot = torch.sum(output_score * one_hot_sparse)

            # self.zero_grad()
            self.classifierCA.zero_grad()
            one_hot.backward()

            grads_val = bbox_feats_new.grad.clone().detach()
            # print("grads_val", grads_val.shape)

            grad_channel_mean = torch.mean(grads_val.view(num_batch, num_channel, -1), dim=2)
            # print("grad_channel_mean", grad_channel_mean.shape)

            grad_channel_mean = grad_channel_mean.view(num_batch, num_channel, 1, 1)
            # print("grad_channel_mean", grad_channel_mean.shape)

            att_all = torch.sum(bbox_feats_new * grad_channel_mean, 1)
            # print("att_all", att_all.shape)

            att_all = att_all.view(num_batch, -1)

            mask_shape = att_all.shape[1]

            assert mask_shape == width * height

            # self.zero_grad()
            self.classifierCA.zero_grad()

            # -------------------------CA ----------------------------
            if self.CA_layer != 4:
                th_l = int(mask_shape / 2)

            thl_mask_value = torch.sort(att_all, dim=1, descending=True)[0][:, th_l]  # torch.Size([8]
            '''
            thl_mask_value tensor([ 0.0005, -0.0003,  0.0004,  0.0005, -0.0001,  0.0007,  0.0007,  0.0005],
            '''
            # print("thl_mask_value", thl_mask_value)

            thl_mask_value = thl_mask_value.view(num_batch, 1).expand(num_batch, mask_shape)  # torch.Size([8, 150])
            # print("thl_mask_value", thl_mask_value, thl_mask_value.shape)

            mask_all_cuda = torch.where(att_all > thl_mask_value, torch.zeros(att_all.shape).cuda(),
                                        torch.ones(att_all.shape).cuda())  # (8,128) 1,和0的矩阵，对于梯度来说的mask至此算完

            # ------------------------ something different ---------------------
            # not sure keep it or not

            # mask_all = mask_all_cuda.detach().cpu().numpy()
            # mask_all_new = np.ones((num_batch, mask_shape), dtype=np.float32)
            #
            # for q in keep_inds:
            #     mask_all_temp = np.ones((mask_shape), dtype=np.float32)
            #
            #     zero_index = np.where(mask_all[q, :] == 0)[0]
            #
            #     num_zero_index = zero_index.size
            #
            #     if num_zero_index >= th_s:
            #         dumy_index = npr.choice(zero_index, size=th_s, replace=False)
            #     else:
            #         zero_index = np.arange(mask_shape)
            #         dumy_index = npr.choice(zero_index, size=th_s, replace=False)
            #
            #     mask_all_temp[dumy_index] = 0
            #     mask_all_new[q, :] = mask_all_temp
            #
            # mask_all = torch.from_numpy(mask_all_new.reshape(num_batch, width, height)).cuda()
            # mask_all = mask_all.view(num_batch, 1, width, height)

            # ------------------------if not use the above block, use this -----------------------------
            mask_all = mask_all_cuda.reshape(num_batch, width, height).view(num_batch, 1, width, height)

            # ------------------------ batch ---------------------
            # （8,2048,16,8） * (8,1,16,8) = （8,2048,16,8）
            pooled_feat_before_after = torch.cat((bbox_feats_new, bbox_feats_new * mask_all), dim=0)

            #avgpooling
            pooled_feat_before_after = self.model.avgpool(pooled_feat_before_after)
            pooled_feat_before_after = pooled_feat_before_after.view(pooled_feat_before_after.size(0),
                                                                     pooled_feat_before_after.size(1))  # (8,2048)

            cls_score_before_after = self.classifierCA(pooled_feat_before_after)  # (16.150)

            # cls_prob_before_after = F.softmax(cls_score_before_after, dim=1) # (16.150)
            cls_prob_before_after = cls_score_before_after

            # print("cls_prob_before_after=", cls_prob_before_after)

            cls_prob_before = cls_prob_before_after[0: num_batch]  # (8.150)

            cls_prob_after = cls_prob_before_after[num_batch: num_batch * 2]  # (8.150)

            label_gt = self.gt.cuda()

            prepare_mask_fg_num = label_gt.nonzero().size(0)

            prepare_mask_bg_num = num_batch - prepare_mask_fg_num

            sp_i = torch.ones([2, num_batch]).long()
            sp_i[0, :] = torch.arange(num_batch)
            sp_i[1, :] = label_gt
            sp_v = torch.ones([num_batch])
            one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v,
                                                      torch.Size([num_batch, self.class_num])).to_dense().cuda()

            before_vector = torch.sum(one_hot_sparse * cls_prob_before, dim=1)
            after_vector = torch.sum(one_hot_sparse * cls_prob_after, dim=1)

            change_vector = before_vector - after_vector - 0.02
            change_vector = torch.where(change_vector > 0, change_vector, torch.zeros(change_vector.shape).cuda())

            fg_index = torch.where(label_gt > 0, torch.ones(before_vector.shape).cuda(),
                                   torch.zeros(before_vector.shape).cuda())

            bg_index = 1 - fg_index

            if fg_index.nonzero().shape[0] != 0:
                not_01_fg_index = fg_index.nonzero()[:, 0].long()
            else:
                not_01_fg_index = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).cuda().long()  # for corner case？？？

            not_01_bg_index = bg_index.nonzero()[:, 0].long()

            change_vector_fg = change_vector[not_01_fg_index]
            # change_vector_bg = change_vector[not_01_bg_index]

            for_fg_change_vector = change_vector.clone()
            # for_bg_change_vector = change_vector.clone()

            for_fg_change_vector[not_01_bg_index] = -10000

            # for_bg_change_vector[not_01_fg_index] = -10000


            th_fg_value = torch.sort(change_vector_fg, dim=0, descending=True)[0][
                int(round(float(prepare_mask_fg_num) / fg_drop_per))]
            # print ("th_fg_value=", th_fg_value)

            drop_index_fg = for_fg_change_vector.gt(th_fg_value)

            drop_index_fg_bg = drop_index_fg  # + drop_index_bg

            # To invert the bool
            ignore_index_fg_bg = drop_index_fg_bg.logical_not()

            not_01_ignore_index_fg_bg = ignore_index_fg_bg.nonzero()[:, 0]

            mask_all[not_01_ignore_index_fg_bg.long(), :] = 1

            # ---------------------------------------------------------
            self.train()
            # print("mask_all.shape",mask_all,mask_all.shape)
            '''

            '''
            mask_all = Variable(mask_all, requires_grad=True)

            # x.shape = ([8, 2048, 16, 8]) *
            # x = x * mask_all
            return mask_all
        else:
            x = torch.ones(1).cuda()
            return x

    def softmax(self, dist, weight):
        max_v = torch.max(dist, dim=1, keepdim=True)[0]
        x = dist - max_v
        Z = torch.sum(torch.exp(x), dim=1, keepdim=True) + 1e-6  # avoid division by zero
        W = weight * torch.exp(x) / Z
        return W

    def invert_softmax(self, dist, weight):
        max_v = torch.max(dist, dim=1, keepdim=True)[0]
        x = dist - max_v
        Z = torch.sum(torch.exp(x), dim=1, keepdim=True) + 1e-6  # avoid division by zero
        W = weight * torch.exp(x) / Z
        return W

    def soft_competitive_attention(self, x, soft_flag):

        channel_flag = True

        # input= ([8, 2048, 16, 8])
        if self.training:
            self.eval()

            self.count_iter = self.count_iter + 1
            if self.count_iter < 82000 * 8:
                th_l = 64
                th_s = 16
                fg_drop_per = 2
                bg_drop_per = fg_drop_per * 3
            else:
                if self.count_iter == 82000 * 8:
                    print("------Change to bigger thresholds------")
                th_l = 75
                th_s = 16
                fg_drop_per = 2
                bg_drop_per = fg_drop_per * 3

            bbox_feats_new = x
            bbox_feats_new = Variable(bbox_feats_new.data, requires_grad=True)

            bbox_feats_tmp = self.model.avgpool(bbox_feats_new)  # (8,2048，1,1)
            bbox_feats_tmp = bbox_feats_tmp.view(bbox_feats_tmp.size(0), bbox_feats_tmp.size(1))  # (8,2048)

            # (batch_size)
            num_batch = bbox_feats_new.shape[0]
            num_channel = bbox_feats_new.shape[1]
            width = bbox_feats_new.shape[2]
            height = bbox_feats_new.shape[3]

            output_score = self.classifierCA(bbox_feats_tmp)
            # shape = （8,150）
            # print("out_score.shape", output_score, output_score.shape)
            index = self.gt

            one_hot = torch.zeros((1), dtype=torch.float32).cuda()
            one_hot = Variable(one_hot, requires_grad=False)

            sp_i = torch.ones([2, num_batch]).long()
            sp_i[0, :] = torch.arange(num_batch)
            sp_i[1, :] = index
            sp_v = torch.ones([num_batch])

            one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v,
                                                      torch.Size([num_batch, self.class_num])).to_dense().cuda()
            one_hot_sparse = Variable(one_hot_sparse, requires_grad=False)  # 8,150

            one_hot = torch.sum(output_score * one_hot_sparse)

            # self.zero_grad()
            self.classifierCA.zero_grad()


            one_hot.backward()


            grads_val = bbox_feats_new.grad.clone().detach()
            # print("grads_val", grads_val.shape)

            # (8,2048)
            grad_channel_mean = torch.mean(grads_val.view(num_batch, num_channel, -1), dim=2)
            # print("grad_channel_mean", grad_channel_mean.shape)

            # (8,2048,1,1)
            grad_channel_mean = grad_channel_mean.view(num_batch, num_channel, 1, 1)
            # print("grad_channel_mean", grad_channel_mean.shape)

            # att_all.shape = (8,16,8)
            att_all = torch.sum(bbox_feats_new * grad_channel_mean, 1)
            # print("att_all", att_all.shape)

            att_all = att_all.view(num_batch, -1)

            mask_shape = att_all.shape[1]

            assert mask_shape == width * height

            # self.zero_grad()
            self.classifierCA.zero_grad()

            # -------------------------CA ----------------------------
            if self.CA_layer != 4:
                th_l = int(mask_shape / 2)
            thl_mask_value = torch.sort(att_all, dim=1, descending=True)[0][:, th_l]  # torch.Size([8]
            '''
            thl_mask_value tensor([ 0.0005, -0.0003,  0.0004,  0.0005, -0.0001,  0.0007,  0.0007,  0.0005],
            '''
            # print("thl_mask_value", thl_mask_value)

            thl_mask_value = thl_mask_value.view(num_batch, 1).expand(num_batch, mask_shape)  # torch.Size([8, 150])
            # print("thl_mask_value", thl_mask_value, thl_mask_value.shape)

            mask_all_cuda = torch.where(att_all > thl_mask_value, torch.zeros(att_all.shape).cuda(),
                                        torch.ones(att_all.shape).cuda())  # (8,128)

            '''
            channel-level的
            '''
            # torch.Size([8, 2048, 16, 8] * [8, 2048, 1, 1] ) = [8, 2048, 16, 8] + avgpool = [8,2048,1,1]
            att_channel = self.model.avgpool(bbox_feats_new * grad_channel_mean)
            att_channel = att_channel.view(att_channel.size(0), att_channel.size(1))  # (8,2048)

            if soft_flag and channel_flag:
                # th_channel = int(num_channel / 2)
                th_channel = self.th
                # print ("th_channel = ", th_channel)
                # print ("att_channel = ", att_channel)

                th_channel_mask_value = torch.sort(att_channel, dim=1, descending=True)[0][:,
                                        th_channel]  # torch.Size([8]

                th_channel_mask_value = th_channel_mask_value.view(num_batch, 1).expand(num_batch,
                                                                                        num_channel)  # torch.Size([8, 2048])

                mask_channel_cuda = torch.where(att_channel > th_channel_mask_value,
                                                torch.zeros(att_channel.shape).cuda(),
                                                torch.ones(att_channel.shape).cuda())  # (8,2048)

                # (8,1,w,h) * (8,2048,1,1) = (8,2048,w,h)
                mask_final = mask_all_cuda.view(num_batch, 1, width, height) * mask_channel_cuda.view(num_batch,
                                                                                                      num_channel, 1, 1)

            elif not soft_flag and channel_flag:
                th_channel_mask_value = self.softmax(att_channel, 1)

            # print("Mask_Final", mask_final)
            # ---------------------------------------------------------
            self.train()

            mask_final = Variable(mask_final, requires_grad=True)

            # x = x * mask_final
            return mask_final
        else:
            x = torch.ones(1).cuda()
            return x

    def forward(self, x, gt, CA):

        """
        x_origin.shape torch.Size([8, 3, 256, 128])
        x_conv1.shape torch.Size([8, 64, 128, 64])
        x_bn1.shape torch.Size([8, 64, 128, 64])
        x_relu.shape torch.Size([8, 64, 128, 64])
        x_maxpool.shape torch.Size([8, 64, 64, 32])
        x_layer1.shape torch.Size([8, 256, 64, 32])
        x_layer2.shape torch.Size([8, 512, 32, 16])
        x_layer3.shape torch.Size([8, 1024, 16, 8])
        x_layer4.shape torch.Size([8, 2048, 16, 8])
        f_avgpool.shape torch.Size([8, 2048, 1, 1])
        """

        """
        x_att3.shape= torch.Size([8, 512, 48, 24])
        x_4.shape= torch.Size([8, 1024, 24, 12])
        x_att4.shape= torch.Size([8, 1024, 24, 12])
        x_5.shape= torch.Size([8, 2048, 24, 12])
        x_final.shape= torch.Size([8, 2048, 24, 12])
        """
        self.gt = gt
        self.SE = False
        # print("x_origin.shape", x.shape)
        x = self.model.conv1(x)
        # print("x_conv1.shape", x.shape)

        x = self.model.bn1(x)
        # print("x_bn1.shape", x.shape)

        x = self.model.relu(x)
        # print("x_relu.shape", x.shape)

        x = self.model.maxpool(x)
        # print("x_maxpool.shape", x.shape)

        if self.SE:
            x_1 = self.att_s1(x)
            x_1 = self.BN1(x_1)
            y_1 = self.att1(x_1)
            x = x_1 * y_1.expand_as(x_1)


        x = self.model.layer1(x)
        # print("x_layer1.shape", x.shape)

        if self.SE:
            x_2 = self.att_s2(x)
            x_2 = self.BN2(x_2)
            y_2 = self.att2(x_2)
            x = x_2 * y_2.expand_as(x_2)

        if self.CA_layer == 1 and CA == True:
            f = x.clone().detach()  # no gradient
            if self.soft:
                mask_all = self.soft_competitive_attention(f, self.soft)
            else:
                mask_all = self.competitive_attention(f)
            x = x * mask_all

        x = self.model.layer2(x)
        # print("x_layer2.shape", x.shape)

        if self.SE:
            x_3 = self.att_s3(x)
            x_3 = self.BN3(x_3)
            y_3 = self.att3(x_3)
            x = x_3 * y_3.expand_as(x_3)

        if self.CA_layer == 2 and CA == True:
            f = x.clone().detach()  # no gradient
            if self.soft:
                mask_all = self.soft_competitive_attention(f, self.soft)
            else:
                mask_all = self.competitive_attention(f)
            x = x * mask_all

        x = self.model.layer3(x)
        # print("x_layer3.shape", x.shape)

        if self.SE:
            x_4 = self.att_s4(x)
            x_4 = self.BN4(x_4)
            y_4 = self.att4(x_4)
            x = x_4 * y_4.expand_as(x_4)

        if self.CA_layer == 3 and CA == True:
            f = x.clone().detach()  # no gradient
            if self.soft:
                mask_all = self.soft_competitive_attention(f, self.soft)
            else:
                mask_all = self.competitive_attention(f)
            x = x * mask_all

        x = self.model.layer4(x)
        # print("x_layer4.shape", x.shape)
        if self.SE:
            x_5 = self.att_s5(x)
            x_5 = self.BN5(x_5)
            y_5 = self.att5(x_5)
            x = x_5 * y_5.expand_as(x_5)

        if self.CA_layer == 4 and CA == True:
            # f = self.model.avgpool(x)  #(8,2048,1,1)
            # f = f.view(f.size(0), f.size(1)) #(8,2048)
            f = x.clone().detach()  # no gradient
            if self.soft:
                mask_all = self.soft_competitive_attention(f, self.soft)
            else:
                mask_all = self.competitive_attention(f)
            x = x * mask_all
        # assert 1!=1

        f = self.model.partpool(x)
        f = f.view(f.size(0), f.size(1) * self.part)
        f = f.detach()  # no gradient


        x_old = self.model.avgpool(x)
        # X.shape= torch.Size([8, 2048, 1, 1])
        # print("X.shape=", x.shape)
        x_old = x_old.view(x_old.size(0), x_old.size(1))


        x1 = self.classifier1(x_old)

        # X1.shape = torch.Size([8, 77])
        # print("X1.shape=", x1.shape)

        x2 = self.classifier2(x_old)

        # X2.shape = torch.Size([8, 77])
        # print("X2.shape=", x2.shape)

        x_old = []
        x_old.append(x1)
        x_old.append(x2)

        if self.using_global:
            # (8, 2048, 24, 12) --> (8, 8, 24, 12)
            attention_maps = self.attentions(x)
            # real (8, 8*2048), fake (8, 8*2048)
            global_feat, global_feat_hat = self.bap(x, attention_maps, counterfactual=True)
            # print("global feature.shape = ", global_feat.shape, "global_hat.shape = ", global_feat_hat.shape)
            global_feat = global_feat.view(global_feat.shape[0], -1)
            global_feat_hat = global_feat_hat.view(global_feat.shape[0], -1)

            # 8,2018*8)->(8,2048)
            global_feat = self.classifier_bap(global_feat)
            global_feat_hat = self.classifier_bap(global_feat_hat)

            if self.training:
                # (8,2048) normalize for angular softmax
                feat_hat = self.bottleneck(global_feat_hat)
                feat = self.bottleneck(global_feat)  # normalize for angular softmax
            else:
                feat_hat = global_feat_hat
                feat = global_feat

            cls_score = self.classifier(feat)
            cls_score_hat = self.classifier(feat_hat)

        # feature, [score1[feature, score], score2[feature, score]]
            return f, x_old, cls_score, cls_score - cls_score_hat
        else:
            return f, x_old

if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it.
    net = ACID(150, stride=1, ca_layer=4)
    #net = ft_net_swin(751, stride=1)
    net.classifier1 = nn.Sequential()
    net.classifier2 = nn.Sequential()
    print(net)
    input = Variable(torch.FloatTensor(8, 3, 224, 224))
    label = torch.Tensor([0,2,4,5,6,7,8,22])
    output = net(input, label)
    print('net output size:')
    print(output.shape)
