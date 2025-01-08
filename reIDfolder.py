# coding=utf-8
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import random
import time

class ReIDFolder(datasets.ImageFolder):

    def __init__(self, root, transform, dataset_name):
        super(ReIDFolder, self).__init__(root, transform)
        targets = np.asarray([s[1] for s in self.samples])
        self.targets = targets
        self.img_num = len(self.samples)
        self.dataset_name = dataset_name
        self.train_ids = []

        print("img num is:", self.img_num)

        if self.dataset_name == 'ltcc':
            """
            Change every path to your own path
            """
            self.info_path = 'YOUR_DATA_PATH/LTCC_ReID/info/'

            self.cloth_changed_id_test = 'cloth-change_id_test.txt'
            self.cloth_unchanged_id_test = 'cloth-unchange_id_test.txt'

            self.cloth_changed_id_trian = 'cloth-change_id_train.txt'
            self.cloth_unchanged_id_trian = 'cloth-unchange_id_train.txt'

            for line in open(self.info_path + self.cloth_changed_id_trian):
                self.train_ids.append(int(line))

        # print(self.train_ids)



    def _get_cam_id(self, path):
        cam2label = {'A': 0, 'B': 1, 'C': 2}
        filename = path.split('/')[-1]
        camera_id = cam2label[filename[0]]
        return int(camera_id)

    def _get_ltcc_cam_id(self, path):
        filename = path.split('/')[-1]
        cam_id = filename.split('_')[1][1:]
        return int(cam_id)

    def _get_cloth_id(self, path):
        filename = path.split('/')[-1]
        cloth_id = filename.split('_')[0]
        return int(cloth_id)

    # 与原图同衣服，但是不同pose的图片
    def _get_pos_sample(self, target, index, path):
        # self.targets [  0   0   0 ... 149 149 149] (17896,)
        # 全部的label顺序，也就是gt
        # print ("self.targets",self.targets, self.targets.shape)

        # label 0,0,0,1与图像对应
        # batch_index 从0到每一张图
        # print("target, index = ", target, index)

        # 找到的对应label下的index
        pos_index = np.argwhere(self.targets == target)
        # 拉成一维
        pos_index = pos_index.flatten()

        # 在pos_index中但不在index中的已排序的唯一值
        pos_index = np.setdiff1d(pos_index, index)

        # 找到cam_id,得到ABC中指定范围的同类型pos
        target_cam = self._get_cam_id(path)

        if len(pos_index)==0:  # in the query set, only one sample
            return path
        else:
            # 当A时，可以用AB， 但是当C时只能用C
            while True:
                rand = random.randint(0, len(pos_index)-1)
                pos_cam = self._get_cam_id(self.samples[pos_index[rand]][0])
                if target_cam == 0 or target_cam == 1:
                    if pos_cam != 2:
                        break
                elif pos_cam == 2:
                    break

        # print("Reture value=", self.samples[pos_index[rand]])
        return self.samples[pos_index[rand]][0]

    def _get_cloth_pos_sample(self, target, index, path):

        # self.targets [  0   0   0 ... 149 149 149] (17896,)
        # 全部的label顺序，也就是gt
        # print ("self.targets",self.targets, self.targets.shape)

        # label 0,0,0,1与图像对应
        # batch_index 从0到每一张图
        # print("target, index = ", target, index)

        # 找到的对应label下的index
        cloth_pos_index = np.argwhere(self.targets == target)
        # 拉成一维
        cloth_pos_index = cloth_pos_index.flatten()

        # 在cloth_pos_index中但不在index中的已排序的唯一值
        cloth_pos_index = np.setdiff1d(cloth_pos_index, index)

        # 找到cam_id,得到ABC中指定范围的同类型pos
        target_cam = self._get_cam_id(path)

        if len(cloth_pos_index)==0:  # in the query set, only one sample
            return path
        else:
            # 当AB时选C， 当C选AB
            while True:
                rand = random.randint(0, len(cloth_pos_index)-1)
                cloth_pos_cam = self._get_cam_id(self.samples[cloth_pos_index[rand]][0])
                if target_cam == 0 or target_cam == 1:
                    if cloth_pos_cam == 2:
                        break
                elif cloth_pos_cam != 2:
                    break
        # print("Reture value=", self.samples[pos_index[rand]])
        return self.samples[cloth_pos_index[rand]][0]

    def _get_ltcc_pos_sample(self, target, index, path):

        # self.targets [  0   0   0 ... 149 149 149] (17896,)
        # 全部的label顺序，也就是gt
        # print ("self.targets",self.targets, self.targets.shape)

        # label 0,0,0,1与图像对应
        # batch_index 从0到每一张图
        # print("target, index = ", target, index)

        # 找到的对应label下的index
        pos_index = np.argwhere(self.targets == target)
        # 拉成一维
        pos_index = pos_index.flatten()

        # 在pos_index中但不在index中的已排序的唯一值
        pos_index = np.setdiff1d(pos_index, index)

        # 找到原图的cloth_id
        target_cloth = self._get_cloth_id(path)

        if len(pos_index)==0:  # in the query set, only one sample
            return path
        else:
            while True:
                rand = random.randint(0, len(pos_index)-1)
                pos_cloth = self._get_cloth_id(self.samples[pos_index[rand]][0])
                # 选pos时当两者衣服相同才可以跳出循环，不管cam
                if target_cloth == pos_cloth:
                    break
        # print("Reture value=", self.samples[pos_index[rand]])
        return self.samples[pos_index[rand]][0]

    def _get_ltcc_cloth_pos_sample(self, target, index, path):

        # self.targets [  0   0   0 ... 149 149 149] (17896,)
        # 全部的label顺序，也就是gt
        # print ("self.targets",self.targets, self.targets.shape)

        # label 0,0,0,1与图像对应
        # batch_index 从0到每一张图
        # print("target, index = ", target, index)

        reletive_label = path.split('/')[-2]
        reletive_label = int(reletive_label)

        if reletive_label not in self.train_ids:
            # print ("I return~", path, target)
            return path

        # 找到的对应label下的index
        cloth_pos_index = np.argwhere(self.targets == target)
        # 拉成一维
        cloth_pos_index = cloth_pos_index.flatten()

        # 在cloth_pos_index中但不在index中的已排序的唯一值
        cloth_pos_index = np.setdiff1d(cloth_pos_index, index)

        # 找到对应的cloth类型
        target_cloth = self._get_cloth_id(path)

        change_count = 0
        if len(cloth_pos_index) == 0:  # in the query set, only one sample
            return path
        else:
            while True:
                rand = random.randint(0, len(cloth_pos_index)-1)
                cloth_pos_cloth = self._get_cloth_id(self.samples[cloth_pos_index[rand]][0])
                # 只需要两者衣服不同，即可跳出循环
                if target_cloth != cloth_pos_cloth:
                    break
                else:
                    change_count = change_count + 1

                if change_count == 100:
                    print("I jump ~", path)
                    break
        # print("Reture value=", self.samples[pos_index[rand]])
        return self.samples[cloth_pos_index[rand]][0]

    def _get_vc_pos_sample(self, target, index, path):

        #首先判断一下图片是不是符合条件
        # print("Run Check: ", path)
        file_name = path.split('/')[-1]
        target_cloth = self._get_cloth_id(path)

        # VC-Clothes/pytorch/train/009/xxxxxx
        # to --> VC-Clothes/pytorch/train/009/

        ID_Folder = path.replace(file_name, "")

        cloth_num = 0
        for file in os.listdir(ID_Folder):
            # print("This file is:", file)
            cloth_id = file.split("_")[0]
            cloth_id = int(cloth_id)
            # 如果与原图cloth_id相等，则数量+1， 注意其中包含了原图本身，所以最少是1
            if target_cloth == cloth_id:
                # 表示第一次见过
                if cloth_num == 0:
                    cloth_num = cloth_num + 1
                else:
                    # 表示至少有两张，符合条件，可以不继续了
                    cloth_num = cloth_num + 1
                    break

        if cloth_num == 1:
            print(path, " 同ID 同Cloth的仅有一张")
            return path

        # self.targets [  0   0   0 ... 149 149 149] (17896,)
        # 全部的label顺序，也就是gt
        # print ("self.targets",self.targets, self.targets.shape)

        # label 0,0,0,1与图像对应
        # batch_index 从0到每一张图
        # print("target, index = ", target, index)

        # 找到的对应label下的index
        pos_index = np.argwhere(self.targets == target)
        # 拉成一维
        pos_index = pos_index.flatten()

        # 在pos_index中但不在index中的已排序的唯一值
        pos_index = np.setdiff1d(pos_index, index)

        # 找到原图的cloth_id, 上面已经找过一次了
        # target_cloth = self._get_cloth_id(path)

        if len(pos_index)==0:  # in the query set, only one sample
            return path
        else:
            while True:
                rand = random.randint(0, len(pos_index)-1)
                pos_cloth = self._get_cloth_id(self.samples[pos_index[rand]][0])
                # 选pos时当两者衣服相同才可以跳出循环，不管cam
                if target_cloth == pos_cloth:
                    break
        # print("Reture value=", self.samples[pos_index[rand]])
        return self.samples[pos_index[rand]][0]

    def _get_vc_cloth_pos_sample(self, target, index, path):
        #首先判断一下图片是不是符合条件
        # print("Run Check: ", path)
        file_name = path.split('/')[-1]
        target_cloth = self._get_cloth_id(path)

        ID_Folder = path.replace(file_name, "")

        Cloth_check = False
        for file in os.listdir(ID_Folder):
            cloth_id = file.split("_")[0]
            cloth_id = int(cloth_id)
            # 如果与原图cloth_id不等，则表示至少存在两套衣服，则更改状态，继续
            if target_cloth != cloth_id:
                Cloth_check = True
                break

        # 如果还是False表示只存在一套衣服，返回自己本身
        # 但是这种情况下提取的Appearance code就是本身的code
        if not Cloth_check:
            # print(path, " 仅存在一套ClothID")
            return path

        # self.targets [  0   0   0 ... 149 149 149] (17896,)
        # 全部的label顺序，也就是gt
        # print ("self.targets",self.targets, self.targets.shape)

        # 找到的对应label下的index
        cloth_pos_index = np.argwhere(self.targets == target)
        # 拉成一维
        cloth_pos_index = cloth_pos_index.flatten()

        # 在cloth_pos_index中但不在index中的已排序的唯一值
        cloth_pos_index = np.setdiff1d(cloth_pos_index, index)

        # 找到对应的cloth类型
        # target_cloth = self._get_cloth_id(path)

        change_count = 0
        if len(cloth_pos_index) == 0:  # in the query set, only one sample
            return path
        else:
            while True:
                rand = random.randint(0, len(cloth_pos_index)-1)
                cloth_pos_cloth = self._get_cloth_id(self.samples[cloth_pos_index[rand]][0])
                # 只需要两者衣服不同，即可跳出循环
                if target_cloth != cloth_pos_cloth:
                    break
                else:
                    change_count = change_count + 1

                if change_count == 100:
                    print("I jump ~", path)
                    break
        # print("Reture value=", self.samples[pos_index[rand]])
        return self.samples[cloth_pos_index[rand]][0]

    def _get_celeb_pos_sample(self, target, index, path):
        return path

    def _get_celeb_cloth_pos_sample(self, target, index, path):

        # self.targets [  0   0   0 ... 149 149 149] (17896,)
        # 全部的label顺序，也就是gt
        # print ("self.targets",self.targets, self.targets.shape)

        # 找到的对应label下的index
        cloth_pos_index = np.argwhere(self.targets == target)
        # 拉成一维
        cloth_pos_index = cloth_pos_index.flatten()

        # 在cloth_pos_index中但不在index中的已排序的唯一值
        cloth_pos_index = np.setdiff1d(cloth_pos_index, index)

        # 找到对应的cloth类型
        target_cloth = self._get_cloth_id(path)

        change_count = 0
        if len(cloth_pos_index) == 0:  # in the query set, only one sample
            return path
        else:
            while True:
                rand = random.randint(0, len(cloth_pos_index)-1)
                cloth_pos_cloth = self._get_cloth_id(self.samples[cloth_pos_index[rand]][0])
                # 只需要两者衣服不同，即可跳出循环
                if target_cloth != cloth_pos_cloth:
                    break
                else:
                    change_count = change_count + 1

                if change_count == 100:
                    print("I jump ~", path)
                    break
        # print("Reture value=", self.samples[pos_index[rand]])
        return self.samples[cloth_pos_index[rand]][0]

    def _get_neg_sample(self, target):
        neg_index = np.argwhere(self.targets != target)
        neg_index = neg_index.flatten()
        rand = random.randint(0,len(neg_index)-1)
        return self.samples[neg_index[rand]]

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        # if self.dataset_name == 'prcc':
        #     pos_path = self._get_pos_sample(target, index, path)
        # elif self.dataset_name == 'ltcc':
        #     if "test" in path or "gallery" in path or "query" in path:
        #         pos_path = path
        #     else:
        #         pos_path = self._get_ltcc_pos_sample(target, index, path)

        if "test" in path or "gallery" in path or "query" in path:
            pos_path = path
        elif self.dataset_name == 'prcc':
            pos_path = self._get_pos_sample(target, index, path)
        elif self.dataset_name == 'ltcc':
            pos_path = self._get_ltcc_pos_sample(target, index, path)
        elif self.dataset_name == 'vc':
            pos_path = self._get_vc_pos_sample(target, index, path)
        elif self.dataset_name == 'celeb' or self.dataset_name == 'celeb_light':
            pos_path = self._get_celeb_pos_sample(target, index, path)

        pos = self.loader(pos_path)

        if "test" in path or "gallery" in path or "query" in path:
            cloth_pos = pos
        else:
            if self.dataset_name == 'prcc':
                cloth_pos_path = self._get_cloth_pos_sample(target, index, path)
            elif self.dataset_name == 'ltcc':
                cloth_pos_path = self._get_ltcc_cloth_pos_sample(target, index, path)
            elif self.dataset_name == 'vc':
                cloth_pos_path = self._get_vc_cloth_pos_sample(target, index, path)
            elif self.dataset_name == 'celeb' or self.dataset_name == 'celeb_light':
                cloth_pos_path = self._get_celeb_cloth_pos_sample(target, index, path)

            cloth_pos = self.loader(cloth_pos_path)

        if self.transform is not None:
            sample = self.transform(sample)
            pos = self.transform(pos)
            cloth_pos = self.transform(cloth_pos)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # print("Path=", path)
        # print("Pos_path=", pos_path)
        # print("Cloth_pos_path=", cloth_pos_path)
        # assert 1!=1
        return sample, target, pos, cloth_pos
        # return sample, target, pos


if __name__ == '__main__':

    # input_folder = /PRCC_reid/train'
    input_folder = '/Celeb_reid/pytorch/train'
    # input_folder = '/LTCC_ReID/pytorch/train'
    transform_list = [transforms.Resize((256, 128), interpolation=3),
                      transforms.ToTensor(),
                      transforms.Normalize((0.485, 0.456, 0.406),
                                           (0.229, 0.224, 0.225))
                      ]

    transform = transforms.Compose(transform_list)

    dataset = ReIDFolder(input_folder, transform=transform, dataset_name='celeb')
    loader = DataLoader(dataset=dataset, batch_size=8, num_workers=0)
    count = 0
    begin = time.time()
    # for (images_a, labels_a, pos_a, cloth_pos_a) in loader:
    for (images_a, labels_a, pos_a, cloth_pos) in loader:
        # shape=  torch.Size([8, 3, 256, 128]) torch.Size([8]) torch.Size([8, 3, 256, 128])
        # print("shape= ", images_a.shape, labels_a.shape, pos_a.shape, cloth_pos_a.shape)
        # print (images_a)
        # print (pos_a)
        # print (cloth_pos)

        count = count +1
        if count == 160:
            break
    end = time.time()

    print(count, "images loaded in", end-begin, "seconds")



