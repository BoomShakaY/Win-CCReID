import os
from shutil import copyfile


"""
LTCC Data Format : 

001 _   1   _   c11  _ 015833.png

ID _ Cloth _ Camera _ non.png 


VC_Cloht Data Format:

0001-01-01-01.jpg

ID-Camear-Cloth-rank.jpg


Celeb Data Format:

100_13_0.jpg
ID _ ith  _ non.jpg
"""
# You only need to change this line to your dataset download path
download_path = 'Celeb_reid_light'

if not os.path.isdir(download_path):
    print('please change the download_path')

save_path = download_path + '/pytorch'
if not os.path.isdir(save_path):
    os.mkdir(save_path)
#-----------------------------------------
#query
query_path = download_path + '/query'
query_save_path = download_path + '/pytorch/query'
if not os.path.isdir(query_save_path):
    os.mkdir(query_save_path)

for root, dirs, files in os.walk(query_path, topdown=True):
    for name in files:
        if not name[-3:] == 'jpg':
            continue
        ID  = name.split('_')
        src_path = query_path + '/' + name
        # only 001
        dst_path = query_save_path + '/' + ID[0]
        # cloth_non
        new_name = ID[1] + "_" + ID[2]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + new_name)
print("Celeb_query prepare Success")
# #-----------------------------------------
# #multi-query
# query_path = download_path + '/gt_bbox'
# # for dukemtmc-reid, we do not need multi-query
# if os.path.isdir(query_path):
#     query_save_path = download_path + '/pytorch/multi-query'
#     if not os.path.isdir(query_save_path):
#         os.mkdir(query_save_path)
#
#     for root, dirs, files in os.walk(query_path, topdown=True):
#         for name in files:
#             if not name[-3:]=='png':
#                 continue
#             ID  = name.split('_')
#             src_path = query_path + '/' + name
#             dst_path = query_save_path + '/' + ID[0]
#             if not os.path.isdir(dst_path):
#                 os.mkdir(dst_path)
#             copyfile(src_path, dst_path + '/' + name)

#-----------------------------------------
#gallery
gallery_path = download_path + '/gallery'
gallery_save_path = download_path + '/pytorch/gallery'
if not os.path.isdir(gallery_save_path):
    os.mkdir(gallery_save_path)

for root, dirs, files in os.walk(gallery_path, topdown=True):
    for name in files:
        if not name[-3:] == 'jpg':
            continue
        ID  = name.split('_')
        src_path = gallery_path + '/' + name
        # only 001 but not 0001
        dst_path = gallery_save_path + '/' + ID[0]
        # cloth_non
        new_name = ID[1] + "_" + ID[2]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + new_name)
print("Celeb_gallery prepare Success")

#---------------------------------------
#train_all
train_path = download_path + '/train'
train_save_path = download_path + '/pytorch/train'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)

for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name[-3:] == 'jpg':
            continue
        ID  = name.split('_')
        src_path = train_path + '/' + name
        # only 001 but not 0001
        dst_path = train_save_path + '/' + ID[0]
        # cloth_non
        new_name = ID[1] + "_" + ID[2]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + new_name)

print("Celeb_train prepare Success")
# #---------------------------------------
# #train_val
# train_path = download_path + '/bounding_box_train'
# train_save_path = download_path + '/pytorch/train'
# val_save_path = download_path + '/pytorch/val'
# if not os.path.isdir(train_save_path):
#     os.mkdir(train_save_path)
#     os.mkdir(val_save_path)
#
# for root, dirs, files in os.walk(train_path, topdown=True):
#     for name in files:
#         if not name[-3:]=='png':
#             continue
#         ID  = name.split('_')
#         src_path = train_path + '/' + name
#         dst_path = train_save_path + '/' + ID[0]
#         if not os.path.isdir(dst_path):
#             os.mkdir(dst_path)
#             dst_path = val_save_path + '/' + ID[0]  #first image is used as val image
#             os.mkdir(dst_path)
#         copyfile(src_path, dst_path + '/' + name)


print("Celeb prepare Success")