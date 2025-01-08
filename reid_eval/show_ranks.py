# coding=utf-8
import json
import  numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import os
import glob
import argparse

parser = argparse.ArgumentParser(description='Showing')
parser.add_argument('--name', default='test', type=str, help='save model path')
parser.add_argument('--cloth', action='store_true', help='standard setting or cloth-change setting')
parser.add_argument('--mix_cloth', action='store_true', help='standard setting + cloth-change setting')
parser.add_argument('--ltcc', action='store_true', help='dataset selection')
parser.add_argument('--vc', action='store_true', help='dataset selection')
parser.add_argument('--celeb', action='store_true', help='dataset selection')
parser.add_argument('--celebL', action='store_true', help='dataset selection')
parser.add_argument('--mat_saved_path', default='pytorch_result.mat', type=str, help='save mat path')

opt = parser.parse_args()

# results_file = "grade_20_first_result_fold5_False.json"
# results_file = "result_rank_0.5.json"
results_file = opt.mat_saved_path + "/result_rank.json"
if opt.cloth:
    query_image_path = "/PRCC_reid/test/C"
    flag = 'Cloth'
elif opt.mix_cloth:
    query_image_path = "/PRCC_reid/test/BC"
    flag = 'Mix_Cloth'
else:
    query_image_path = "/PRCC_reid/test/B"
    flag = 'Standard'

gallery_image_path = "/PRCC_reid/test/A"

IMAGE_SAVE_PATH = ''  # 图片转换后的地址

IMAGE_SIZE = 300  # 每张小图片的大小
IMAGE_HEIGHT = 300  # 每张小图片的高
IMAGE_WIDTH = 100  # 每张小图片的宽
IMAGE_ROW = 1  # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 11  # 图片间隔，也就是合并成一张图后，一共有几列

# Load the result file of test
with open(results_file, "r") as f:
    results = json.load(f)
    all_count = 0
    jump_count = 0
    for result in results["rank_results"]:
        r_count = 0
        if result["is_top1"] == 1:
            r_count = r_count + 1

        # {'id': 0, 'label': 1, 'image_name': 'A_cropped_rgb001.jpg',
        # 'top5': [2033, 2046, 2031, 2037, 2047], 'top1': 2033,
        # 'top10': [2033, 2046, 2031, 2037, 2047, 2042, 2035, 2039, 2040, 2038],
        # 'rank_name': ['C_cropped_rgb013.jpg', 'C_cropped_rgb052.jpg', 'C_cropped_rgb007.jpg',
        # 'C_cropped_rgb025.jpg', 'C_cropped_rgb055.jpg', 'C_cropped_rgb040.jpg', 'C_cropped_rgb019.jpg',
        # 'C_cropped_rgb031.jpg', 'C_cropped_rgb034.jpg', 'C_cropped_rgb028.jpg'],
        # 'rank_label': [159, 159, 159, 159, 159, 159, 159, 159, 159, 159],
        # 'is_top1': 0, 'is_top5': 0, 'is_top10': 0}

        if result["is_top1"] != -1:

            to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_WIDTH, IMAGE_ROW * IMAGE_HEIGHT))  # 创建一个新图

            folder_path = os.path.join('../rank_results', opt.name, flag)

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            if result["is_top10"] == 1:
                IMAGE_SAVE_PATH = (folder_path+'/inTen_' + '%03d' % result["label"] + "_" + result["image_name"])
            else:
                IMAGE_SAVE_PATH = (folder_path+'/outTen_' + '%03d' % result["label"] + "_" + result["image_name"])

            if os.path.exists(IMAGE_SAVE_PATH):
                os.remove(IMAGE_SAVE_PATH)

            query_image = os.path.join(query_image_path, '%03d' % result["label"], result["image_name"])
            #             print("Query file name :", query_image)

            from_image = Image.open(query_image).resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)
            if result["is_top10"] != 1:
                draw = ImageDraw.Draw(from_image)
                draw.rectangle([0, 0, IMAGE_WIDTH, IMAGE_HEIGHT], outline=(255, 0, 0), width=5)

                # font = ImageFont.truetype("consola.ttf", 16, encoding="unic")  # 设置字体

                # draw.text([10, 10], 'All False', (255, 0, 0), font)

            to_image.paste(from_image, ((0) * IMAGE_WIDTH, (0) * IMAGE_HEIGHT))

            for i, gallery_imgs in enumerate(result["rank_name"]):
                gallery_image = os.path.join(gallery_image_path, '%03d' % result["rank_label"][i], gallery_imgs)

                from_image = Image.open(gallery_image).resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)

                draw = ImageDraw.Draw(from_image)
                # if right results -------  GREEN
                if result["rank_label"][i] == result["label"]:
                    draw.rectangle([0, 0, IMAGE_WIDTH, IMAGE_HEIGHT], outline=(0, 255, 0), width=5)
                # Wrong results ------ RED
                #                 else:
                #                     draw.rectangle([0, 0, IMAGE_WIDTH, IMAGE_HEIGHT], outline=(255, 0, 0), width=5)

                to_image.paste(from_image, ((i + 1) * IMAGE_WIDTH, (0) * IMAGE_HEIGHT))

            all_count = all_count + 1
            to_image.save(IMAGE_SAVE_PATH)

            if all_count == 3:
                break
    print(all_count, " Images write Success!")

