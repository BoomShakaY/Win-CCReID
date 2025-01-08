# Modify the path to your own data path
# Same as other datasets

#--name  # model name
#--which_epoch  # selects the i-th model
#--cloth # consider clothing

#--single_shot # single shot setting for PRCC only
#--seed # seed for single shot setting

# --ltcc # for LTCC
# --vc # for VC-Cloth
# --celeb # for celeb
# --celebL # for celeb-light


# for PRCC
python test_2label_cloth.py --name ACID_Default \
--test_dir YOUR_DATA_PATH/PRCC_reid/ \
--which_epoch 134221 \

python test_2label_cloth.py --name ACID_Default \
--test_dir YOUR_DATA_PATH/PRCC_reid/ \
--which_epoch 134221 \
--cloth

python test_2label_cloth.py --name ACID_Default \
--test_dir YOUR_DATA_PATH/PRCC_reid/ \
--which_epoch 134221 \
--single_shot \
--seed 0


# for LTCC
python test_2label_cloth.py --name ACID_Default_LTCC \
--test_dir YOUR_DATA_PATH/LTCC_ReID/pytorch \
--which_epoch 74000 \
--cloth \
--ltcc

# for VC-Cloth
#Run your code.
python test_2label_cloth.py --name ACID_Default_VC \
--test_dir YOUR_DATA_PATH/VC-Clothes/pytorch \
--which_epoch 74000 \
--cloth \
--vc

# for celeb
#Run your code.
python test_2label_cloth.py --name ACID_Default_Celeb \
--test_dir YOUR_DATA_PATH \
--which_epoch 74000 \
--cloth \
--celeb

# for celeb-light
#Run your code.
python test_2label_cloth.py --name ACID_Default_CelebL \
--test_dir YOUR_DATA_PATH \
--which_epoch 74000 \
--cloth \
--celebL





