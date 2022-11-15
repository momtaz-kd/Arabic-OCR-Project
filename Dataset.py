import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import random
import os
import json
from math import floor
from numpyencoder import NumpyEncoder
from utils import min_area_rectangle
from pathlib import Path

# This folder will be the input for the data generator
# The data generator will take each image in this folder
# Then it'll generate different instances from this image  
dataset_dir = Path('./dataset')
# image_path = './Data/sample.png'

# This folder will be the output of the data generator
out_dataset_dir = 'Arabic'

# objs_coor represents coordinates of the objects for each image
objs_coor = {'sample':{'id':(150,193,525,625), 'date':(150,193,212,428)},
            'real':{'id':(370,475,75,220),'date':(500,590,845,1175)},
            'reall':{'id':(75,175,190,360),'date':(200,290,900,1265)},  
}

# For Testing
# im = cv2.imread(image_path)
# print(im.shape)
# Convert from BGR color system to RGB color system
# im = im[:,:,[2,1,0]]
# Or another method using opencv
# im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
# plt.imshow(im)
# plt.show()


def generate_instances(dataset_dir, coordiantes, min_rot_deg = -8, max_rot_deg = 8, num_of_instances=100, train_val_split=0.7):
    """
    This function generates a number of instances from one image
    by applying pixel transformations on the image itself spatial transformations
    on the coordinates
    """
    generated_train_imgs = {}
    generated_val_imgs = {}
    generated_imgs = {}
    c = 0  # to count the images

    num_of_train_imgs = floor(train_val_split * num_of_instances)
    # num_of_val = num_of_instances - num_of_train
    # print('num_of_train_imgs are: ', num_of_train_imgs)

    for image_path in dataset_dir.iterdir():
        image_path = str(image_path)
        print('image_path is: ', image_path)
        c+=1

        image = cv2.imread(image_path)
        last_sub_string = os.path.basename(image_path)
        image_name = last_sub_string[:last_sub_string.find('.')]
        ext = last_sub_string[last_sub_string.find('.')+1:]
        print('image_name is: ', image_name)
        print('extension is: ', ext)

        height, width, _ = image.shape
        print('image shape is: ', image.shape)
        # create mask for each object
        id_mask = np.zeros((height, width), np.uint8)
        date_mask = np.zeros((height, width), np.uint8)

        # Assign coordinates for every item
        basic_obj_coor = coordiantes[image_name]
        (id_ymin, id_ymax, id_xmin, id_xmax) = basic_obj_coor['id']
        (date_ymin, date_ymax, date_xmin, date_xmax) = basic_obj_coor['date']

        cv2.rectangle(id_mask, (id_xmin, id_ymin), (id_xmax, id_ymax), 255, thickness=-1)
        cv2.rectangle(date_mask, (date_xmin, date_ymin), (date_xmax, date_ymax), 255, thickness=-1)

        for i in range(num_of_instances):

            # Initialize the image and the masks for each augmentation
            temp_img = image.copy()
            temp_id_mask = id_mask.copy()
            temp_date_mask = date_mask.copy()

            # temp_img = temp_img[:,:,[2,1,0]]
            # plt.imshow(temp_img)
            # plt.show()

            #################  Pixel-level transforms  ###################

            pixel_aug = A.OneOf([A.Blur(always_apply=False, p=1, blur_limit=(1,3)),
                    A.GaussNoise(always_apply=False, p=1, var_limit=(10.0, 50.0)), 
                    A.ISONoise(always_apply=False, p=1, intensity=(0.5, 1), color_shift=(0.01, 0.05)),
                    A.MotionBlur(always_apply=False, p=1, blur_limit=(3, 5)), 
                    A.RandomBrightnessContrast(always_apply=False, p=1.0, brightness_limit=(-0.4, 0.4), contrast_limit=(-0.4, 0.4), brightness_by_max=True)] ,p=1)

            augmented_img = pixel_aug(image=temp_img)
            temp_img = augmented_img['image']

            #################  Spatial-level transforms  ###################

            p = random.uniform(0, 1)
            selected_deg = random.randint(min_rot_deg, max_rot_deg)
            #  Create the transform based on probability
            if p > 0.5:
                # Make rotation transform 
                sp_aug = A.Affine(rotate=selected_deg, fit_output=True, p=1)
            else:
                # Make shearing transformation
                if p > 0.25:
                    # make shearing along x-axis
                    sp_aug = A.Affine(shear={'x':selected_deg}, fit_output=True, p=1)
                else:
                    # make shearing along y-axis
                    sp_aug = A.Affine(shear={'y':selected_deg}, fit_output=True, p=1)


            # Apply spatial transform
            id_augmented = sp_aug(image=temp_img, mask=temp_id_mask)
            date_augmented = sp_aug(image=temp_img, mask=temp_date_mask)

            temp_img, temp_id_mask = id_augmented['image'], id_augmented['mask']
            temp_date_mask = date_augmented['mask']

            # Extract the object for each image
            id = cv2.bitwise_and(temp_img, temp_img, mask=temp_id_mask)
            date = cv2.bitwise_and(temp_img, temp_img, mask=temp_date_mask)

            #  Find the coordinates for each object
            id_box, _, id_ploy = min_area_rectangle(id)
            date_box, _, date_poly = min_area_rectangle(date)

            # Draw the image with desired objects
            final_res = cv2.drawContours(temp_img.copy(), [id_box], 0, (255,0,0), 1)
            final_res = cv2.drawContours(final_res.copy(), [date_box], 0, (0,0,255), 1)

            new_img_name = image_name + str(i) + '.' + ext
            # debug_img_name = os.path.join('images_with_transforms', new_img_name)
            # cv2.imwrite(debug_img_name, final_res)

            # id
            id_x_coor = [p[0] for p in id_box]
            id_y_coor = [p[1] for p in id_box]
            id_bbox = [min(id_x_coor), min(id_y_coor), max(id_x_coor), max(id_y_coor)]
            id_seg = list(id_box.flat)
            
            # date
            date_x_coor = [p[0] for p in date_box]
            date_y_coor = [p[1] for p in date_box]
            date_bbox = [min(date_x_coor), min(date_y_coor) ,max(date_x_coor), max(date_y_coor)]
            date_seg = list(date_box.flat)
            
            objs_coor = {'id':[id_bbox, id_seg], 'date':[date_bbox, date_seg]}

            # form img_id from image id and instance id
            img_id = int(str(c) + str(i))

            if i < num_of_train_imgs:
                generated_train_imgs[img_id] = {'image_name':new_img_name, 'image':temp_img, 'objs_coor':objs_coor}
            else:
                generated_val_imgs[img_id] = {'image_name':new_img_name, 'image':temp_img, 'objs_coor':objs_coor}

    
    generated_imgs['train'] = generated_train_imgs
    generated_imgs['val'] = generated_val_imgs
    
    return generated_imgs


def make_detectron2_format(dic_images, dataset_dir):
    """
    This function makes the desired format for detectron2 framework
    Which means it'll make a dataset.json file that contains the labels 
    for each image
    """

    dic_dataset = {'train':[], 'val':[]}
    # train_dataset = []
    # val_dataset = []

    labels = {'id':0, 'date':1}

    for train_val in dic_images.keys():

        for img_id in dic_images[train_val].keys():

            # print('img_id is: ', img_id)

            # Extract the information from the dictonary
            image_name = dic_images[train_val][img_id]['image_name']
            img = dic_images[train_val][img_id]['image']
            objs_coor = dic_images[train_val][img_id]['objs_coor']

            saved_img_path = os.path.join(dataset_dir, train_val, image_name)
            cv2.imwrite(saved_img_path, img)

            record = {}
            record['image_id'] = img_id
            record['file_name'] = image_name
            record["height"] = img.shape[0]
            record["width"] = img.shape[1]

            objs = []
            for obj_coor in objs_coor.keys():
                bbox = objs_coor[obj_coor][0]
                seg = objs_coor[obj_coor][1]

                obj = {
                    "bbox": bbox,
                    "bbox_mode": 1,
                    "segmentation": [seg],
                    "category_id": labels[obj_coor]
                    }
                objs.append(obj)
            
            record["annotations"] = objs

            dic_dataset[train_val].append(record)
     
            # if train_val == 'train':
            #     train_dataset.append(record)
            # else:
            #     val_dataset.append(record)

        with open(os.path.join(dataset_dir, train_val ,"dataset.json") ,"w") as outfile:
            json.dump(dic_dataset[train_val], outfile, cls=NumpyEncoder)
    
    # dic_dataset['train'] = train_dataset
    # dic_dataset['val'] = val_dataset

    # return None
    
generated_imgs = generate_instances(dataset_dir, objs_coor)

make_detectron2_format(generated_imgs, out_dataset_dir)


# with open(os.path.join(out_dataset_dir, 'train' ,"dataset.json") ,"w") as outfile:
#      json.dump(dic_dataset['train'], outfile, cls=NumpyEncoder)

# with open(os.path.join(out_dataset_dir, 'val' ,"dataset.json") ,"w") as outfile:
#      json.dump(dic_dataset['val'], outfile, cls=NumpyEncoder)










