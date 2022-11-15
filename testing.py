import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import imutils
import os

# detection libraries
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg



class Detection:
    
    def __init__(self, config_path, model_path ,device='cpu'):
        
        self.cfg = get_cfg()
        self.cfg.merge_from_file(config_path)   ####### "./config.yml"
        self.cfg.OUTPUT_DIR = model_path
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        self.cfg.MODEL.DEVICE = device          ###########  'cpu' 
        self.predictor = DefaultPredictor(self.cfg)
        
    def min_area_rectangle(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # draw the contours of c
        # print('c is: ',type(c))
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)  # turn into ints
        # rect2 = cv2.drawContours(image.copy(), [box], 0, (255,0,0), 2)
        return box,rect,c
    
    def find_objs_position(self, im, show_results=True):
        # im = cv2.imread(image_path)
        start_time = time.time()
        outputs = self.predictor(im)
        end_time = time.time() - start_time
        print('detection infer time is: ', end_time)
        #v = Visualizer(im[:, :, ::-1], metadata=balloon_metadata, scale=0.8)
        #v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        #plt.figure(figsize = (14, 10))
        #plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        #plt.show()
        
        pred_classes = np.array(outputs['instances'].pred_classes.data.cpu(), dtype=np.int32)
        pred_masks = np.array(outputs['instances'].pred_masks.data.cpu())
        pred_scores = np.array(outputs['instances'].scores.data.cpu())
        
        print('pred_classes is: ', pred_classes)
        print('scores is: ', pred_scores)
        #print(pred_masks.shape)
        
        # pred_classes = np.array([3,4,5]) for testing
        
        id_index = np.where(pred_classes==0) 
        date_index = np.where(pred_classes==1) 
        # num_index = np.where(pred_classes==2)
        
        if id_index[0].size == 0:
            # assign an empty image
            #city = np.full((2,2,3),250,dtype=np.uint8)
            id_img = False
            id_box = np.array([[1,1],[1,1],[1,1],[1,1]])
            
        else:
            id_index = id_index[0][0]  # take the first occurrance
            id_mask = pred_masks[id_index]
            id_mask = id_mask.astype(np.uint8)
            id_img = cv2.bitwise_and(im, im, mask=id_mask)
            id_box,_,_ = self.min_area_rectangle(id_img)
            #city = crop_polygon(im, city_poly)
            
        if date_index[0].size == 0:
            # assign an empty image
            #ch = np.full((2,2,3),250,dtype=np.uint8)
            date_img = False
            date_box = np.array([[1,1],[1,1],[1,1],[1,1]])
        else:
            date_index = date_index[0][0]  # take the first occurrance
            date_mask = pred_masks[date_index]
            date_mask = date_mask.astype(np.uint8)
            date_img = cv2.bitwise_and(im, im, mask=date_mask)
            date_box,_,_ = self.min_area_rectangle(date_img)
            #ch = crop_polygon(im, ch_poly)
            
        # if num_index[0].size == 0:
        #     # assign an empty image
        #     #num = np.full((2,2,3),250,dtype=np.uint8)
        #     num_img = False
        #     num_box = np.array([[1,1],[1,1],[1,1],[1,1]])
        # else:
        #     num_index = num_index[0][0]  # take the first occurrance
        #     num_mask = pred_masks[num_index]
        #     num_mask = num_mask.astype(np.uint8)
        #     num_img = cv2.bitwise_and(im, im, mask=num_mask)
        #     num_box,_,num_poly = self.min_area_rectangle(num_img)
        #     #num = crop_polygon(im, num_poly)
            
        # convert the mask to two dimensions 
        #city_mask = np.squeeze(city_mask, axis=0) if city_mask.shape[0] == 1 else None
        #ch_mask = np.squeeze(ch_mask, axis=0)
        #num_mask = np.squeeze(num_mask, axis=0)

        # find the images for each object
        # print('city_box is: ',city_box)
        
        if show_results:
            final_res = cv2.drawContours(im.copy(), [id_box], 0, (255,0,0), 1)
            final_res = cv2.drawContours(final_res.copy(), [date_box], 0, (255,0,0), 1)
            # final_res = cv2.drawContours(final_res.copy(), [num_box], 0, (255,0,0), 1)
            plt.imshow(final_res)
            plt.show()

        return id_img, date_img # , num_img


device = 'cpu'
config_path = './detection files/config.yml'
# Download the model from this url
# https://drive.google.com/drive/folders/13DTFAgE2l6kMWZCM4Y2EuxZNIw9RP4aU?usp=share_link
model_path = 'detection files'
testing_image_path = './val_img.jpg'
show_results = True


img = cv2.imread(testing_image_path)
objs_detection = Detection(config_path, model_path ,device)
id_img, date_image = objs_detection.find_objs_position(img, show_results)




