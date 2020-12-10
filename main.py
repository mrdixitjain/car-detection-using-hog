import os
import math
import glob
import cv2
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
from moviepy.editor import VideoFileClip
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from scipy.ndimage.measurements import label
from DetectHistory import Detect_history
from Processing import *
from GetData import *
from GetFeatures import *
from Train import *
import pickle

### Parameters
spatial = 32
hist_bins = 32
colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb #YCrCb best
orient = 9
pix_per_cell = 8
cell_per_block = 2
spatial_size= (32, 32)
heat_threshold= 4 # 12
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL" #ALL,0 best
ystart_ystop_scale = [(405, 510, 1), (400, 600, 1.5), (500, 710, 2.5)]

def process_image(img):     
    # Using Subsampled HOG windows to get possible detections 
    bbox_detection_list = find_cars(img, svc, X_scaler, orient, pix_per_cell, cell_per_block, 
                                      spatial_size, hist_bins, ystart_ystop_scale)
    blank = np.zeros_like(img[:,:,0]).astype(np.float)
    # Smoothing out previous detections
    detect_history.put_labels(bbox_detection_list)
    bbox_detection_list = detect_history.get_labels()

    # Add heat to detections
    heatmap = add_heat(blank, bbox_detection_list)
    heatmap = apply_threshold(heatmap, 7)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    # Draw bounding box 
    result = draw_labeled_bboxes(np.copy(img), labels)
    return result

if __name__=='__main__' :
	print("Please read requirement.txt to get all requirement")
	print("If done than you can comment out these lines")
	print()
	car_images, non_car_images, test_images = getData()

	if(not os.path.isfile("params.p")) :
		train_svc_and_save(car_images, non_car_images)

	print()
	print('Loading Classifier parameters...')
	# opening params.p file
	data = pickle.load( open("params.p", "rb" ) )
	svc = data["svc"]
	X_scaler = data["scaler"]
	orient = data["orient"]
	pix_per_cell = data["pix_per_cell"]
	cell_per_block = data["cell_per_block"]
	spatia = data["spatial"]
	hist_bins = data["hist_bins"]
	print('Loading is done!')

	detect_history = Detect_history()
	project_video_res = 'project_video_result.mp4'
	clip1 = VideoFileClip("project_video.mp4")
	project_video_clip = clip1.fl_image(process_image)
	project_video_clip.write_videofile(project_video_res, audio=False)

