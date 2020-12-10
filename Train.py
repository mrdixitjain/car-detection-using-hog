import cv2
import numpy as np
import matplotlib.image as mpimg
from GetFeatures import *
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
import pickle

def train_svc_and_save(car_images, non_car_images) :
	print("training svc")
	print()

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
	ystart_ystop_scale = [(400, 464, 1), (400, 480, 1.5), (500, 612, 2.5), (550, 760, 3)]

	car_features = extract_features(car_images, cspace=colorspace, orient=orient, 
	                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
	                        hog_channel=hog_channel, spatial_size=(spatial, spatial),
	                        hist_bins=hist_bins, hist_range=(0, 256))

	non_car_features = extract_features(non_car_images,cspace=colorspace,orient=orient, 
	                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
	                        hog_channel=hog_channel, spatial_size=(spatial, spatial),
	                        hist_bins=hist_bins, hist_range=(0, 256))

	# Create an array stack of feature vectors
	X = np.vstack((car_features, non_car_features)).astype(np.float64)                        
	# Fit a per-column scaler
	X_scaler = StandardScaler().fit(X)
	# Apply the scaler to X
	scaled_X = X_scaler.transform(X)

	# Define the labels vector
	y = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))

	# Split up data into randomized training and test sets
	X_train, X_test, y_train, y_test = train_test_split(
	    scaled_X, y, test_size=0.2, shuffle=True)

	print('Using spatial binning of:',spatial,'and', hist_bins,'histogram bins')
	print('Feature vector length:', len(X_train[0]))


	# Use a linear SVC 
	svc = LinearSVC()

	# Check the training time for the SVC
	t1=time.time()
	svc.fit(X_train, y_train)
	t2 = time.time()

	print(round(t2-t1, 2), ' Seconds to train SVC...')
	# Check the score of the SVC
	print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

	# Save data to file
	data = {}
	data["svc"] = svc
	data["scaler"] = X_scaler
	data["orient"] = orient
	data["pix_per_cell"] = pix_per_cell
	data["cell_per_block"] = cell_per_block
	data["spatial"] = spatial
	data["hist_bins"] = hist_bins
	
	pickle.dump(data, open("params.p", 'wb') ) 

	print('Classifier parameters saved to file!')
	print()