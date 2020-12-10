import numpy as np 
import os
import glob
import matplotlib.pyplot as plt

def getData() :
	test_images = np.array([plt.imread(i) for i in glob.glob('./test_images/*.jpg')])

	car_images = []
	non_car_images = []

	for root, dirs, files in os.walk('./dataset/vehicles/'):
	    for file in files:
	        if file.endswith('.png'):
	            car_images.append(os.path.join(root, file))
	            
	for root, dirs, files in os.walk('./dataset/non-vehicles/'):
	    for file in files:
	        if file.endswith('.png'):
	            non_car_images.append(os.path.join(root, file))

	return car_images, non_car_images, test_images
