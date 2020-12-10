import numpy as np 
import cv2
from skimage.feature import hog
import matplotlib.image as mpimg

#Resizing Image to extract features, so as to reduce the feature vector size
def SpatialBinningFeatures(image, size=(32, 32)):
    features = cv2.resize(image,size)
    return features.ravel()

# Compute color histogram features  
def color_histogram(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    hist1 = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    hist2 = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    hist3 = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    hist_features = np.concatenate((hist1[0], hist2[0], hist3[0]))
    return hist_features

# General method to extract the HOG of the image
def GetFeaturesFromHog(image, orient, pixelsPerCell, cellsPerBlock, visualise=False,
                     feature_vector_flag=True):
    if(visualise==True):
        hog_features, hog_image = hog(image, orientations=orient,
                          pixels_per_cell=(pixelsPerCell, pixelsPerCell), 
                          cells_per_block=(cellsPerBlock, cellsPerBlock), 
                          visualize=True, feature_vector=feature_vector_flag)
        return hog_features, hog_image
    else:
        hog_features = hog(image, orientations=orient,
                          pixels_per_cell=(pixelsPerCell, pixelsPerCell), 
                          cells_per_block=(cellsPerBlock, cellsPerBlock), 
                          visualize=False, feature_vector=feature_vector_flag)
        return hog_features

# Extract feature wrapper that extracts and combines all features
def extract_features(imgs, cspace='RGB', orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
    # it'll hold feature vectors
    features = []

    for file in imgs:
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB' 
            
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: 
            feature_image = np.copy(image)   

        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(GetFeaturesFromHog(feature_image[:,:,channel], 
                                orient, pix_per_cell, cell_per_block, 
                                False, True))
        hog_features = np.ravel(hog_features)
        
        spatial_features = SpatialBinningFeatures(feature_image, size=spatial_size)
        hist_features = color_histogram(feature_image, nbins=hist_bins, bins_range=hist_range)
        features.append(np.concatenate((spatial_features, hist_features, hog_features)))        
    return features