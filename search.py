import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sliding_window import *
from feature_extraction import *
from image_processing import *
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
import glob
import pickle
import os


color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL'# Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
x_start_stop = [None, None] # Min and max in x to search in slide_window()

ystart = 400
ystop = 656
scale = 1.5


def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    on_windows = []
    #Iterate over all windows in the list
    for window in windows:
        #Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        #Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #Predict using your classifier
        prediction = clf.predict(test_features)
        #If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #Return windows for positive detections
    return on_windows

def search_subsample(img, ystart, ystop, scale, svc, X_scaler,
                      orient, pix_per_cell, cell_per_block, spatial_size, 
                      hist_bins):
    return (feature_subsample(img, ystart, ystop, scale, svc, X_scaler, orient,
                      pix_per_cell, cell_per_block, spatial_size, hist_bins))
    

def train(img_slice=slice(None)):    
    vehicle_images = glob.glob("../vehicles/**/*.png")
    non_vehicle_images = glob.glob("../non-vehicles/**/*.png")
    
    print('Extracting features')    
    car_features = extract_features(vehicle_images, file_type="png",
                                    color_space=color_space,
                                    spatial_size=spatial_size,
                                    hist_bins=hist_bins,
                                    orient=orient,
                                    pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel,
                                    spatial_feat=spatial_feat, 
                                    hist_feat=hist_feat,
                                    hog_feat=hog_feat)
    notcar_features = extract_features(non_vehicle_images, file_type="png",
                                       color_space=color_space,
                                       spatial_size=spatial_size,
                                       hist_bins=hist_bins,
                                       orient=orient,
                                       pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel,
                                       spatial_feat=spatial_feat, 
                                       hist_feat=hist_feat,
                                       hog_feat=hog_feat)
    
    X = np.vstack((car_features, notcar_features)).astype(np.float64) 
    print("Scaling features")

    features, scaler = scale_training_features(X)

    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    rand = np.random.randint(0, 100)

    print("Splitting features")
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        y,
        test_size=0.4,
        random_state=rand)
    
    print("Feature vector lenght:", len(X_train[0]))

    svc = LinearSVC()
    
    # Check the training time for the SVC
    t=time.time()

    print("Fitting data to classifier")
    svc.fit(X_train, y_train)
    
    
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')

    print("Getting classifier score")
    score = svc.score(X_test, y_test)

    print("Score: {0:.4f}".format(score))
    
    # Check the prediction time for a single sample
    t=time.time()
    
    n_predict = 1000
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
    print(accuracy_score(y_test, svc.predict(X_test)))
    
    # save the model to disk
    filename = 'finalized_model.pkl'
    
    model_data = {
        "clf": svc,
        "scaler": scaler
    }
    
    pickle.dump(model_data, open(filename, 'wb'))

def test(): 
    # load the model from disk
    filename = 'finalized_model.pkl'
    loaded_model = pickle.load(open(filename, 'rb'))
    
    images = glob.glob('../../CarND-Vehicle-Detection/test_images/test*.jpg')
    
    for index, fname in enumerate(images):
        image = mpimg.imread(fname)
        
        draw_image = np.copy(image)
        
        heat = np.zeros_like(image[:,:,0]).astype(np.float)
        
        y_start_stop = [400, 656] # Min and max in y to search in slide_window()
        # extracted training data from .png images (scaled 0 to 1 by mpimg) and the
        # image you are searching is a .jpg (scaled 0 to 255)
        #image = image.astype(np.float32)/255
        
        windows = sliding_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop, 
                        xy_window=(96, 96), xy_overlap=(0.5, 0.5))
    
        hot_windows = search_windows(image, windows, loaded_model["clf"], loaded_model["scaler"], color_space=color_space, 
                                spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                hist_feat=hist_feat, hog_feat=hog_feat)    
    
        # Add heat to each box in box list
        heat = add_heat(heat,hot_windows)
            
        # Apply threshold to help remove false positives
        heat = apply_threshold(heat,1)
        
        # Visualize the heatmap when displaying    
        heatmap = np.clip(heat, 0, 255)
        
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        window_img = draw_labeled_bboxes(draw_image, labels)
    
        #window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    
           
        plt.imshow(window_img)
        write_name = 'labeled_box'+str(index)+'.jpg'
        temp = cv2.cvtColor(window_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(dirname1,write_name), temp)
  
def test_subsample():
    # load the model from disk
    filename = 'finalized_model.pkl'
    loaded_model = pickle.load(open(filename, 'rb'))  
    
    images = glob.glob('../../CarND-Vehicle-Detection/test_images/test*.jpg')
    
    for index, fname in enumerate(images):
        image = mpimg.imread(fname)
    
        draw_image = np.copy(image)
        
        out_image, heat = search_subsample(image, ystart, ystop, scale,
                                     loaded_model["clf"], loaded_model["scaler"],
                                     orient, pix_per_cell, cell_per_block,
                                     spatial_size, hist_bins)
        
        heat = apply_threshold(heat, 1)
        # Visualize the heatmap when displaying    
        heatmap = np.clip(heat, 0, 255)
        
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        
        window_img = draw_labeled_bboxes(draw_image, labels)
        
        plt.imshow(window_img)
        write_name = 'subsample_window_test'+str(index)+'.jpg'
        temp = cv2.cvtColor(window_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(dirname1,write_name), temp)

if __name__ == "__main__":
    
    dirname1 = 'output_images'
    if not os.path.exists(dirname1):
        os.mkdir(dirname1)
