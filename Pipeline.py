
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from sliding_window import *
from extract_feature import *
from image_proc import *
import numpy as np

color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL'# Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
ystart = 400
ystop = 656

filename = 'finalized_model.pkl'
loaded_model = pickle.load(open(filename, 'rb')) 
svc = loaded_model["clf"]
X_scaler = loaded_model["scaler"]

class History():
    def __init__(self, history_length):
        # Length of history to keep for heatmaps
        self.history_length = history_length 
        # List of recent heatmaps to average.
        self.recent_heatmaps = []

def detect(img, scale):
        
    draw_img = np.copy(img)
    
    # Crop the image so that it only looks at the road and ignores the sky.
    img_tosearch = img[ystart:ystop,:,:]
    # Convert the cropped image to the given color space.
    ctrans_tosearch = rgb_to_colorspace(img_tosearch, color_space=color_space)
    
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    # Boxes that have a positive match
    boxes = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            # Where are we in pixel space for this particular cell
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                box = (xbox_left, ytop_draw + ystart), (xbox_left +win_draw, ytop_draw + win_draw + ystart)
                
                boxes.append(box)
    return draw_img, boxes

def pipeline(img):
    scale = 1.25
    out_image, box = detect(img, scale)
    
    # Make a heatmap from the bounding boxes
    heatmap = np.zeros_like(out_image[:,:,0])
    heatmap = add_heat(heatmap, box)
    heatmap = apply_threshold(heatmap, 7)
    
    # Add heatmap to recent history
    history.recent_heatmaps.append(heatmap)
    
    if len(history.recent_heatmaps) > history.history_length:
        # Remove the oldest map from the history.
        history.recent_heatmaps = history.recent_heatmaps[1:]
        
    # Get the total heatmap over n frames
    sum_heatmap = np.sum(history.recent_heatmaps, axis=0)
    # Filter out predictions that happened for 4 or less frames over n frames.
    thres_heatmap = apply_threshold(sum_heatmap, 4)
    
    labels = label(thres_heatmap)
    
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    
    return draw_img
    
def process_image(img):
    return(pipeline(img))

if __name__ == "__main__":
    
    test_output = 'output_project_video.mp4'

    history_length = 5
    history = History(history_length)

    clip = VideoFileClip("../../CarND-Vehicle-Detection/project_video.mp4")
   
    test_clip = clip.fl_image(process_image)
    
    test_clip.write_videofile(test_output, audio=False)