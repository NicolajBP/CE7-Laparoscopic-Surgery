import sys
from utils import *
import glob



if __name__ == '__main__':
    # video_file = "..\\..\\cholec80\\videos\\video19.mp4"
    video_file = "vehicles.mp4"
    video = cv.VideoCapture(video_file)

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
 
    # Read first frame.
    # for i in range(2800):
    for i in range(1):
        ok, frame = video.read()
        h,w = frame.shape[:2]
        rescale_factor = 4
        frame = cv.resize(frame, (int(w/rescale_factor), int(h/rescale_factor)))
    if not ok:
        print('Cannot read video file')
        sys.exit()

    vid_writer = cv.VideoWriter("tracker_res.mp4", fourcc=-1, fps=10, frameSize=frame.shape[1::-1])
    spt_map_writer = cv.VideoWriter("tracker_spt_map.mp4", fourcc=-1, fps=10, frameSize=frame.shape[1::-1])
    
    # choose ROI as initial bounding box
    bbox = cv.selectROI(frame, False) # [x, y, w, h]
    initial_patch = frame[int(bbox[1]):int(bbox[1] + bbox[3]), # y:(y+h)
                          int(bbox[0]):int(bbox[0] + bbox[2])] # x:(x+w)
    hog_writer = cv.VideoWriter("tracker_hog.mp4", fourcc=-1, fps=10, frameSize=bbox[2:])
    
    frame_cpy = frame.copy()
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv.rectangle(frame_cpy, p1, p2, (255, 0, 0), 2, 1)
    vid_writer.write(frame_cpy)
    # extract features from patch of bounding box
    features_hog_desc, features_hog = extract_features(initial_patch)
    visualize_HOG(initial_patch, features_hog)
    hog_writer.write(np.hstack((cv.cvtColor(initial_patch, cv.COLOR_BGR2GRAY), 255*features_hog.astype('uint8'))))

    # initialize filter
    filter = init_filter(features_hog_desc)

    fg_hist, bg_hist = extract_fg_bg_histograms(frame, bbox)

    scales = [0.9, 1, 1.1] # for bbox w and h change
    while True:
        ok, frame = video.read()
        frame = cv.resize(frame, (int(w/rescale_factor), int(h/rescale_factor)))
        if not ok:
            break

        search_window_size = (bbox[2]//3, bbox[3]//3) # (w,h) around bbox for search window

        # define the search window around prev position
        bbox_s = [max(0, bbox[0] - search_window_size[0]), # x
                  max(0, bbox[1] - search_window_size[1]), # y
                  min(frame.shape[1] - bbox[0], bbox[2] + 2*search_window_size[0]), # w
                  min(frame.shape[0] - bbox[1], bbox[3] + 2*search_window_size[1])] # h
        
        # get search window of frame
        search_window = frame[int(bbox_s[1]):int(bbox_s[1] + bbox_s[3]),
                              int(bbox_s[0]):int(bbox_s[0] + bbox_s[2])]
        
        # Calculate the probability maps for foreground and background
        curr_patch = frame[int(bbox[1]):int(bbox[1] + bbox[3]), # y:(y+h)
                           int(bbox[0]):int(bbox[0] + bbox[2])] # x:(x+w)
        fg_prob_map = calculate_color_probability(search_window, fg_hist, bins=(16, 16, 16))
        bg_prob_map = calculate_color_probability(search_window, bg_hist, bins=(16, 16, 16))
        # cv.namedWindow("Foreground | Background probability map", cv.WINDOW_NORMAL) 
        # cv.resizeWindow("Foreground | Background probability map", 600, 300)
        cv.imshow("Foreground | Background probability map",
                  np.hstack((fg_prob_map.astype('uint8'), np.zeros((bbox_s[3], 5), np.uint8),
                             bg_prob_map.astype('uint8'))))

        # Compute the spatial reliability map
        spatial_reliability_map = compute_spatial_reliability(fg_prob_map, bg_prob_map)
        # morphological closing to fill the object in spatial map
        spatial_reliability_map = cv.morphologyEx(spatial_reliability_map, cv.MORPH_CLOSE, kernel=np.ones((3,3),np.uint8))
        
        weighted_search = search_window * np.repeat(spatial_reliability_map[:,:, np.newaxis], 3, axis=2)
        # cv.namedWindow("Spatial reliability map", cv.WINDOW_NORMAL)
        # cv.resizeWindow("Spatial reliability map", 300, 300)
        spatial_reliability_map = (255*spatial_reliability_map).astype('uint8')
        cv.imshow("Spatial reliability map", spatial_reliability_map)
        spt_map_writer.write(spatial_reliability_map)
        
        # extract features of frame
        search_features_hog_desc, search_features_hog = extract_features(search_window)
            # extract_features(weighted_search)
            # extract_features(search_window)
            
        # apply filter to track the object
        response = apply_filter(filter, search_features_hog_desc)

        # find estimated location of object
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(response)
        top_left = max_loc
        w_hog, h_hog = filter.shape[1::-1]
        w_patch, h_patch = initial_patch.shape[1::-1]
        bbox = (int((top_left[0]+0.5) * w_patch/w_hog) + bbox_s[0],
                int((top_left[1]+0.5) * h_patch/h_hog) + bbox_s[1],
                w_patch,
                h_patch)

        # Step 1: Get rescaled patches for multiple scales
        rescaled_patches = get_rescaled_patches(frame, bbox, scales)

        # Step 2: Apply correlation and find the best scale
        best_scale_idx = find_best_scale(filter, rescaled_patches)

        # Step 3: Update the bounding box with the best scale
        bbox = update_bbox_scale(bbox, best_scale_idx, scales)

        # Step 4: Update the filter using the best scale patch
        new_patch = rescaled_patches[best_scale_idx]

        # update filter by new features
        # new_patch = frame[int(bbox[1]):int(bbox[1] + bbox[3]), # y:(y+h)
        #                   int(bbox[0]):int(bbox[0] + bbox[2])] # x:(x+w)
        new_features_hog_desc, new_features_hog = extract_features(new_patch)

        # pad or crop the filter to be compatible with the search window
        if np.any(filter.shape[:2] < new_features_hog_desc.shape[:2]):
            zeros_pad_size = (np.array(new_features_hog_desc.shape[:2]) - np.array(filter.shape[:2]))/2
            filter = np.pad(filter, ((np.floor(zeros_pad_size[0]).astype('int'), np.ceil(zeros_pad_size[0]).astype('int')),
                                     (np.floor(zeros_pad_size[1]).astype('int'), np.ceil(zeros_pad_size[1]).astype('int')), (0,0),(0,0),(0,0)), 'constant')
        elif np.any(filter.shape[:2] > new_features_hog_desc.shape[:2]):
            filter = filter[:new_features_hog_desc.shape[0], :new_features_hog_desc.shape[1],:,:,:]
        
        filter = update_filter(filter, new_features_hog_desc)

        # draw new bounding box
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        frame_cpy = frame.copy()
        cv.rectangle(frame_cpy, p1, p2, (255, 0, 0), 2, 1)

        # display result
        cv.imshow("Search window", search_window)
        cv.imshow("Tracking", frame_cpy)
        visualize_HOG(new_patch, new_features_hog, cv_flag=True)
        # Exit if ESC pressed
        k = cv.waitKey(3) & 0xff
        vid_writer.write(frame_cpy)
        hog_writer.write(np.hstack((cv.cvtColor(new_patch, cv.COLOR_BGR2GRAY), 255*new_features_hog.astype('uint8'))))
        if k == 27 : break
        if k == ord("p"):
            cv.waitKey(0)
        # if k == ord("s"):
        #     cv.imwrite("Frame_rectangle.png", frame)
    vid_writer.release()
    spt_map_writer.release()
    hog_writer.release()