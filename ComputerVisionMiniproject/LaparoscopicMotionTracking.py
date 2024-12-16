import cv2 as cv
import sys


if __name__ == '__main__' :

    # Set up tracker.
    tracker_types = ['BOOSTING', 'MIL','KCF', 'GOTURN', 'CSRT']
    tracker_type = tracker_types[-1]
    if tracker_type == 'BOOSTING':
        tracker = cv.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv.TrackerKCF_create()
    if tracker_type == 'GOTURN':
        tracker = cv.TrackerGOTURN_create()
    if tracker_type == "CSRT":
        tracker = cv.TrackerCSRT_create()
 
    # Read video
    # video_file = "cholec80\\videos\\video19.mp4"
    video_file = "vehicles.mp4"
    video = cv.VideoCapture(video_file)
    rescale_factor = 4

 
    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
 
    # Read first frame.
    # for i in range(2700):
    for i in range(1):
        ok, frame = video.read()
        h,w = frame.shape[:2]
        frame = cv.resize(frame, (int(w/rescale_factor), int(h/rescale_factor)))
    if not ok:
        print('Cannot read video file')
        sys.exit()
     
    vid_writer = cv.VideoWriter("CSRT_res.mp4", fourcc=-1, fps=10, frameSize=frame.shape[1::-1])
    # Define an initial bounding box
    bbox = cv.selectROI(frame, False)
 
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    vid_writer.write(frame)
    
    frame_counter = 0
    while True:
        frame_counter += 1
        # Read a new frame
        ok, frame = video.read()
        frame = cv.resize(frame, (int(w/rescale_factor), int(h/rescale_factor)))
        if not ok:
            break
         
        # Start timer
        timer = cv.getTickCount()
 
        # Update tracker
        ok, bbox = tracker.update(frame)
 
        # Calculate Frames per second (FPS)
        fps = cv.getTickFrequency() / (cv.getTickCount() - timer)
 
        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else :
            # Tracking failure
            cv.putText(frame, "Tracking failure detected", (100,80), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
 
        # Display tracker type on frame
        cv.putText(frame, tracker_type + " Tracker", (100,20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
     
        # Display FPS on frame
        cv.putText(frame, "FPS : " + str(int(fps)), (100,50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
 
        # Display result
        cv.imshow("Tracking", frame)
        # sv.plot_image(frame)
        vid_writer.write(frame)

        # Exit if ESC pressed
        k = cv.waitKey(40) & 0xff
        if k == 27 : break
        if k == ord("s"):
            cv.imwrite("Frame_" + str(frame_counter) + ".png", frame)
    vid_writer.release()
