from parseCsv import *
from performHungarianMatching import *
from drawDetectionsOnVideo import *

def main():
    video_number = "13"
    detections_file = f'tracker_res_video{video_number}.csv'  # CSV file containing bboxes (and APMs) from Asafs script "ExtractLaparoscopicAPM.py"
    # detections_file = 'tracker_res_video01_1minute.csv'  # CSV file containing bboxes (and APMs) from Asafs script "ExtractLaparoscopicAPM.py"

    video_path = f'YoloFinetuning/video{video_number}.mp4'  
    # video_path = 'ByteTrack/video01_1minute.mp4'  
    

    # output_path_1 = 'output_video_with_ids_overlay.mp4'  # Path to save the video with overlay from CSV
    output_path_2 = f'output_video{video_number}_with_hungarian_matching.mp4'  # Path to save the video with Hungarian matching
    
    # Parse detections from CSV
    detections = parse_csv(detections_file)
    
    # Video 1: Overlay IDs and bounding boxes from CSV
    # draw_detections_on_video(video_path, detections, output_path_1)
    # print(f"Video with overlay from CSV saved to {output_path_1}")
    
    # Video 2: Apply Hungarian matching and overlay
    matched_detections, matches = perform_hungarian_matching(detections)
    print("Drawing bounding boxes and track IDs on video")
    # draw_detections_on_video(video_path, matched_detections, output_path_2, matches)
    # print(f"Video with Hungarian matching saved to {output_path_2}")

if __name__ == '__main__':
    main()