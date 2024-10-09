'''
    This file is to make sure that the video files is readeable by moviepy, such that the data loader can read these files.
'''
import os
from moviepy.editor import VideoFileClip

if __name__ == "__main__":
    video_dir = "../webvid_sample"
    delete_abnormal_video = True    # Whether you want to delete these abnormal video directly

    for video_name in sorted(os.listdir(video_dir)):
        video_path = os.path.join(video_dir, video_name)
        try:
            objVideoreader = VideoFileClip(filename=video_path)
        except Exception:
            print("There is an exception of reading: ", video_path)
            if delete_abnormal_video:
                print("We will remove this abnormal video source")
                os.remove(video_path)