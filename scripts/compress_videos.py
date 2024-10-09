import os, shutil, sys
from moviepy.editor import ImageSequenceClip


def compress_video(sub_folder_path, video_name):
    store_path = os.path.join(sub_folder_path, video_name)

    if os.path.exists(store_path):
        os.remove(store_path)


    # Check valid length
    all_files = os.listdir(sub_folder_path)
    num_frames_input = 0
    valid = True
    for file_name in os.listdir(sub_folder_path):
        if file_name.startswith("im_"):
            num_frames_input += 1
    for idx in range(num_frames_input):
        img_path = 'im_' + str(idx) + '.jpg'
        if img_path not in all_files:            # Should be sequential existing
            valid = False
            break
    if not valid:
        print("We cannot generate a video because the video is not sequential")
        return False
    

    if num_frames_input == 0:
        print("We cannot generate a video because the input length is 0")
        return False

    img_lists = []
    for idx in range(num_frames_input):
        img_path = os.path.join(sub_folder_path, "im_" + str(idx) + ".jpg")
        img_lists.append(img_path)
    
    clip = ImageSequenceClip(img_lists, fps=4)
    clip.write_videofile(store_path)

    return True


if __name__ == "__main__":
    dataset_path = "../datasets_rob/Bridge_v2_raw"  # ../datasets_rob/Bridge_v1_raw

    for sub_folder_name in sorted(os.listdir(dataset_path)):
        sub_folder_path = os.path.join(dataset_path, sub_folder_name)

        status = compress_video(sub_folder_path)

       


        