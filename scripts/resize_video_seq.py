'''
    This file is designed to resize the video sequence to the target resolution
'''
import os, sys, shutil
import cv2

if __name__ == "__main__":
    input_folder = "/nfs/turbo/jjparkcv-turbo-large/boyangwa/model_results/SVD_results"
    store_path = "/nfs/turbo/jjparkcv-turbo-large/boyangwa/model_results/SVD_results_resized"    
    target_height, target_width = 256, 384

    if os.path.exists(store_path):
        shutil.rmtree(store_path)
    os.makedirs(store_path)

    for video_name in sorted(os.listdir(input_folder)):
        print("We are processing ", video_name)
        sub_video_folder = os.path.join(input_folder, video_name)
        sub_store_folder = os.path.join(store_path, video_name)
        os.makedirs(sub_store_folder)

        for img_name in os.listdir(sub_video_folder):
            if not img_name.endswith("jpg") and not img_name.endswith("png"):
                continue

            img_path = os.path.join(sub_video_folder, img_name)
            store_img_path = os.path.join(sub_store_folder, img_name)
            img = cv2.imread(img_path)

            # Resize 
            img = cv2.resize(img, (target_width, target_height))
            cv2.imwrite(store_img_path, img)
            
