'''
    This file is to match the selected frames with the bridge dataset
    We need to use some tricks to select the item
'''
import os, sys, shutil
import cv2
import numpy as np




def compare_img(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err



def search_path(dataset_path, target_path, store_txt_path):

    # We only needs to care about Bridge v1 dataset area 
    target_img_path = os.path.join(target_path, "im_0.jpg")
    if not os.path.exists(target_img_path):
        print("The image we read is False")
        return False
    target_img = cv2.imread(target_img_path)

    # Iterate all the folders inside
    for scene_name in sorted(os.listdir(dataset_path)):
        scene_dir = os.path.join(dataset_path, scene_name)

        for task_name in sorted(os.listdir(scene_dir)):
            task_dir = os.path.join(scene_dir, task_name)

            for order_name in sorted(os.listdir(task_dir)):
                order_dir = os.path.join(task_dir, order_name)

                for time_clock in sorted(os.listdir(order_dir)):
                    if time_clock == "lmdb":
                        continue    # Skip lmdb folder
                    
                    time_dir = os.path.join(order_dir, time_clock, "raw", "traj_group0")
                    if not os.path.exists(time_dir):
                        continue

                    for traj_name in sorted(os.listdir(time_dir)):
                        traj_path = os.path.join(time_dir, traj_name)
                        if not os.path.isdir(traj_path):
                            continue
                        
                        # Directly move policy_out_file_path; just in case there is also valuable information there
                        policy_out_file_path = os.path.join(traj_path, "policy_out.pkl")
                        if not os.path.exists(policy_out_file_path):
                            continue

                        # Check the lang txt file
                        lang_txt_file_path = os.path.join(traj_path, "lang.txt")
                        if not os.path.exists(lang_txt_file_path):
                            continue


                        for img_name in sorted(os.listdir(traj_path)):
                            if img_name != "images0":       # Only consider one camera angle
                                continue

                            img_folder_path = os.path.join(traj_path, img_name)
                            if not os.path.isdir(img_folder_path):
                                continue


                            # Compare two image 
                            img_path = os.path.join(img_folder_path, "im_0.jpg")
                            if not os.path.exists(img_path):
                                print(img_folder_path + " doesn't even have im_0.jpg")
                                continue
                            # print("img_folder_path is ", img_path)
                            compare_sample_img = cv2.imread(img_path)
                            # try:
                            #     compare_sample_img.shape
                            # except Exception:
                            #     print("The compare_sample_img cannot be red")
                            #     continue
                            error = compare_img(target_img, compare_sample_img)

                            if error == 0:
                                # Continue to all the rest for at least 5 images
                                status = True
                                for idx in range (10):
                                    idx_img_path = os.path.join(img_folder_path, "im_"+str(idx)+".jpg")
                                    idx_target_img_path = os.path.join(target_path, "im_"+str(idx)+".jpg")
                                    if not os.path.exists(idx_img_path):
                                        print("The idx_img_path long idx we see only at ", idx)
                                        continue
                                    if not os.path.exists(idx_target_img_path):
                                        print("The idx_target_img_path long idx we see only at ", idx)
                                        continue
                                    idx_compare_sample_img = cv2.imread(idx_img_path)
                                    idx_target_img = cv2.imread(idx_target_img_path)
                                    error = compare_img(idx_target_img, idx_compare_sample_img)

                                    if error != 0:
                                        status = False
                                        break
                                    
                                if status:
                                    print("We found one at ", img_path)
                                    f = open(store_txt_path, "a")
                                    f.write(target_path + " " + img_folder_path + "\n")
                                    return True
                       
    return False


if __name__ == "__main__":
    input_path = "/nfs/turbo/jjparkcv-turbo-large/boyangwa/datasets_rob/Bridge_v2_test_raw"
    dataset_path = "/nfs/turbo/jjparkcv-turbo-large/boyangwa/raw/bridge_data_v2"       # 直接从本地新unzip的获取，怕之前的被xuweiyi改动过
    store_txt_path = "match_info_v2_p1.txt"
    start_idx = 0
    end_idx = 500

    if os.path.exists(store_txt_path):
        os.remove(store_txt_path)

    for img_name in sorted(os.listdir(input_path))[start_idx:end_idx]:
        target_path = os.path.join(input_path, img_name)
        print("We are finding for ", target_path)

        status = search_path(dataset_path, target_path, store_txt_path)
        
        if not status:
            print("we cannot find one")