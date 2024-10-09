import os, sys, shutil
import random


if __name__ == "__main__":
    base_dataset_path = "../datasets_rob/Bridge_v1_raw"
    test_store_path = "../datasets_rob/Bridge_v1_test_raw"
    split_ratio = 0.1       # [0, 1] range

    # Prepare the folder
    if os.path.exists(test_store_path):
        shutil.rmtree(test_store_path)
    os.makedirs(test_store_path)

    full_img_lists = os.listdir(base_dataset_path)
    random.shuffle(full_img_lists)
    target_test_length = int(len(full_img_lists) * split_ratio)
    test_img_lists = full_img_lists[-1 * target_test_length : ]

    # Move the lists based on test_img_lists
    for test_img_name in test_img_lists:
        shutil.move(os.path.join(base_dataset_path, test_img_name), os.path.join(test_store_path, test_img_name))

