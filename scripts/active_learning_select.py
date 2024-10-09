import os, shutil
import random


if __name__ == "__main__":
    start_idx = 950
    end_idx = 1020
    select_num = 70

    label_start_idx = 632
    input_parent_dir = "../Bridge"
    store_dir = "../bridge_select3"

    if os.path.exists(store_dir):
        shutil.rmtree(store_dir)
    os.makedirs(store_dir)

    for idx in range(start_idx, end_idx):
        folder_path = os.path.join(input_parent_dir, str(idx))
        select_idx = random.randint(0, len(os.listdir(folder_path)))
        for idx, img_name in enumerate(os.listdir(folder_path)):
            if idx == select_idx and img_name != "policy_out.pkl":
                img_path = os.path.join(folder_path, img_name)
                target_path = os.path.join(store_dir, str(label_start_idx) + ".jpg")
                label_start_idx += 1
                shutil.copy(img_path, target_path)
                
