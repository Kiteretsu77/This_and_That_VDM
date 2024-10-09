'''
    THis file is to collect all lang.txt and move to a new directory, this is for the convenience to compress and scp the lang for post-processing
'''
import os, sys, shutil

if __name__ == "__main__":
    parent_dir = "../datasets_rob"
    dataset_paths = ["Bridge_v1_TT14", "Bridge_v2_TT14"]
    store_folder = "../full_text_tmp"

    # Manage the store folder
    if os.path.exists(store_folder):
        shutil.rmtree(store_folder)
    os.makedirs(store_folder)
    

    for dataset_name in dataset_paths:
        store_path = os.path.join(store_folder, dataset_name)
        if os.path.exists(store_path):
            shutil.rmtree(store_path)
        os.makedirs(store_path)

        # Iterate all the files
        for sub_folder_name in os.listdir(os.path.join(parent_dir, dataset_name)):
            print("We are processing ", sub_folder_name)
            lang_txt_path = os.path.join(parent_dir, dataset_name, sub_folder_name, "lang.txt")

            # Store on the new address
            store_file_path = os.path.join(store_path, sub_folder_name)
            os.makedirs(store_file_path)
            shutil.copyfile(lang_txt_path, os.path.join(store_file_path, "lang.txt"))
