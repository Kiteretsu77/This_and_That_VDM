'''
    Sometimes, Bridge dataset will contain strange downloads, we need to clean them
'''
import os, shutil

# TODO: 后面把这个直接merge 到prepare_bridge_dataset中
if __name__ == "__main__":
    dataset_path = "/nfs/turbo/jjparkcv-turbo-large/boyangwa/Bridge"

    for sub_folder in sorted(os.listdir(dataset_path)):
        sub_folder_path = os.path.join(dataset_path, sub_folder)

        img_lists = os.listdir(sub_folder_path)
        if len(img_lists) < 14:
            print("The folder is too short, we will remove them all")
            shutil.rmtree(sub_folder_path)
            continue
        for img_name in img_lists:
            img_path = os.path.join(sub_folder_path, img_name)
            if not img_name.startswith("im_"):
                print("We remove ", img_path)
                os.remove(img_path)