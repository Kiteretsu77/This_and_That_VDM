'''
    This file is to prepare the dataset in csv file following the format required by Opne-SORA
'''

import os, sys, shutil
import json
import csv

# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
# from curation_pipeline.prepare_bridge_v1 import read_bridge_v1
# from curation_pipeline.prepare_bridge_v2 import read_bridge_v2



def iter_dataset(dataset_path):
    lists = []
    for sub_folder_name in os.listdir(dataset_path):
        sub_folder_path = os.path.join(dataset_path, sub_folder_name)

        # Check number of frames
        max_length = len(os.listdir(sub_folder_path))
        for check_idx in range(max_length):
            if not os.path.exists(os.path.join(sub_folder_path, 'im_' + str(check_idx) + '.jpg')):  # Should be sequentially exists
                break
        num_frames = check_idx

        # Read the text
        txt_path = os.path.join(sub_folder_path, "lang.txt")
        f = open(txt_path, "r")
        lang_prompt = f.readline()

        lists.append([sub_folder_path, lang_prompt, num_frames, 480, 640])
        # break
    return lists



if __name__ == "__main__":
    v1_dataset_path = "/nfs/turbo/jjparkcv-turbo-large/boyangwa/sanity_check/bridge_v1_raw"
    v2_dataset_path = "/nfs/turbo/jjparkcv-turbo-large/boyangwa/sanity_check/bridge_v2_raw"
    store_name = "Bridge_raw.csv"

    if os.path.exists(store_name):
        os.remove(store_name)
    

    # Execute
    full_lists = [["path", "text", "num_frames", "height", "width"]]

    v1_lists = iter_dataset(v1_dataset_path)
    full_lists.extend(v1_lists)
    v2_lists = iter_dataset(v2_dataset_path)
    full_lists.extend(v2_lists)
    print("Full length is ", len(full_lists))


    # Store as csv file
    with open(store_name, 'w') as outfile:
        write = csv.writer(outfile)
        write.writerows(full_lists)



    # with open('output.jsonl', 'w') as outfile:
    #     for entry in JSON_file:
    #         json.dump(entry, outfile)
    #         outfile.write('\n')