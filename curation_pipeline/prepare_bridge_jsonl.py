'''
    This file is to prepare the dataset in jsonl file
'''

import os, sys, shutil
import json

# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from curation_pipeline.prepare_bridge_v1 import read_bridge_v1
from curation_pipeline.prepare_bridge_v2 import read_bridge_v2


if __name__ == "__main__":
    v1_dataset_path = "/nfs/turbo/jjparkcv-turbo-large/boyangwa/raw/bridge_data_v1/berkeley"
    v2_dataset_path = "/nfs/turbo/jjparkcv-turbo-large/boyangwa/raw/bridge_data_v2"
    store_name = "store.jsonl"

    if os.path.exists(store_name):
        os.remove(store_name)
    

    # Execute
    full_lists = []

    v1_lists = read_bridge_v1(v1_dataset_path, "", copyfile=False)
    full_lists.extend(v1_lists)
    v2_lists = read_bridge_v2(v2_dataset_path, "", copyfile=False)
    full_lists.extend(v2_lists)
    print("Full length is ", len(full_lists))


    with open(store_name, 'w') as outfile:
        for list_name in full_lists:
            instance = dict()
            instance["file_path"] = list_name

            json.dump(instance, outfile)
            outfile.write('\n')



    # with open('output.jsonl', 'w') as outfile:
    #     for entry in JSON_file:
    #         json.dump(entry, outfile)
    #         outfile.write('\n')