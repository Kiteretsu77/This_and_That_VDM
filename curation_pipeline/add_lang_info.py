'''
    Add the processed lang information
'''
import os, sys, shutil
import json


if __name__ == "__main__":

    # Main config file path information
    processed_json_file_path = "updated_bridge_v2.json"


    # Read the json file
    file = open(processed_json_file_path)
    data = json.load(file)


    # Iterate all the folders inside
    start_idx = 0
    for seq_instance in data:
        target_path = seq_instance["images0"]
        print("We are processing ", target_path)

        processed_lang_txt_path = os.path.join(target_path, "processed_lang.txt")
        if os.path.exists(processed_lang_txt_path):
            os.remove(processed_lang_txt_path)
        
        # Write the action + This + That into the sequence.
        processed_lang_txt = open(processed_lang_txt_path, "a")
        processed_lang_txt.write(str(seq_instance["action"])+"\n")
        processed_lang_txt.write(str(seq_instance["this"])+"\n")
        processed_lang_txt.write(str(seq_instance["that"])+"\n")


        start_idx += 1

    print("We have ", start_idx)