'''
    Process the llama file for the next step
'''
import os, shutil, sys
import json
import pandas as pd  
import collections


if __name__ == "__main__":

    # Define important path
    json_path = "../SVD1/v1.jsonl"
    folder_path = "/home/kiteret/Desktop/StableVideoDiffusion/full_text_tmp/"
    

    # Read the json file
    with open(json_path, 'r') as json_file:
        json_list = list(json_file)

    # Iterate all the json files
    length_stats = collections.defaultdict(int)
    for json_info in json_list:
        json_info = json.loads(json_info)


        # Define the path to write
        key_start = len("/home/chfeng/llama3/full_text_tmp/")
        key_end = len("lang.txt")
        sub_path = json_info["file_path"][key_start:int(-1*key_end)]
        new_text_path = os.path.join(folder_path, sub_path, "processed_text.txt")
        if os.path.exists(new_text_path):
            os.remove(new_text_path)


        # Sanity check for the case where input is missed
        if json_info["input"] == "":
            print("It is weird for the input is empty in the LLM process for ", sub_path)
            continue


        # Re-Define the content
        outputs = json_info["output"]
        if outputs.find("action:") != 0:
            print("It is weird for no actions: keyword in the outputs for ", sub_path, " with prompt ", outputs)
            continue
        
        # Prepare write file
        contents = outputs.split('\n')
        f = open(new_text_path, "a")

        # Itearte
        effective_length = 0
        for idx, content in enumerate(contents):
            key_word = content.split(":")[1][1:]
            if key_word != "":
                effective_length += 1
            else:
                if idx == 1:
                    print("It is abnormal for the this content to be empty ", sub_path, " with prompt ", outputs)
            f.write(key_word + "\n")
        # if effective_length == 2:
        #     print("short prompt case is ", sub_path, " with prompt ", outputs)
        if effective_length < 2:  # For those only 1 or zero, we won't consider them
            print("The prompt is too short for ", sub_path, " with prompt ", outputs)
            os.remove(new_text_path)

        length_stats[effective_length] += 1
    
    print("length_stats is ", length_stats)




