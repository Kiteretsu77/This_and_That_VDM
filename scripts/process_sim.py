'''
    This is a script to processs Mark's data.
'''
import os, sys, shutil

if __name__ == "__main__":
    file_path = "/nfs/turbo/coe-jjparkcv/datasets/isaac-gym-pick-place/full/dataset_v3_proc"
    store_path = "../datasets_rob/sim_raw"
    most_descriptive_prompt_idx = 6     # Start from the 0


    # Folder management
    if os.path.exists(store_path):
        shutil.rmtree(store_path)
    os.makedirs(store_path)

    # Check length
    file_names = os.listdir(file_path)
    target_length = len(file_names) // 10   # 10 files as a cycle

    
    for idx in range(target_length):
        sub_folder_path = os.path.join(file_path, "run_"+str(10*idx))
        if not os.path.exists(sub_folder_path):
            continue

        # Prepare the target position
        sub_store_path = os.path.join(store_path, str(idx))
        os.makedirs(sub_store_path)
        
        # Find the key prompt to read it
        prompt_content = []
        for tmp_idx in range(10):
            tmp_text_path = os.path.join(file_path, "run_"+str(10*idx + tmp_idx), "lang.txt")    # Usually, the 6th is the most concrete version
            if not os.path.exists(tmp_text_path):
                continue
            file = open(tmp_text_path, 'r')
            prompt_content.append(file.readlines()[0])
            file.close()
        print("prompt_content we have num ", len(prompt_content))



        # Copy the image into the target position and copy the data.txt
        for file_name in os.listdir(sub_folder_path):
            if file_name == "lang.txt":
                continue
            shutil.copyfile(os.path.join(sub_folder_path, file_name), os.path.join(sub_store_path, file_name))

        # Handle the lang.txt
        target_lang_txt_path = os.path.join(sub_store_path, "lang.txt")
        f = open(target_lang_txt_path, "a")
        f.write(prompt_content[most_descriptive_prompt_idx]+"\n")
        for tmp_idx in range(10):
            if tmp_idx == most_descriptive_prompt_idx:
                continue
            f.write(prompt_content[tmp_idx]+"\n")
        f.close()

