'''
    This repository is used to prepare Bridge dataset
'''
import os, sys, shutil


def read_bridge_v1(dataset_path, train_store_path, test_store_path, test_dataset_lists, copyfile=True):
    # copyfile is True when we need to copy the file to the target destination

    start_idx = 0
    target_lists = []
    prefix_len = len(dataset_path) + 1

    # Iterate all the folders inside
    for scene_name in sorted(os.listdir(dataset_path)):
        print("We are reading scene ", scene_name)
        scene_dir = os.path.join(dataset_path, scene_name)
        for task_name in sorted(os.listdir(scene_dir)):
            task_dir = os.path.join(scene_dir, task_name)

            for time_clock in sorted(os.listdir(task_dir)):
                if time_clock == "lmdb":
                    continue    # Skip lmdb folder
                
                time_dir = os.path.join(task_dir, time_clock, "raw", "traj_group0")
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
                        
                        ############################################ Main Process ####################################################

                        # # First Sanity check (Make sure the input source is jpg good)
                        # length = len(os.listdir(img_folder_path))
                        # status = True
                        # for check_idx in range(length):
                        #     if not os.path.exists(os.path.join(img_folder_path, 'im_' + str(check_idx) + '.jpg')):  # Should be sequentially exists
                        #         status = False
                        #         break

                        # Now we can copy the folder to our destination
                        target_lists.append(img_folder_path)
                        if copyfile:
                            print("img_folder_path[prefix_len:] is ", img_folder_path[prefix_len:])
                            if img_folder_path[prefix_len:] in test_dataset_lists:
                                # Store to test set
                                target_dir = os.path.join(test_store_path, str(start_idx))
                            else:
                                # This is training set
                                target_dir = os.path.join(train_store_path, str(start_idx))
                            
                            print("Copy " + str(img_folder_path) + " to " + str(target_dir))
                            shutil.copytree(img_folder_path, target_dir)


                            # Sanity check
                            length = len(os.listdir(target_dir))
                            status = True
                            for check_idx in range(length):
                                if not os.path.exists(os.path.join(target_dir, 'im_' + str(check_idx) + '.jpg')):  # Should be sequentially exists
                                    status = False
                                    break

                            if not status:
                                # If they didn't have sequential files we need, we will remove and begin again without updating start_idx
                                print("This file cannot pass the sanity check. We will remove it!")
                                shutil.rmtree(target_dir)
                                continue
                            
                            # Move other auxiliary files
                            shutil.copy(policy_out_file_path, os.path.join(target_dir, "policy_out.pkl"))
                            shutil.copy(lang_txt_file_path, os.path.join(target_dir, "lang.txt"))

                        ################################################################################################################

                        # Update the idx
                        start_idx += 1

    print("We have ", start_idx, " number of cases")

    # Return a list of file path
    return target_lists



if __name__ == "__main__":
    dataset_path = "/Path/to/Bridge/raw/bridge_data_v1/berkeley"   # Until Bridge v1 - berkeley section
    train_store_path = "/Path/to/Bridge/train/bridge_v1_raw"
    test_store_path = "/Path/to/Bridge/train/bridge_v1_test_raw"
    test_dataset_predefined_path = "test_path.txt"      # This will be providede by us


    # Make dir if needed
    if os.path.exists(train_store_path):
        shutil.rmtree(train_store_path)
    os.makedirs(train_store_path)
    if os.path.exists(test_store_path):
        shutil.rmtree(test_store_path)
    os.makedirs(test_store_path)


    # Read Test dataset path
    test_dataset_lists = []
    read_file = open(test_dataset_predefined_path, "r")
    for line in read_file.readlines():
        test_dataset_lists.append(line[:-1])
    print("test_dataset_lists is ", test_dataset_lists)


    read_bridge_v1(dataset_path, train_store_path, test_store_path, test_dataset_lists)
