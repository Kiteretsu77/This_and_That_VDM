'''
    This repository is used to prepare Bridge dataset
'''
import os, sys, shutil



def read_bridge_v2(dataset_path, train_store_path, test_store_path, test_dataset_lists, copyfile=True):
    # copyfile is True most of the time

    start_idx = 0
    target_lists = []
    prefix_len = len(dataset_path) + 1

    # Iterate all the folders inside
    for scene_name in sorted(os.listdir(dataset_path)):
        print("We are reading scene ", scene_name)
        scene_dir = os.path.join(dataset_path, scene_name)

        for task_name in sorted(os.listdir(scene_dir)):
            task_dir = os.path.join(scene_dir, task_name)

            for order_name in sorted(os.listdir(task_dir)):
                order_dir = os.path.join(task_dir, order_name)

                for time_clock in sorted(os.listdir(order_dir)):
                    if time_clock == "lmdb":
                        continue    # Skip lmdb folder
                    
                    time_dir = os.path.join(order_dir, time_clock, "raw", "traj_group0")
                    if not os.path.exists(time_dir):
                        print("time_dir does not exist for ", time_dir)
                        continue

                    for traj_name in sorted(os.listdir(time_dir)):
                        traj_path = os.path.join(time_dir, traj_name)
                        if not os.path.isdir(traj_path):
                            print("traj_path does not exist for ", traj_path)
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
                                print("img_folder_path does not exist for ", img_folder_path)
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
                                
                                # Now we can copy the folder to our destination
                                print("Copy " + str(img_folder_path) + " to " + str(os.path.join(train_store_path, str(start_idx))))
                                shutil.copytree(img_folder_path, target_dir)
                                
                                # Sanity check
                                length = len(os.listdir(target_dir))
                                status = True
                                for check_idx in range(length):
                                    if not os.path.exists(os.path.join(target_dir, 'im_' + str(check_idx) + '.jpg' )):    # Should be sequentially exists
                                        status = False
                                        break
                                
                                if not status:
                                    # If they didn't have sequential files we need, we will remove and begin again without updating start_idx
                                    print("This file cannot pass the sanity check. We will remove it!")
                                    shutil.rmtree(target_dir)
                                    continue
                                
                                # Move other auxilary files
                                shutil.copy(policy_out_file_path, os.path.join(target_dir, "policy_out.pkl"))
                                shutil.copy(lang_txt_file_path, os.path.join(target_dir, "lang.txt"))

                            # Update the idx
                            start_idx += 1

    print("We have ", start_idx)
    
    # Return a list of file path
    return target_lists



if __name__ == "__main__":
    dataset_path = "/nfs/turbo/jjparkcv-turbo-large/boyangwa/raw/bridge_data_v2"
    train_store_path = "../sanity_check/bridge_v2_raw"
    test_store_path = "../sanity_check/bridge_v2_test_raw"
    test_dataset_predefined_path = "test_path_v2.txt"


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


    read_bridge_v2(dataset_path, train_store_path, test_store_path, test_dataset_lists)

    