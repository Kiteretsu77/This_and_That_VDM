'''
    This repository is used to prepare Bridge dataset
'''
import os, sys, shutil




if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help="/path/to/Bridge/raw/bridge_data_v2 this is for the V2")
    parser.add_argument('--destination_path', type=str, required=True, help="Store path")
    args = parser.parse_args()

    # Variable management
    dataset_path = args.dataset_path            # like "DISK/datasets/bridge/raw/bridge_data_v2"
    destination_path = args.destination_path    # like "../datasets_rob/Bridge_v2"
    start_idx = 0



    # Make dir if needed
    if not os.path.exists(destination_parent_path):
        os.mkdir(destination_parent_path)


    # Iterate all the folders inside
    for scene_name in sorted(os.listdir(dataset_path)):
        print("We are reading scene", scene_name)
        scene_dir = os.path.join(dataset_path, scene_name)

        for task_name in os.listdir(scene_dir):
            task_dir = os.path.join(scene_dir, task_name)

            for order_name in os.listdir(task_dir):
                order_dir = os.path.join(task_dir, order_name)

                for time_clock in os.listdir(order_dir):
                    if time_clock == "lmdb":
                        continue    # Skip lmdb folder
                    
                    time_dir = os.path.join(order_dir, time_clock, "raw", "traj_group0")
                    if not os.path.exists(time_dir):
                        print("time_dir is not exists for ", time_dir)
                        continue

                    for traj_name in os.listdir(time_dir):
                        traj_path = os.path.join(time_dir, traj_name)
                        if not os.path.isdir(traj_path):
                            print("traj_path is not exists for ", traj_path)
                            continue
                        
                        # Directly move policy_out_file_path; just in case there is also valuable information there
                        policy_out_file_path = os.path.join(traj_path, "policy_out.pkl")
                        if not os.path.exists(policy_out_file_path):
                            continue

                        # Check the lang txt file
                        lang_txt_file_path = os.path.join(traj_path, "lang.txt")
                        if not os.path.exists(lang_txt_file_path):
                            continue


                        for img_name in os.listdir(traj_path):
                            if img_name != "images0":       # Only consider one camera angle
                                continue

                            img_folder_path = os.path.join(traj_path, img_name)
                            if not os.path.isdir(img_folder_path):
                                print("img_folder_path does not exist for ", img_folder_path)
                                continue
                            
                            destination_path = os.path.join(destination_parent_path, str(start_idx))
                            if os.path.exists(destination_path):
                                shutil.rmtree(destination_path)

                            # Now we can copy the folder to our destination
                            shutil.copytree(img_folder_path, destination_path)
                            print("Copy " + str(img_folder_path) + " to " + str(os.path.join(destination_path, str(start_idx))))


                            # Sanity check
                            # length = len(os.listdir(destination_path))
                            # status = True
                            # for check_idx in range(length):
                            #     if not os.path.exists(os.path.join(destination_path, 'im_' + str(check_idx) + '.jpg' )):    # Should be sequentially exists
                            #         status = False
                            #         break
                            
                            # if not status:
                            #     # If they didn't have sequential files we need, we will remove and begin again without updating start_idx
                            #     print("This file cannot pass the sanity check. We will remove it!")
                            #     shutil.rmtree(destination_path)
                            #     continue
                            
                            # Move other auxilary files
                            shutil.copy(policy_out_file_path, os.path.join(destination_path, "policy_out.pkl"))
                            shutil.copy(lang_txt_file_path, os.path.join(destination_path, "lang.txt"))

                            start_idx += 1

    print("We have ", start_idx)