'''
    This repository is used to prepare Bridge dataset
'''
import os, sys, shutil




if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help="/path/to/Bridge/raw/bridge_data_v1/berkeley this is for the V1")
    parser.add_argument('--destination_path', type=str, required=True, help="Store path")
    args = parser.parse_args()

    # Variable management
    dataset_path = args.dataset_path    # "/nfs/turbo/coe-jjparkcv/datasets/bridge/raw/bridge_data_v1/berkeley"
    destination_path = args.destination_path    # "/nfs/turbo/jjparkcv-turbo-large/boyangwa/datasets_rob2/Bridge_v1"
    start_idx = 0


    # Make dir if needed
    if os.path.exists(destination_path):
        shutil.rmtree(destination_path)
    os.makedirs(destination_path)


    # Iterate all the folders inside
    for scene_name in sorted(os.listdir(dataset_path)):
        print("We are reading scene", scene_name)
        scene_dir = os.path.join(dataset_path, scene_name)

        for task_name in os.listdir(scene_dir):
            task_dir = os.path.join(scene_dir, task_name)

            for time_clock in os.listdir(task_dir):
                if time_clock == "lmdb":
                    continue    # Skip lmdb folder
                
                time_dir = os.path.join(task_dir, time_clock, "raw", "traj_group0")
                if not os.path.exists(time_dir):
                    continue

                for traj_name in os.listdir(time_dir):
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


                    for img_name in os.listdir(traj_path):
                        if img_name != "images0":       # Only consider one camera angle
                            continue

                        img_folder_path = os.path.join(traj_path, img_name)
                        if not os.path.isdir(img_folder_path):
                            continue
                        
                        ############################################ Main Process ####################################################

                        # Now we can copy the folder to our destination
                        target_dir = os.path.join(destination_path, str(start_idx))
                        shutil.copytree(img_folder_path, target_dir)
                        print("Copy " + str(img_folder_path) + " to " + str(os.path.join(destination_path, str(start_idx))))


                        # Sanity check
                        length = len(os.listdir(target_dir))
                        status = True
                        for check_idx in range(length):
                            if not os.path.exists(os.path.join(target_dir, 'im_' + str(check_idx) + '.jpg' )):  # Should be sequentially exists
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

    print("We have ", start_idx)