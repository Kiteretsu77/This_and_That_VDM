'''
    This file is trying to repeat the frames such the it reaches target frames needed
'''
import os, shutil, sys

if __name__ == "__main__":
    input_path = "/nfs/turbo/coe-jjparkcv/boyangwa/AVDC/AVDC_results"
    store_path = "/nfs/turbo/coe-jjparkcv/boyangwa/AVDC/AVDC_results_interpolated"
    total_frames_needed = 14

    # Handle the file folder management
    if os.path.exists(store_path):
        shutil.rmtree(store_path)
    os.makedirs(store_path)

    for video_name in sorted(os.listdir(input_path)):
        sub_input_path = os.path.join(input_path, video_name)
        sub_store_path = os.path.join(store_path, video_name)

        # Create the store place
        os.makedirs(sub_store_path)

        # Find valid image lists
        num_frames_input = 0
        for file_name in os.listdir(sub_input_path):
            if file_name.endswith("png"):
                num_frames_input += 1
        print("num_frames_input is ", num_frames_input)
        
        # Calculate needed parameters
        division_factor = total_frames_needed // num_frames_input 
        remain_frames = (total_frames_needed % num_frames_input) - 1    # -1 for adaptation

        # Define the gap
        gaps = [division_factor for _ in range(num_frames_input)]
        for idx in range(remain_frames):
            if idx % 2 == 0:
                gaps[idx//2] += 1      # Start to end order
            else:
                gaps[-1*(1+(idx//2))] += 1   # End to start order

        print("gaps is ", gaps)


        # Write to the new folder
        store_idx = 0
        for frame_idx, gap in enumerate(gaps):
            for tmp in range(gap): # Repeat copy gap num of times
                img_path = os.path.join(sub_input_path, str(frame_idx)+".png")
                shutil.copyfile(img_path, os.path.join(sub_store_path, str(store_idx)+".png"))
                store_idx += 1
   
            


