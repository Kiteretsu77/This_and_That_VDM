'''
    This file is to add point to the first image
'''

import os, shutil, sys

if __name__ == "__main__":
    input_folder_path = "/nfs/turbo/jjparkcv-turbo-large/boyangwa/model_results/Human_Study/Input_Bridge_human_evaluation"
    store_path = "point_highlighted"

    if os.path.exists(input_folder_path):
        shutil.rmtree(input_folder_path)
    os.makedirs(input_folder_path)


    for instance_name in os.listdir(input_folder_path):

        sub_folder_dir = os.path.join(input_folder_path, instance_name)

        # Read file
        file_path = os.path.join(sub_folder_dir, "data.txt")
        file1 = open(file_path, 'r')
        Lines = file1.readlines()

        # Read the first img
        first_img_path = os.path.join(sub_folder_dir, "im_0.jpg")


        # Init the image
        base_img = cv2.imread(first_img_path).astype(np.float32)      # Use the original image size

        # Draw the point
        for idx in range(len(Lines)):
            # Read points
            frame_idx, horizontal, vertical = Lines[idx].split(' ')
            frame_idx, vertical, horizontal = int(frame_idx), int(float(vertical)), int(float(horizontal))

            # Draw square around the target position
            dot_range = 15       # Diameter
            for i in range(-1*dot_range, dot_range+1):
                for j in range(-1*dot_range, dot_range+1):
                    dil_vertical, dil_horizontal = vertical + i, horizontal + j
                    if (0 <= dil_vertical and dil_vertical < base_img.shape[0]) and (0 <= dil_horizontal and dil_horizontal < base_img.shape[1]):
                        if idx == 0:
                            base_img[dil_vertical][dil_horizontal] = [0, 0, 255]        # The first point should be red
                        else:
                            base_img[dil_vertical][dil_horizontal] = [0, 255, 0]        # The second point should be green to distinguish the first point
        
        


