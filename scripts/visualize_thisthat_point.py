'''
    This repo is provided to change the destination area.
'''

import os, cv2


def draw_dot(ref_img, new_h, new_w):
    # Draw the dot
    dot_range = 3 
    for i in range(-1*dot_range, dot_range+1):
        for j in range(-1*dot_range, dot_range+1):
            dil_vertical, dil_horizontal = new_h + i, new_w + j
            if (0 <= dil_vertical and dil_vertical < ref_img.shape[0]) and (0 <= dil_horizontal and dil_horizontal < ref_img.shape[1]):
                ref_img[dil_vertical, dil_horizontal, :] = [0, 128, 0]

    return ref_img


if __name__ == "__main__":
    instance_path = "datasets/validation_thisthat14/000049/"
    new_w, new_h = 385, 310
    # 256.1850280761719 241.71287155151367

    # Read the items
    data_path = os.path.join(instance_path, "data.txt")
    ref_img_path = os.path.join(instance_path, "im_0.jpg")
    ref_img = cv2.imread(ref_img_path)


    # Read the first point
    file1 = open(data_path, 'r')
    Lines = file1.readlines()
    frame_idx, horizontal, vertical = Lines[0].split(' ')
    ref_img = draw_dot(ref_img, int(float(vertical)), int(float(horizontal)))

    # Second dot
    ref_img = draw_dot(ref_img, new_h, new_w)


    
    # Store the image
    cv2.imwrite("visual.png", ref_img)