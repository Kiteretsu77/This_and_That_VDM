'''
    This repository is used to prepare Bridge dataset with this that conditioning
'''
import os, sys, shutil
import pickle
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import cv2
import math
import collections
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry


def show_mask(mask, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    return mask_image * 255


def read_center_point(model, img_path, do_visualization, store_path):

    action_img = Image.open(img_path)
    prediction = model.predict(source=action_img, save=False)[0]  # Only 1 frame
    
    if not hasattr(prediction, "boxes"):
        print("Detection Fail: We cannot have boxes attribute")
        return None, None   # -1 means NAN and pass this case
    
    # save at the temp_places for visualizaiton
    if do_visualization:
        prediction.save(filename=store_path)


    bounding_boxes = prediction.boxes.xywh
    num, dim = bounding_boxes.shape
    assert(dim == 4)
    
    # Catch up all center point of all bounding boxes
    edge_point_cord = []
    center_points = []
    for idx in range(num):
        x, y, w, h = bounding_boxes[idx].detach().cpu().numpy()
        center_point = [x, y]   # TODO: y+(h/4) 根据经验，往下飘逸25%的高度，一般来说比较有帮助

        edge_point_cord.extend([ (x+w//2, y+h//2), (x-w//2, y+h//2), (x-w//2, y-h//2), (x+w//2, y-h//2) ])


        if w <= 15 or h <= 15:    # If a bounding box is too small, we will disregard this case
            return None, None

        # Calculate the distance between current one and previous points for sanity check
        for point in center_points: # Check all previous points
            give_up_threshold = 90
            if center_point[0] - point[0] >= give_up_threshold:  
                print("Two points are too far away and neglect the case")
                return None, None
            if center_point[1] - point[1] >= give_up_threshold:
                print("Two points are too far away and neglect the case")
                return None, None
        
        # Append to the list
        center_points.append(center_point)

    
    if len(center_points) == 0 or len(center_points) > 2:
        print("Detection Fail: We cannot detect bounding boxes")
        return None, None
    
    # Calculating the average distance among center_points
    if len(center_points) == 2:
        first_box, second_box = center_points
        
        center_x = (first_box[0] + second_box[0]) / 2
        center_y = (first_box[1] + second_box[1]) / 2

        distance = math.sqrt(abs(first_box[0] - second_box[0])**2 + abs(first_box[1] - second_box[1])**2)
        
        return [center_x, center_y, distance], edge_point_cord
        
    return [*center_points[0], 100], edge_point_cord # if len(center_points) == 1, distance is 0; however, to avoid 2-1-2 box detection in sequential, we set it as a higher value



def detect_gripper(gripper_detection_model, input_dir, action_start, action_end, do_visualization, store_dir, sample_failure_collect_folder=None):

    # 先处理第一个point的（这个比较重要，所以要重复3次）；然后再快速处理最后一个point

    # Process the first action frame by iterating next three frames and choose the closest one
    first_center_points = []
    edge_point_cords = []
    for idx in range(3):    # Repeat 3 times
        action_start_path = os.path.join(input_dir, "im_"+str(action_start + idx)+".jpg")
        first_center_point, edge_point_cord = read_center_point(gripper_detection_model, action_start_path, do_visualization, os.path.join(store_dir, "contact_first"+str(idx)+".jpg"))    # The first frame
        
        if idx == 0 and first_center_point is None:
            message = "Cannot find the first contact point!"

            print("The contact point we cannot detect is at ", action_start_path)
            if sample_failure_collect_folder != "":
                shutil.copyfile(action_start_path, os.path.join(sample_failure_collect_folder, str(len(os.listdir(sample_failure_collect_folder)))+".jpg") )

            return (None, None, message)

        if first_center_point is not None:
            first_center_points.append([action_start + idx, first_center_point])

            # Add edge points
            print(edge_point_cord)
            edge_point_cords.extend(edge_point_cord)    # 我有点担心所有point就这么extend会对一些的edge case不是那么robust


    # Select the closest point between two
    first_center_points.sort(key=lambda x: x[1][2])
    first_center_point = first_center_points[0][1][:2]
    start_idx = first_center_points[0][0]
    print("first_center_point is " + str(first_center_point)  + " with idx " + str(start_idx))
    order_idx = [start_idx, action_end]
    

    # Find the xmin, ymin, xmax, ymax for based all three points as the bounding box for the SAM
    edge_point_cords.sort(key=lambda x: x[0])
    xmin = int(edge_point_cords[0][0])
    xmax = int(edge_point_cords[-1][0])

    edge_point_cords.sort(key=lambda x: x[1])
    ymin = int(edge_point_cords[0][1])
    ymax = int(edge_point_cords[-1][1])

    bbox_info = (xmin, xmax, ymin, ymax)


    # Process the last action frame
    action_end_path = os.path.join(input_dir, "im_"+str(action_end)+".jpg")
    last_center_point, edge_point_cord = read_center_point(gripper_detection_model, action_end_path, do_visualization, os.path.join(store_dir, "contact_last.jpg"))  # The last frame
    if last_center_point is None:
        message = "Cannot find the last contact point!"

        print("The contact point we cannot detect is at ", action_start_path)
        if sample_failure_collect_folder != "":
            store_name = str(len(os.listdir(sample_failure_collect_folder))) + ".jpg"
            shutil.copyfile(action_start_path, os.path.join(sample_failure_collect_folder, store_name) )

        return (None, bbox_info, message)
    last_center_point = last_center_point[:2]
    

    # Check if two center points is too close, if they are too close, we will merge to one point
    merge_threshold = 30
    if math.sqrt((first_center_point[0] - last_center_point[0])**2 + (first_center_point[1] - last_center_point[1])**2) <= merge_threshold:
        print("Merge two points to one!")
        message = "Success!"
        return ([[first_center_point], order_idx], bbox_info, message)


    # Return needed information
    message = "Success!"
    return ([[first_center_point, last_center_point], order_idx], bbox_info, message)




def visualize_this_that(base_img, bbox_info, this_that_points):

    # Draw a green dot only for the start point
    for point in this_that_points:
        print("point is ", point)
        target_horizontal, target_vertical = point
        target_horizontal, target_vertical = int(target_horizontal), int(target_vertical)
        
        dot_range = 3 
        for i in range(-1*dot_range, dot_range+1):
            for j in range(-1*dot_range, dot_range+1):
                dil_vertical, dil_horizontal = target_vertical + i, target_horizontal + j
                if (0 <= dil_vertical and dil_vertical < base_img.shape[0]) and (0 <= dil_horizontal and dil_horizontal < base_img.shape[1]):
                    base_img[dil_vertical, dil_horizontal, :] = [0, 128, 0]
                # else:
                #     # print("The traj is out of boundary!!!!!!!!!!!!!!!!!!!!! and we won't consider it")      # 现在
                #     return (False, base_img)
            
    # Draw the bounding box
    xmin, xmax, ymin, ymax = bbox_info
    base_img = cv2.rectangle(base_img, (xmin, ymin), (xmax, ymax), color=(0,0,255), thickness=2)

    return (True, base_img)



def manage_seq_range(input_dir, store_dir, sample_failure_collect_folder, total_frames_needed, 
                        max_original_input_tolerate, gripper_detection_model, sam_predictor, do_visualization):

    # Find valid image lists
    num_frames_input = 0
    for file_name in os.listdir(input_dir):
        if file_name.startswith("im_"):
            num_frames_input += 1
    for idx in range(num_frames_input):
        target_path = os.path.join(input_dir, "im_"+str(idx)+".jpg")
        if not os.path.exists(target_path):
            print("We don't have ", target_path)
            message = "Invalid error"   # Make sure that every file in this order is existed, this is quite important
            return (False, message)
    

    if num_frames_input > max_original_input_tolerate:
        message = "The number of frames is too long for constructing the sequence length needed"
        return (False, message)
    
    if num_frames_input < total_frames_needed:
        message = "The number of frames is too short for constructing the sequence length needed"
        return (False, message)
    


    # Prepare this and that based on policy_out.pkl
    policy_out_file_path = os.path.join(input_dir, "policy_out.pkl")
    with open(policy_out_file_path, "rb") as f:
        policy = pickle.load(f)

    actions_codes = []
    action_start, action_end = None, None
    for idx, item in enumerate(policy):
        action_value = item["actions"][-1]
        if action_start is None and action_value == 0.0:
            action_start = idx

        if (action_start is not None) and (action_end is None) and (action_value == 1.0): 
            action_end = idx    # Until record the first 1.0 exists after the first 0.0 appears
        actions_codes.append(action_value)
        
    if action_start is None or action_end is None:  
        message = "We cannot read an action_start or action_end code!"
        return (False, message)    # Requires to have both start and end actions (Usually, they are a pair)
    
    print("actions_codes is ", actions_codes)
    print("the start end idx we read is ", action_start, action_end)


    # Detect the gripper (should return a list with exactly two x,y coordinate points)
    detection_retrun_info, bbox_info, detect_message = detect_gripper(
                                                                        gripper_detection_model, 
                                                                        input_dir,
                                                                        action_start,
                                                                        action_end,
                                                                        do_visualization = do_visualization,
                                                                        store_dir = store_dir,
                                                                        sample_failure_collect_folder = sample_failure_collect_folder, 
                                                                    )
    if detection_retrun_info is None:
        return (False, detect_message)
    
    detected_point, old_seq_idx = detection_retrun_info
    print("detected_point is ", detected_point)


    # Visualize if needed    
    base_img = cv2.imread(os.path.join(input_dir, "im_0.jpg"))
    if do_visualization:
        status, visual_img = visualize_this_that(base_img, bbox_info, detected_point)
        if status:
            cv2.imwrite(os.path.join(store_dir, "visualization.png"), visual_img)



    # SAM process based on bbox_info
    xmin, xmax, ymin, ymax = bbox_info
    sam_predictor.set_image(np.uint8(base_img))
    positive_point_cords = np.array([[ int(detected_point[0][0]), int(detected_point[0][1]) ]])
    positive_point_cords = np.array(positive_point_cords)
    positive_point_labels = np.ones(len(positive_point_cords))

    # Predict the mask based on the point and bounding box designed
    masks, scores, logits = sam_predictor.predict(  
                                                    point_coords = positive_point_cords,
                                                    point_labels = positive_point_labels,
                                                    box = np.array([xmin, ymin, xmax, ymax])[None, :],
                                                    multimask_output = False,
                                                )
    print(scores)
    for mask_idx, mask in enumerate(masks):
        mask_img = show_mask(mask)
        cv2.imwrite(os.path.join(store_dir, "mask_" + str(mask_idx) + ".png"), mask_img)



    ################################ Move the img ######################################
    # Calculate needed parameters
    division_factor = num_frames_input // total_frames_needed
    remain_frames = (num_frames_input % total_frames_needed) - 1    # -1 for adaptation

    # Define the gap
    gaps = [division_factor for _ in range(total_frames_needed-1)]
    for idx in range(remain_frames):
        if idx % 2 == 0:
            gaps[idx//2] += 1      # Start to end order
        else:
            gaps[-1*(1+(idx//2))] += 1   # End to start order

    # Map the gap to the specific orders
    idx_orders = [1]    # 从1还是shift一下问题应该不大
    for global_idx, gap in enumerate(gaps):
        idx_orders.append(idx_orders[-1] + gap)
    if idx_orders[-1] >= num_frames_input:
        message = "Invalid error"
        return (False, message)
    # assert(idx_orders[-1] < num_frames_input)
    assert(len(idx_orders) == total_frames_needed)
    

    # Copy the essential files first
    for global_idx, cur_idx in enumerate(idx_orders):
        source_path = os.path.join(input_dir, "im_"+str(cur_idx)+".jpg")
        destination_path = os.path.join(store_dir, "im_"+str(global_idx)+".jpg")

        if not os.path.exists(source_path):     # Theoretically, source_path must exists
            message = "We couldn't find the source path. Theoretically, source_path must exists!"  # 有一种可能就是我们丢失了一些地方，在cp或者本来就没有，记得统计数量
            return (False, message)

        shutil.copyfile(source_path, destination_path)

    # Map order_idx to the cropped version
    mapped_seq_idx = []
    for old_idx in old_seq_idx:
        tmp = []
        for tmp_idx, new_idx in enumerate(range(len(idx_orders))):
            tmp.append((tmp_idx, abs(old_idx - idx_orders[new_idx])))
        # Sort the smallest fistance
        tmp.sort(key=lambda x: x[1])
        mapped_seq_idx.append(tmp[0][0])

    print("Before the idx is ", old_seq_idx)
    print("mapped idx is ", mapped_seq_idx)


    # Write the information to new destination
    f = open(os.path.join(store_dir, "data.txt"), "a")
    f.write(str(mapped_seq_idx[0]) + " " + str(detected_point[0][0]) + " " + str(detected_point[0][1]) + "\n")
    if len(detected_point) == 2:   # Two points excluding the last idx 
        f.write(str(mapped_seq_idx[1]) + " " + str(detected_point[1][0]) + " " + str(detected_point[1][1]) + "\n")
    f.close()


    # Move lang.txt file
    shutil.copyfile(os.path.join(input_dir, 'lang.txt'), os.path.join(store_dir, 'lang.txt'))


    message = "Success!"
    return (True, message)




if __name__ == "__main__":

    # General storage setting
    dataset_path = "../datasets_rob/Bridge_v2_raw"
    destination_path = "../sanity_check/bridge_v2_TT14_longer_tolerance"
    sample_failure_collect_folder = ""      # This is to collect cases that fail for active learning

    total_frames_needed = 14
    max_original_input_tolerate = 56        # 40 for 14 fps; 60 for 25fps; 
    do_visualization = True


    # YOLO model init
    yolo_pretarined_path = "pretrained/yolov8n_best.pt"
    gripper_detection_model = YOLO("yolov8n.yaml")  # build a new model from scratch
    gripper_detection_model = YOLO(yolo_pretarined_path)  # load a pretrained model (recommended for training)

    # SAM model init
    model_type = "vit_h"
    sam_pretrained_path = "pretrained/sam_vit_h_4b8939.pth"
    sam = sam_model_registry[model_type](checkpoint=sam_pretrained_path).to(device="cuda")
    sam_predictor = SamPredictor(sam)     # There is a lot of setting here


    # Make dir if needed
    if os.path.exists(destination_path):
        shutil.rmtree(destination_path)    
    os.makedirs(destination_path)

    # Prepare the folder to collect failure cases
    if sample_failure_collect_folder != "":
        if os.path.exists(sample_failure_collect_folder):
            shutil.rmtree(sample_failure_collect_folder) 
        os.makedirs(sample_failure_collect_folder)



    # Collect the message
    message_dict = collections.defaultdict(int)


    store_idx = 0
    for folder_name in sorted(os.listdir(dataset_path)):
        input_folder_path = os.path.join(dataset_path, folder_name)
        store_folder_path = os.path.join(destination_path, "0"*(6-len(str(store_idx)))+str(store_idx))
        print("We are processing ", input_folder_path)

        # Prepare store_folder_path folder
        os.makedirs(store_folder_path)
        
        status, message = manage_seq_range(input_folder_path, store_folder_path, sample_failure_collect_folder, total_frames_needed, max_original_input_tolerate, gripper_detection_model, sam_predictor, do_visualization)
        if status:      # We will only update the store_idx only when this file is successfully written
            store_idx += 1
        else:
            print("This status failed! Message: " + message)
            shutil.rmtree(store_folder_path)
        # break # For debug
            
        # Collect the infor to dict
        message_dict[message] += 1
    
    print("We have " + str(store_idx) + " valid dataset")
    print("message_dict info is ", message_dict)

