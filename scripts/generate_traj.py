import sys
import argparse
import copy
import os, shutil
import imageio
import cv2
from PIL import Image, ImageDraw
import os.path as osp
import random
import numpy as np
import torch.multiprocessing as mp
from multiprocessing import set_start_method
import math, time, gc
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry


# Import files from the local path
root_path = os.path.abspath('.')
sys.path.append(root_path)
from config.flowformer_config import get_cfg
from flowformer_code.utils import flow_viz, frame_utils
from flowformer_code.utils.utils import InputPadder
from flowformer_code.FlowFormer import build_flowformer




TRAIN_SIZE = [432, 960]

def show_anns(anns):
    if len(anns) == 0:
        return
    
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(True)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    
    return img*255


def show_mask(mask, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    return mask_image * 255


def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
  if min_overlap >= TRAIN_SIZE[0] or min_overlap >= TRAIN_SIZE[1]:
    raise ValueError(
        f"Overlap should be less than size of patch (got {min_overlap}"
        f"for patch size {patch_size}).")
  if image_shape[0] == TRAIN_SIZE[0]:
    hs = list(range(0, image_shape[0], TRAIN_SIZE[0]))
  else:
    hs = list(range(0, image_shape[0], TRAIN_SIZE[0] - min_overlap))
  if image_shape[1] == TRAIN_SIZE[1]:
    ws = list(range(0, image_shape[1], TRAIN_SIZE[1]))
  else:
    ws = list(range(0, image_shape[1], TRAIN_SIZE[1] - min_overlap))

  # Make sure the final patch is flush with the image boundary
  hs[-1] = image_shape[0] - patch_size[0]
  ws[-1] = image_shape[1] - patch_size[1]
  return [(h, w) for h in hs for w in ws]



def compute_flow(model, image1, image2, weights=None):
    print(f"computing flow...")

    image_size = image1.shape[1:]

    image1, image2 = image1[None].cuda(), image2[None].cuda()

    hws = compute_grid_indices(image_size)
    if weights is None:     # no tile
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_pre, _ = model(image1, image2)

        flow_pre = padder.unpad(flow_pre)
        flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()
    else:                   # tile
        flows = 0
        flow_count = 0

        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]    
            flow_pre, _ = model(image1_tile, image2_tile)
            padding = (w, image_size[1]-w-TRAIN_SIZE[1], h, image_size[0]-h-TRAIN_SIZE[0], 0, 0)
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

        flow_pre = flows / flow_count
        flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()

    return flow


def compute_adaptive_image_size(image_size):
    target_size = TRAIN_SIZE
    scale0 = target_size[0] / image_size[0]
    scale1 = target_size[1] / image_size[1] 

    if scale0 > scale1:
        scale = scale0
    else:
        scale = scale1

    image_size = (int(image_size[1] * scale), int(image_size[0] * scale))

    return image_size


def prepare_image(viz_root_dir, fn1, fn2, keep_size):
    print(f"preparing image...")

    image1 = frame_utils.read_gen(fn1)
    image2 = frame_utils.read_gen(fn2)
    image1 = np.array(image1).astype(np.uint8)[..., :3]
    image2 = np.array(image2).astype(np.uint8)[..., :3]
    if not keep_size:
        dsize = compute_adaptive_image_size(image1.shape[0:2])
        image1 = cv2.resize(image1, dsize=dsize, interpolation=cv2.INTER_CUBIC)
        image2 = cv2.resize(image2, dsize=dsize, interpolation=cv2.INTER_CUBIC)
    image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float()


    dirname = osp.dirname(fn1)
    filename = osp.splitext(osp.basename(fn1))[0]

    viz_dir = osp.join(viz_root_dir, dirname)
    # if not osp.exists(viz_dir):
    #     os.makedirs(viz_dir)

    viz_fn = osp.join(viz_dir, filename + '.png')

    return image1, image2, viz_fn


def build_model():
    print(f"building  model...")
    cfg = get_cfg()
    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model))

    model.cuda()
    model.eval()

    return model


def filter_uv(flow, threshold_factor = 0.2):
    u = flow[:,:,0]
    v = flow[:,:,1]

    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)

    threshold = threshold_factor * rad_max
    flow[:,:,0][rad < threshold] = 0
    flow[:,:,1][rad < threshold] = 0

    return flow


def visualize_traj(base_img, traj_path, connect_points = True):
    target_vertical, target_horizontal = traj_path[-1]

    if connect_points and len(traj_path) > 1:
        # Draw a line to connect two point to show motion direction
        start_coordinate = (traj_path[-2][1], traj_path[-2][0])
        end_coordinate = (traj_path[-1][1], traj_path[-1][0])
        pil_img = Image.fromarray(base_img)

        # Draw the line
        color = 'red'
        draw = ImageDraw.Draw(pil_img)
        draw.line([start_coordinate, end_coordinate], fill = color, width = 3)

        base_img = np.array(pil_img)


    # Draw a green dot only for the start point
    if len(traj_path) == 1:
        dot_range = 3 
        for i in range(-1*dot_range, dot_range+1):
            for j in range(-1*dot_range, dot_range+1):
                dil_vertical, dil_horizontal = target_vertical + i, target_horizontal + j
                if (0 <= dil_vertical and dil_vertical < base_img.shape[0]) and (0 <= dil_horizontal and dil_horizontal < base_img.shape[1]):
                    base_img[dil_vertical][dil_horizontal] = [0, 128, 0]
                else:
                    print("The traj is out of boundary!!!!!!!!!!!!!!!!!!!!! and we won't consider it")      # 现在
                    return (False, base_img)
            
    return (True, base_img)



def calculate_flow(viz_root_dir, store_dir, img_pairs, optical_flow_model, sam_predictor, SAM_positive_sample_num, SAM_negative_sample_num, mask_generator, traj_visualization, keep_size, verbose=False):

    # Trajectory prepare
    traj_path = []              # It collects all points traversed in a temporal order
    is_hard_to_track = False    # If this is True, it means that, we have a time in tracking hard to find dx and dy movement. Under this circumstance, we are not very recommended to use it
    hard_track_idxs = set()
    traj_image_lists = []


    # Iterate all image pairs
    for idx, img_pair in enumerate(img_pairs):

        fn1, fn2 = img_pair
        print(f"processing {fn1}, {fn2}...")

        image1, image2, viz_fn = prepare_image(viz_root_dir, fn1, fn2, keep_size)     # Be very careful, image1 and image2 may be different resolution shape if keep_size is False
        # Generate the optical flow and filter those that is small motion
        flow_uv = filter_uv(compute_flow(optical_flow_model, image1, image2, None))

        # if verbose:
            # Store the visualization of flow_uv
            # flow_img = flow_viz.flow_to_image(flow_uv)
            # cv2.imwrite("optical_flow_" + str(idx+1) + ".png", flow_img[:, :, [2,1,0]])

        if idx == 0:
            # We will store the first image to memory for further visualization purpose

            # Base img
            # base_img = np.uint8(np.transpose(image1.numpy(), (1,2,0)))

            # SAM figure
            # sam_all = mask_generator.generate(image1)
            # base_img = show_anns(sam_all)
            # base_img = np.transpose(base_img, (1,2,0))

            # Plain white image
            base_img = np.zeros(np.transpose(image1.numpy(), (1,2,0)).shape, dtype=np.uint8)
            base_img.fill(255) 




        # Extract moving points (positive point)
        positive_point_cords = []
        nonzeros = np.nonzero(flow_uv)          # [(vertical), (horizontal)]
        if len(nonzeros[0]) < SAM_positive_sample_num:
            # We require the number of points to be more than SAM_positive_sample_num
            return False
        positive_orders = np.random.choice(len(nonzeros[0]), SAM_positive_sample_num, replace=False)    # we have randomly select instead of use all in the sam_predictor prediction
        for i in range(len(nonzeros[0])):    
            if i in positive_orders:  
                positive_point_cords.append([nonzeros[1][i], nonzeros[0][i]])       # 根据document来看，这个就应该是先horizontal再vertical，也就是这个顺序
        positive_point_cords = np.array(positive_point_cords)
        positive_point_labels = np.ones(len(positive_point_cords))


        # Define negative sample (outside the optical flow choice)
        if SAM_negative_sample_num != 0:
            skip_prob = 2 * SAM_negative_sample_num / (flow_uv.shape[0]*flow_uv.shape[1] - len(nonzeros[0]))
            negative_point_cords = []
            for i in range(flow_uv.shape[0]):
                for j in range(flow_uv.shape[1]):
                    if flow_uv[i][j][0] == 0 and flow_uv[i][j][1] == 0:         # 0 means the no motion zone and we have already filter low motion as zero before
                        if random.random() < skip_prob:
                            negative_point_cords.append([j, i])                 # 根据document来看，这个就应该是先horizontal再vertical，也就是这个顺序
            negative_point_cords = np.array(negative_point_cords)       # [:SAM_negative_sample_num]
            negative_point_labels = np.zeros(len(negative_point_cords))         # Make sure that it is less than / equals to SAM_negative_sample_num quantity



        ################## Use SAM to filter out what we need (& use negative points) ##################
        if idx == 0:    # Only consider the first frame now.
            # With sample coordinate
            sam_predictor.set_image(np.uint8(np.transpose(image1.numpy(), (1,2,0))))
            if SAM_negative_sample_num != 0 and len(negative_point_cords) != 0:
                all_point_cords = np.concatenate((positive_point_cords, negative_point_cords), axis=0)
                all_point_labels = np.concatenate((positive_point_labels, negative_point_labels), axis=0)
            else:
                all_point_cords = positive_point_cords
                all_point_labels = positive_point_labels
            
            masks, scores, logits = sam_predictor.predict(
                                            point_coords=all_point_cords, 
                                            point_labels=all_point_labels, 
                                            multimask_output=False,
                                            )
            mask = masks[0]      # TODO: 一定要确定我们这里选择了最大的mask，而没有考虑的第二大和其他的, 这里可能有bug，我们默认了第一个就是最大的mask
            # if verbose:
                # cv2.imwrite("mask_"+str(idx+1)+".png", (np.uint8(mask)*255))
                # annotated_img = show_mask(mask)
                # cv2.imwrite("annotated.png", annotated_img)


            ################## Choose the one we need as the reference for the future tracking ##################
            # Choose a random point in the mask
            target_zone = np.nonzero(mask)      # [(vertical), (horizontal)]
            target_zone = [(target_zone[0][i], target_zone[1][i]) for i in range(len(target_zone[0]))]      # Now, the sturcture is [(vertical, horizontal), ...]
        
            repeat_time = 0
            loop2find = True
            while loop2find:
                loop2find = False
                start_point = target_zone[np.random.choice(len(target_zone), 1, replace=False)[0]]
                start_vertical, start_horizontal = start_point

                repeat_time += 1
                if repeat_time == 100:
                    # In some minor case, it may have infinite loop, so we need to manually break if it is looping
                    print("We are still hard to find a optimal first point, but we cannot let it loop")
                    break

                # Try to choose a start_point that is more centralized (Not close to the border)
                fast_break = False
                for i in range(-15, 15):
                    for j in range(-15, 15):
                        dil_vertical, dil_horizontal = start_vertical + i, start_horizontal + j
                        if (0 <= dil_vertical and dil_vertical < mask.shape[0]) and (0 <= dil_horizontal and dil_horizontal < mask.shape[1]):
                            if mask[dil_vertical][dil_horizontal] == 0:
                                print("We need to change to a new position for the start p Since this one is close to the border of the object...........")
                                loop2find = True
                                fast_break = True
                                break
                        else:
                            # We won't want to consider those that is close to the boundary
                            print("We need to change to a new position Since this one is close to the border of the image...........")
                            loop2find = True
                            fast_break = True
                            break
                    if fast_break:
                        break
            traj_path.append(start_point)

            status, base_img = visualize_traj(base_img, traj_path)
            if status == False:       # If the traj is False, we won't consider it anymore.
                file = open("log.txt", "a")
                file.write("Invalid start point\n")
                return False

        # Read from the last one in traj
        ref_vertical, ref_horizontal = traj_path[-1][0], traj_path[-1][1]


        # Get the average motion vector for point surrounding (8+1 directions) the ref_point; This is because this is the most accurate statistics
        horizon_lists, vertical_lists = [], []
        start_range, end_range = -5, 5

        # Calculate the average motion based on surrounding motion
        search_times = 0
        while len(horizon_lists) == 0:  # If we cannot find a direction, we use average value inside this mask, but we will flag it.
            search_times += 1
            
            if search_times > 1:
                print("This is hard to track!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! and we have tracked " + str(search_times) + " times")
                # TODO: 如果out of boundary那种，search times到了8-10次的就砍掉那后面frame吧，这种非常inaccurate了， 你也可以retrack一个新的点，但是没有什么意义，看整体数量来定吧
                is_hard_to_track = True
                hard_track_idxs.add(idx)

                if abs(start_range) >= flow_uv.shape[0]//2:
                    file = open("log.txt", "a")
                    file.write("This folder has search all space but didn't find any place to track optical flow\n")
                    return False    # If we have already search for the whole graph but didn't find anything to track, we discard this sample

            # Search for a larger space which is nearby 我觉得扩大搜索范围应该是最稳定的选择吧
            for i in range(start_range, end_range):
                for j in range(start_range, end_range):
                    target_vertical, target_horizontal = ref_vertical + i, ref_horizontal + j
                    if 0 <= target_vertical and target_vertical < flow_uv.shape[0] and 0 <= target_horizontal and target_horizontal < flow_uv.shape[1]:
                        if flow_uv[target_vertical, target_horizontal, 0] == 0 or flow_uv[target_vertical, target_horizontal, 1] == 0:
                            continue     # Ignore zero vector to ensure only calculate moving position
                        horizon_lists.append(flow_uv[target_vertical, target_horizontal, 0])      # Horizontal motion strength
                        vertical_lists.append(flow_uv[target_vertical, target_horizontal, 1])     # Vertical motion strength

            # If there isn't any to search, we kepp on a larger space
            start_range -= 10
            end_range += 10

        average_dx = sum(horizon_lists)/len(horizon_lists)
        average_dy = sum(vertical_lists)/len(vertical_lists)
        print("average movement is ", (average_dx, average_dy))
        traj_path.append(( int(traj_path[-1][0] + average_dy), int(traj_path[-1][1] + average_dx)))    # Append the motion in independent order

        print(traj_path)
    

        ##################### Visualize the trajectory path (Debug Purpose) #####################
        status, base_img = visualize_traj(base_img, traj_path)
        if status == False:       # If the traj is False, we won't consider it anymore.
            return False

        cv2.imwrite(os.path.join(store_dir, "traj_path.png"), cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB))

        if traj_visualization:
            status, single_traj_img = visualize_traj(np.uint8(np.transpose(image1.numpy(), (1,2,0))), traj_path[:-1], connect_points=False)
            if status == False:       # If the traj is False, we won't consider it anymore.
                return False
        
            traj_write_path = os.path.join(store_dir, "traj_"+str(idx)+".png")
            # cv2.imwrite(traj_write_path, cv2.cvtColor(single_traj_img, cv2.COLOR_BGR2RGB))
            traj_image_lists.append(traj_write_path)


    # if traj_visualization:
    #     images = []
    #     for filename in traj_image_lists:
    #         images.append(imageio.imread(filename))
    #         # os.remove(filename)     # Remove when used
    #     imageio.mimsave(os.path.join(store_dir, 'traj_motion.gif'), images, duration=0.05)


    # TODO: 可以如果hard to track，就aggressivly多试即便，我们根据这个hard_track_idxs的长度来粗略判断哪个最好，三次里面选最好的
    if is_hard_to_track:
        if len(hard_track_idxs) >= len(img_pairs)//3:       # If more than half of the traj is hard to track, we need to consider discard this one
            file = open("log.txt", "a")
            file.write("we have a lot of times hard to find dx and dy movement. Under this circumstance, we are not very recommended to use the track\n")
            return False


    # Write a file store all position for further utilization
    txt_path = os.path.join(store_dir, "traj_data.txt")
    if os.path.exists(txt_path):
        os.remove(txt_path)
    file = open(txt_path, "a")
    for traj in traj_path:
        file.write(str(traj[0]) + " " + str(traj[1]) + "\n")
    # Save in numpy information
    # with open(os.path.join(store_dir, 'traj_data.npy'), 'wb') as f:
    #     np.save(f, flow_uv)
    print("We write ", traj_path)
    return True



def manage_seq_range(input_dir, store_dir, total_frame_needed):

    lists = os.listdir(input_dir)
    lists = lists[2:-2]
    num_frames_input = len(lists) 
    
    if num_frames_input < total_frame_needed:
        print("The number of frames is too short for constructing the sequnece length needed")
        return False
    

    division_factor = num_frames_input // total_frame_needed
    remain_frame = num_frames_input % total_frame_needed

    gaps = [division_factor for _ in range(total_frame_needed)]
    for idx in range(remain_frame):
        gaps[idx] += 1


    cur_idx = 2
    for global_idx, gap in enumerate(gaps):
        source_path = os.path.join(input_dir, "im_"+str(cur_idx)+".jpg")
        destination_path = os.path.join(store_dir, "im_"+str(global_idx)+".jpg")

        shutil.copyfile(source_path, destination_path)
        cur_idx += gap

    return True


def generate_pairs(dirname, start_idx, end_idx):
    img_pairs = []
    for idx in range(start_idx, end_idx):
        img1 = osp.join(dirname, f'im_{idx}.jpg')
        img2 = osp.join(dirname, f'im_{idx+1}.jpg')
        # img1 = f'{idx:06}.png'
        # img2 = f'{idx+1:06}.png'
        img_pairs.append((img1, img2))

    return img_pairs


def process_partial_request(request_list, num_frames, traj_visualization, viz_root_dir):
    

    # Init the optical flow model
    optical_flow_model = build_model()

    # Init SAM for segmentation task
    model_type = "vit_h"
    weight_path = "pretrained/sam_vit_h_4b8939.pth"
    SAM_positive_sample_num = 20    # How many points we use for the positive sample num ()
    SAM_negative_sample_num = 0    # How many points we use for the negative sample num

    print("In multi processing, we will build an instance of mask_generator independently")
    sam = sam_model_registry[model_type](checkpoint=weight_path).to(device="cuda")
    mask_generator = SamAutomaticMaskGenerator(sam)
    print("In multi processing, we will build an instance of sam_predictor independently")
    sam_predictor = SamPredictor(sam)


    counter = 0
    while True:
        counter += 1
        if counter == 10:
            counter = 0
            gc.collect()
            print("We will sleep here to clear memory")
            time.sleep(5)
        info = request_list[0]
        request_list = request_list[1:]
        if info == None:
            print("This queue ends")
            break
        

        # Process each sub_input_dir and store the information there
        sub_input_dir = info


        img_pairs = generate_pairs(sub_input_dir, 0, num_frames-1)
        print(img_pairs)

        with torch.no_grad():

            # Calculate the optical flow and return a status to say whther this generated flow is usable
            status = calculate_flow(viz_root_dir, sub_input_dir, img_pairs, optical_flow_model, sam_predictor, SAM_positive_sample_num, SAM_negative_sample_num, 
                                    mask_generator, traj_visualization, keep_size = True)

            # file = open("log.txt", "a")
            print("The status for folder " + sub_input_dir + " is " + str(status) + "\n")

            if status == False:
                # If the status is failed, we will remove it afterwords
                print("The status is Failed, so we won't store this one as one promising data")
            else:
                print("We have successfully process one!")


if __name__ == '__main__':

    # Manage the paramter
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default = '../validation_flow14/')
    parser.add_argument('--num_workers', type = int, default = 1)       # starting index of the image sequence
    parser.add_argument('--viz_root_dir', default = 'viz_results')
    parser.add_argument('--traj_visualization', default = True)      # If this is True, 
    
    # list_start = 0
    # list_end = 25000
    num_frames = 14

    args = parser.parse_args()
    input_dir = args.input_dir
    num_workers = args.num_workers
    viz_root_dir = args.viz_root_dir
    traj_visualization = args.traj_visualization



    store_idx = 0
    dir_list = []
    for sub_input_name in sorted(os.listdir(input_dir)):
        sub_input_dir = os.path.join(input_dir, sub_input_name)
        # sub_store_dir = os.path.join(store_dir, "0"*(7-len(str(store_idx)))+str(store_idx))
        store_idx += 1
        dir_list.append(sub_input_dir)

    # Truncate the list to the target
    # dir_list = dir_list[list_start:]


    # Use multiprocessing to handle to speed up
    num = math.ceil(len(dir_list) / num_workers)
    for idx in range(num_workers):
        # set_start_method('spawn', force=True)

        request_list = dir_list[:num]
        request_list.append(None)
        dir_list = dir_list[num:]


        process_partial_request(request_list, num_frames, traj_visualization, viz_root_dir)   # This is for debug purpose
        # p = mp.Process(target=process_partial_request, args=(request_list, num_frames, traj_visualization, viz_root_dir, ))
        # p.start()

    print("Submitted all jobs!")
    # p.join()        # 好像不加这个multiprocess就莫名自己结束了
    print("All task finished!")


        