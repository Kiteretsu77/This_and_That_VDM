import os, sys
import PIL
from tqdm import tqdm
import numpy as np
import argparse

# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from track_anything_code.tools.interact_tools import SamControler
from track_anything_code.tracker.base_tracker import BaseTracker


class TrackingAnything():
    def __init__(self, sam_checkpoint, xmem_checkpoint, args):
        self.args = args
        self.sam_checkpoint = sam_checkpoint
        self.xmem_checkpoint = xmem_checkpoint
        self.samcontroler = SamControler(self.sam_checkpoint, args.sam_model_type, args.device)
        self.xmem = BaseTracker(self.xmem_checkpoint, device=args.device)

    
    def first_frame_click(self, image: np.ndarray, points:np.ndarray, labels: np.ndarray, multimask=True):
        mask, logit, painted_image = self.samcontroler.first_frame_click(image, points, labels, multimask)
        return mask, logit, painted_image


    def generator(self, images: list, template_mask:np.ndarray):
        
        masks = []
        logits = []
        painted_images = []
        for i in tqdm(range(len(images)), desc="Tracking image"):
            if i ==0:           
                mask, logit, painted_image = self.xmem.track(images[i], template_mask)
                masks.append(mask)
                logits.append(logit)
                painted_images.append(painted_image)
                
            else:
                mask, logit, painted_image = self.xmem.track(images[i])
                masks.append(mask)
                logits.append(logit)
                painted_images.append(painted_image)
        return masks, logits, painted_images
    
        
def parse_augment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--sam_model_type', type=str, default="vit_h")
    parser.add_argument('--port', type=int, default=6080, help="only useful when running gradio applications")  
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--mask_save', default=False)
    args = parser.parse_args()

    if args.debug:
        print(args)
    return args 


# if __name__ == "__main__":
#     masks = None
#     logits = None
#     painted_images = None
#     images = []
#     image  = np.array(PIL.Image.open('/hhd3/gaoshang/truck.jpg'))
#     args = parse_augment()
#     # images.append(np.ones((20,20,3)).astype('uint8'))
#     # images.append(np.ones((20,20,3)).astype('uint8'))
#     images.append(image)
#     images.append(image)

#     mask = np.zeros_like(image)[:,:,0]
#     mask[0,0]= 1
#     trackany = TrackingAnything('/ssd1/gaomingqi/checkpoints/sam_vit_h_4b8939.pth','/ssd1/gaomingqi/checkpoints/XMem-s012.pth', args)
#     masks, logits ,painted_images= trackany.generator(images, mask)
        
        
    
    
    