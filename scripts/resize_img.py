import os, sys, shutil
import cv2

if __name__ == "__main__":
    input_path = "/nfs/turbo/jjparkcv-turbo-large/boyangwa/resize"
    output_path = "/nfs/turbo/jjparkcv-turbo-large/boyangwa/resize_resized"

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    for img_name in os.listdir(input_path):
        img_path = os.path.join(input_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (384, 256))
        store_path = os.path.join(output_path, img_name)
        cv2.imwrite(store_path, img)