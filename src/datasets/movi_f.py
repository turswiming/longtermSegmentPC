import os
from PIL import Image
import torch
from einops import rearrange
import numpy as np
from torchvision import transforms

# using einops read this file
path = "/home/lzq/workspace/guess-what-moves/data/DAVIS2016/Flows_gap-1/480p/blackswan/00001.flo"




origin_path = "/home/lzq/workspace/movi-f/outputs"
save_path = "/home/lzq/workspace/guess-what-moves/data/MOVI_F"

annotation_save_dir = os.path.join(save_path, "Annotations/480p")
JPEGImages_save_dir = os.path.join(save_path,"JPEGImages/480p")
FlowImagesdir1_save_dir = os.path.join(save_path,"FlowImages_gap-1/480p")
FlowImagesdir2_save_dir = os.path.join(save_path,"FlowImages_gap1/480p")
Flowdir1_save_dir = os.path.join(save_path,"Flows_gap-1/480p")
Flowdir2_save_dir = os.path.join(save_path,"Flows_gap1/480p")
def save_flo(image, filename):
    import numpy as np
    TAG_FLOAT = 202021.25

    flow = np.array(image, dtype=np.float32)
    #drop the last channel
    flow = flow[:,:,:2]
    h, w, c = flow.shape
    assert c == 2, "Flow must have shape (H, W, 2)"

    with open(filename, 'wb') as f:
        np.array(TAG_FLOAT, dtype=np.float32).tofile(f)
        np.array(w, dtype=np.int32).tofile(f)
        np.array(h, dtype=np.int32).tofile(f)
        flow.tofile(f)

for dir_name in os.listdir(origin_path):
    dir_path = os.path.join(origin_path,dir_name)
    print(dir_path)

    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path,file)
        print(file_path)
        if "segmentation"   in file_path:
            image = Image.open(file_path)
            save_annotation = os.path.join(annotation_save_dir,dir_name,os.path.splitext(file)[0] + '.png').replace("segmentation_","")
            if not os.path.exists(os.path.join(annotation_save_dir,dir_name)):
                os.makedirs(os.path.join(annotation_save_dir,dir_name))
            image.save(save_annotation)
            pass
        if "rgba" in file_path:
            image = Image.open(file_path).convert("RGB")
            save_RGBA = os.path.join(JPEGImages_save_dir,dir_name,os.path.splitext(file)[0] + '.jpg').replace("rgba_","")
            if not os.path.exists(os.path.join(JPEGImages_save_dir,dir_name)):
                os.makedirs(os.path.join(JPEGImages_save_dir,dir_name))
            image.save(save_RGBA)
            pass
        if "forward_flow" in file_path:
            # Load the image
            image = Image.open(file_path)
            # Convert the image to a tensor
  
            # Save the tensor as a .flo file
            if not os.path.exists(os.path.join(Flowdir1_save_dir,dir_name)):
                os.makedirs(os.path.join(Flowdir1_save_dir,dir_name))
            if not os.path.exists(os.path.join(Flowdir2_save_dir,dir_name)):
                os.makedirs(os.path.join(Flowdir2_save_dir,dir_name))
            flo_file_path1 = os.path.join(Flowdir1_save_dir,dir_name,os.path.splitext(file)[0] + '.flo').replace("forward_flow_","")
            flo_file_path2 = os.path.join(Flowdir2_save_dir,dir_name,os.path.splitext(file)[0] + '.flo').replace("forward_flow_","")
            #print the max and min in the image in three channels

            save_flo(image, flo_file_path1)
            save_flo(image, flo_file_path2)
            #copy image to FlowImagesdir
            if not os.path.exists(os.path.join(FlowImagesdir1_save_dir,dir_name)):
                os.makedirs(os.path.join(FlowImagesdir1_save_dir,dir_name))
            if not os.path.exists(os.path.join(FlowImagesdir2_save_dir,dir_name)):
                os.makedirs(os.path.join(FlowImagesdir2_save_dir,dir_name))
            image.save(os.path.join(FlowImagesdir1_save_dir,dir_name,os.path.splitext(file)[0] + '.jpg').replace("forward_flow_",""))
            image.save(os.path.join(FlowImagesdir2_save_dir,dir_name,os.path.splitext(file)[0] + '.jpg').replace("forward_flow_",""))
