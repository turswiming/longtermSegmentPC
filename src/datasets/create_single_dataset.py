#search folders in val_data_list and train_data_list
#copy the file to another path whith the same relative path
#example: src/datasets/val/dog/1.jpg -> worksapce/datasets2/val/dog/1.jpg
#example: src/datasets/train/bear/1.jpg -> worksapce/datasets2/train/bear/1.jpg

import os
import shutil

val_data_list = ["car-shadow"]
train_data_list = ["car-shadow"]

# Define source and destination base paths
src_base_path = "/workspace/guess-what-moves/DAVIS2016"
dest_base_path = "/workspace/guess-what-moves/data/DAVIS2016"

subfolderList = [
    "Annotations", 
    "FlowImages_gap-1",
    "FlowImages_gap-2",
    "FlowImages_gap1",
    "FlowImages_gap2",
    "Flows_gap2",
    "Flows_gap1",
    "Flows_gap-1",
    "Flows_gap-2",
    "JPEGImages",
    # "ImageSets"
    ]
# Copy validation data files
for subfolder in subfolderList:
    src_base_path_val = "/workspace/guess-what-moves/DAVIS2016/" + subfolder+"/480p/"
    dest_base_path_val = "/workspace/guess-what-moves/data/DAVIS2016/" + subfolder+"/480p/"
    for data in val_data_list:
        src_path = os.path.join(src_base_path_val, data)
        dest_path = os.path.join(dest_base_path_val, data)
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        file_to_copy = sorted([x.split(".")[0] for x in os.listdir(src_path)])
        if "00007" in file_to_copy:
            file_to_copy = "00007.jpg"
        elif "00008" in file_to_copy:
            file_to_copy = "00008.jpg"
        for file in os.listdir(src_path):
            src_file_path = os.path.join(src_path, file_to_copy.split(".")[0]+"."+file.split(".")[1])
            src_file_path = os.path.join(src_path, file)
            dest_file_path = os.path.join(dest_path, file)
            shutil.copy(src_file_path, dest_file_path)
            print("Copied file: " + src_file_path + " to " + dest_file_path)
    
    # for data in train_data_list:
    #     src_path = os.path.join(src_base_path_val, data)
    #     dest_path = os.path.join(dest_base_path_val, data)
    #     if not os.path.exists(dest_path):
    #         os.makedirs(dest_path)
    #     file_to_copy = sorted(os.listdir(src_path))[0]
    #     for file in os.listdir(src_path):
    #         src_file_path = os.path.join(src_path, file_to_copy.split(".")[0]+"."+file.split(".")[1])
    #         dest_file_path = os.path.join(dest_path, file)
    #         shutil.copy(src_file_path, dest_file_path)
    #         print("Copied file: " + src_file_path + " to " + dest_file_path)

