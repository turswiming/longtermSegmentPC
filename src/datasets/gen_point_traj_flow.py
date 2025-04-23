import os
import numpy as np
import json
import cv2
import pyquaternion as pyquat
from .reverse_projection import image_reverse_projection, point_reverse_projection
from .datasetutil import clamp, rgb_array_to_int32, camera_space_to_world_space
from .datasetutil import read_forward_flow,read_segmentation
from .datasetutil import visualize_scene_flow, vis,visualize_point_trajectory


def most_likely_instance(segmentations,color,instances)-> int:
    """
    Identify the most likely color for a given segmentation.
    Args:
        segmentation (numpy.ndarray): Segmentation image.
        instances (list): List of instances object, same in metadata.
    Returns:
        int: The most likely color.
    """
    # Count the occurrences of each instance in the segmentation
    gt_features = [instance["visibility"] for instance in instances]
    color_feature = [len(segmentation[segmentation==color]) for segmentation in segmentations]
    gt_features = np.array(gt_features)
    color_feature = np.array(color_feature)
    #compute the most likely instance
    diff = []
    for gt_feature in gt_features:
        difference = sum(abs(gt_feature - color_feature))
        diff.append(difference)    
    #get the min index
    diff = np.array(diff)
    min_index = np.argmin(diff, axis=0)
    return min_index

def get_obj_rotation_list_list(metadata):
    """
    Get the object rotation list from the metadata.
    Args:
        metadata (dict): Metadata containing object rotations.
    Returns:
        list: List of object rotation matrices.
    """
    object_rotation_list_list = []
    for instance in metadata["instances"]:
        object_rotation_list = []
        for j in range(len(instance["quaternions"])):
            quaternions = np.array(instance["quaternions"][j])
            rot = pyquat.Quaternion(quaternions).rotation_matrix
            object_rotation_list.append(rot)
        object_rotation_list_list.append(object_rotation_list)
    return object_rotation_list_list

def get_obj_center_list(metadata):
    """
    Get the object center list from the metadata.
    Args:
        metadata (dict): Metadata containing object centers.
    Returns:
        list: List of object centers.
    """
    object_center_list_list = []
    for instance in metadata["instances"]:
        object_center_list = []
        for j in range(len(instance["positions"])):
            position = instance["positions"][j]
            object_center_list.append(position)
        object_center_list_list.append(object_center_list)
    return object_center_list_list  

def get_metadata(metadata_path):
    """
    Get the metadata from the dataset path.
    Args:
        metadata_path (str): Path to the metadata_path.
    Returns:
        dict: Metadata containing camera parameters and object instances.
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata

def get_object_rotation_tensors(res,num_frames,num_objects, indices_of_instance,object_rotation_list_list):
    """
    Get the object rotation tensors for each frame.
    Args:
        res (int): Resolution of the images.
        num_frames (int): Number of frames in the dataset.
        num_objects (int): Number of objects in the dataset.
        indices_of_instance (numpy.ndarray): Indices of instances in the segmentation.
        object_rotation_list_list (list): List of object rotation matrices.
    Returns:
        list: List of object rotation tensors for each frame.
    """
    object_rotation_tensor = np.zeros((res,res,3,3), dtype=np.float32)
    #object rotation tensor is a 4D tensor, the last dimension is 3x3 matrix
    #defult is eye(3)
    object_rotation_tensor[...,0,0] = 1
    object_rotation_tensor[...,1,1] = 1
    object_rotation_tensor[...,2,2] = 1
    object_rotation_tensor_list = []
    for frame in range(num_frames):
        object_rotation_tensor_list.append(np.copy(object_rotation_tensor))
    for frame in range(num_frames):
        for obj_id in range(num_objects):
            object_rotation_tensor_list[frame][indices_of_instance == obj_id] = object_rotation_list_list[obj_id][frame]
    return object_rotation_tensor_list


def process_one_sample(metadata_path,dep_img_path,segmentation_path, visualize=False):
    metadata = get_metadata(metadata_path)

    K = np.array(metadata["camera"]["K"]).reshape(3, 3)
    positions = metadata["camera"]["positions"]
    quaternions = metadata["camera"]["quaternions"]
    res = metadata["flags"]["resolution"]
    num_frames = len(positions)
    world_space_points_list = []
    segmentation_list = []
    seg_color_to_instance_ID_map = {}
    for frame in range(num_frames):

        camera_position = positions[frame]
        camera_quaternion = quaternions[frame]
        camera_position = np.array(camera_position).reshape(3, 1)
        camera_quaternion = np.array(camera_quaternion).reshape(4, 1)
        camera_quaternion = camera_quaternion.flatten()

        distance = cv2.imread(dep_img_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

        segmentation = read_segmentation(segmentation_path)
        
        internal_matrix = K * res
        fx, fy = internal_matrix[0, 0], internal_matrix[1, 1]
        cx, cy = internal_matrix[0, 2], internal_matrix[1, 2]
        camera_space_points = image_reverse_projection(distance, fx, fy, cx, cy)
        camera_space_points = camera_space_points.reshape(-1, 3)
        world_space_points = camera_space_to_world_space(camera_space_points, camera_position, camera_quaternion)
        world_space_points_list.append(world_space_points)
        segmentation = rgb_array_to_int32(segmentation)
        segmentation_list.append(segmentation)
    
    # identify the instance ID
    # convert 8 bit r, g, b to int 32
    unique_colors = np.unique(np.array(segmentation_list))
    for color in unique_colors:
        if color == 0:
            continue
        if color not in seg_color_to_instance_ID_map:
            #return the most likely id
            id = most_likely_instance(segmentation_list,color,metadata["instances"])
            seg_color_to_instance_ID_map[color] = id
            show_axis=False,
    
    object_rotation_list_list = get_obj_rotation_list_list(metadata)
    world_space_center_list_list = get_obj_center_list(metadata)
    f = dep_img_path.split("/")[-1].split(".")[0]
    f = int(f)

    segmentation = segmentation_list[f]
    world_space_points = world_space_points_list[f]
    indices_of_instance = np.zeros_like(segmentation, dtype=np.int8)
    indices_of_instance -= 1 #set background to -1 so that in following processing we can ignore it
    #set the instance id to the segmentation
    for color, instance_id in seg_color_to_instance_ID_map.items():
        indices_of_instance[segmentation == color] = instance_id
    
    num_objects = len(seg_color_to_instance_ID_map)

    world_space_center_tensor = np.zeros((res,res,3), dtype=np.float32)
    world_space_center_tensor_list = []
    for frame in range(num_frames):
        world_space_center_tensor_list.append(np.copy(world_space_center_tensor))
    for frame in range(num_frames):
        for obj_id in range(num_objects):
            world_space_center_tensor_list[frame][indices_of_instance == obj_id] = world_space_center_list_list[obj_id][frame]
    world_space_center_tensor = world_space_center_tensor_list[f]
    world_space_center_tensor = world_space_center_tensor.reshape(-1, 3)


    object_rotation_tensor_list = get_object_rotation_tensors(res, num_frames, num_objects, indices_of_instance, object_rotation_list_list)
    object_rotation_tensor = object_rotation_tensor_list[f]
    object_rotation_tensor = object_rotation_tensor.reshape(-1, 3, 3)
    
    object_space_points =  np.einsum('nji,nj->ni', object_rotation_tensor, (world_space_points - world_space_center_tensor))
    #compute world space points num_frames
    world_space_trajectories_list = []
    for frame in range(num_frames):
        object_rotation_tensor = object_rotation_tensor_list[frame]
        object_rotation_tensor = object_rotation_tensor.reshape(-1, 3, 3)
        world_space_trajectory = np.einsum('nij,nj->ni', object_rotation_tensor, (object_space_points))
        world_space_trajectory = world_space_trajectory + world_space_center_tensor_list[frame].reshape(-1, 3)
        world_space_trajectories_list.append(world_space_trajectory)
        
    world_space_trajectories_tensor = np.array(world_space_trajectories_list)
    if visualize:
        # visualize_scene_flow(world_space_points, world_space_points_next, scene_flow)
        visualize_point_trajectory(world_space_trajectories_tensor)
    return world_space_trajectories_tensor

def get_traj_flow_pointcloud(dataset_path, data_name, frame_idx):
    """
    Get the trajectory and flow data from the dataset path.
    
    Args:
        dataset_path (str): Path to dataset directory
        data_name (str): Name of the data
        frame_idx (int): Frame index for reading

    Returns:
        np.ndarray: Trajectory data
        np.ndarray: Flow data
    """
    metadata_path = os.path.join(dataset_path, "metadata","480p",data_name, "metadata.json")
    dep_img_path = os.path.join(dataset_path, "Depth","480p",data_name, f"{frame_idx}.tiff")
    segmentation_path = os.path.join(dataset_path, "Annotations","480p",data_name, f"{frame_idx}.png")
    trajectory = process_one_sample(metadata_path,dep_img_path,segmentation_path,visualize=False)
    frame_idx_int = int(frame_idx)
    if frame_idx_int < trajectory.shape[0]-1:
        scene_flow = trajectory[frame_idx_int+1] -trajectory[frame_idx_int]
    else:
        scene_flow = trajectory[frame_idx_int] - trajectory[frame_idx_int-1]
    point_cloud = trajectory[frame_idx_int]
    return trajectory ,scene_flow, point_cloud

def process_gt(gt_path):
    """
    Get the ground truth data from the dataset path.
    
    Args:
        gt_path (str): Path to ground truth directory

    Returns:
    tuple:
        np.ndarray: Ground truth color data
        np.ndarray: Instance IDs
        np.ndarray: One-hot encoded instance IDs
    """
    image = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
    np_image = np.array(image)
    # Convert the image to a 3D numpy array
    np_image = np_image.reshape(-1, 3)
    int32_image = rgb_array_to_int32(np_image)
    unique_colors = sorted(np.unique(int32_image))
    # Create a dictionary to map colors to instance IDs
    color_to_instance_id = {}
    for color in unique_colors:
        if color not in color_to_instance_id:
            # Assign a new instance ID to the color
            instance_id = len(color_to_instance_id)
            color_to_instance_id[color] = instance_id
    # Create an array to hold the instance IDs
    instance_ids = np.zeros_like(int32_image, dtype=np.int32)
    # Map the colors to instance IDs
    onehot = np.zeros((len(unique_colors), int32_image.shape[0]), dtype=np.int32)
    for color, instance_id in color_to_instance_id.items():
        instance_ids[int32_image == color] = instance_id
        onehot[instance_id][int32_image == color] = 1
    
    return np_image, instance_ids, onehot
def get_gt(dataset_path,data_name,frame_idx):
    """
    Get the ground truth data from the dataset path.
    
    Args:
        dataset_path (str): Path to dataset directory
        data_name (str): Name of the data
        frame_idx (int): Frame index for reading

    tuple:
        np.ndarray: Ground truth color data
        np.ndarray: Instance IDs
        np.ndarray: One-hot encoded instance IDs
    """
    gt_path = os.path.join(dataset_path, "Annotations","480p",data_name, f"{frame_idx}.png")
    np_color, instance_ids, onehot = process_gt(gt_path)
    np_color = np_color.astype(np.float32)
    np_color /= 255.0
    return np_color, instance_ids, onehot

def get_rgb(dataset_path,data_name,frame_idx):
    """
    Get the RGB data from the dataset path.
    
    Args:
        dataset_path (str): Path to dataset directory
        data_name (str): Name of the data
        frame_idx (int): Frame index for reading

    Returns:
        np.ndarray: RGB data
    """
    rgb_path = os.path.join(dataset_path, "JPEGImages","480p",data_name, f"{frame_idx}.jpg")
    rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
    rgb = np.array(rgb)
    # Convert the image to a 3D numpy array
    rgb = rgb.reshape(-1, 3)
    rgb = rgb.astype(np.float32)
    rgb /= 255.0
    return rgb