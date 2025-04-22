import os
import numpy as np
import json
import cv2

#copy from movi_f README
"""python
if "forward_flow" in layers:
    result["metadata"]["forward_flow_range"] = [
        data_ranges["forward_flow"]["min"] / scale * 512,
        data_ranges["forward_flow"]["max"] / scale * 512]
    result["forward_flow"] = [
        subsample_nearest_neighbor(read_png(frame_path)[..., :2], 
                                   target_size)
        for frame_path in paths["forward_flow"]]
"""
# I don`t know how to express what I feel after check the details of the code
# Firstly, after check the data distribution, the image store u, v optical flow on y,z channel
# Not x, y channel as movi_f write.
# So we need to select [...,1:] from png, rather than [..., :2]
# Later, the magic number 512 don`t have any reason, we can just skip it.
# Then we get the correct results.


"""
- "forward_flow": (s, h, w, 2) [uint16]
  Forward optical flow in the form (delta_row, delta_column).
  The values are stored as uint16 and span the range specified in
  sample["metadata"]["forward_flow_range"]. To convert them back to pixels use:
    minv, maxv = sample["metadata"]["forward_flow_range"]
    depth = sample["forward_flow"] / 65535 * (maxv - minv) + minv
"""


def read_forward_flow(img_path, data_ranges_path):
    """
    Reads and processes the forward flow data from an image file and a JSON file containing data ranges.
    Args:
        img_path (str): The file path to the image containing the forward flow data.
        data_ranges_path (str): The file path to the JSON file containing the data ranges for normalization.
    Returns:
        numpy.ndarray: A 2D array representing the processed forward flow data with normalized values.
    Notes:
        - this is different from the original movi_f code, check comments for detail.
        - The function reads the forward flow image and selects only the y and z channels (ignoring the x channel).
        - The forward flow values are normalized using the min and max values provided in the JSON file.
        - The normalization formula used is:
          `normalized_value = (raw_value / 65535) * (max_value - min_value) + min_value`
    """
    with open(data_ranges_path, 'r') as f:
        data_ranges = json.load(f)
    forward_flow_max = data_ranges["forward_flow"]["max"]
    forward_flow_min = data_ranges["forward_flow"]["min"]
    minv, maxv = forward_flow_min, forward_flow_max # skip multi 512

    forward_flow = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    forward_flow = forward_flow[...,1:] # select y,z channel, not x,y channel
    forward_flow = forward_flow / 65535 * (maxv - minv) + minv

    return forward_flow

def read_segmentation(img_path):
    """
    Reads a segmentation image and processes it to extract the instance IDs.
    Args:
        img_path (str): The file path to the segmentation image.
    Returns:
        numpy.ndarray: A 2D array representing the segmentation image with instance IDs.
    """
    seg_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    seg_img = seg_img.astype(np.uint8)
    return seg_img