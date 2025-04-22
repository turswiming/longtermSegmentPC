import numpy as np
import pyquaternion as pyquat
def clamp(value,min_val,max_val):
    """
    Clamp a given value within a specified range.
    Parameters:
    -----------
    value : float or int
        The value to be clamped.
    min_val : float or int
        The minimum allowable value.
    max_val : float or int
        The maximum allowable value.
    Returns:
    --------
    float or int
        The clamped value, which will be:
        - `min_val` if `value` is less than `min_val`.
        - `max_val` if `value` is greater than `max_val`.
        - `value` if it is within the range [min_val, max_val].
    """
    return min(max(value,min_val),max_val)

def rgb_array_to_int32(rgb_array):
    """
    Converts a NumPy array of shape (N, 3) representing RGB values to a 32-bit integer array.
    Args:
        rgb_array (numpy.ndarray): Array of shape (N, 3) with 8-bit RGB values.
    Returns:
        numpy.ndarray: Array of 32-bit integers representing the RGB colors.
    """
    return (rgb_array[..., 0].astype(np.uint32) << 16) | \
           (rgb_array[..., 1].astype(np.uint32) << 8) | \
           rgb_array[..., 2].astype(np.uint32)

def camera_space_to_world_space(camera_space_points, camera_position, camera_quaternion):
    """
    Convert camera space points to world space points using camera position and quaternion.
    Args:
        camera_space_points (numpy.ndarray): 3D points in camera space.
        camera_position (numpy.ndarray): Camera position in world space.
        camera_quaternion (numpy.ndarray): Camera orientation as a quaternion.
    Returns:
        numpy.ndarray: 3D points in world space.
    """
    rot = pyquat.Quaternion(camera_quaternion).rotation_matrix
    return (rot @ camera_space_points.T).T + camera_position.T