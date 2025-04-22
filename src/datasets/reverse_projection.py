
import numpy as np

PROJECTION_FACTOR = [1,1,-1]

def point_reverse_projection(
    u:float,
    v:float,
    fx:float,
    fy:float,
    cx:float,
    cy:float,
    distance_map:np.array
    )-> tuple[float, float, float]:
    """
    Parameters:
        u (float): Pixel x-coordinate.
        v (float): Pixel y-coordinate.
        fx (float): Focal length in the x direction. Remember to multiply by the resolution.
        fy (float): Focal length in the y direction. Remember to multiply by the resolution.
        cx (float): Principal point x-coordinate (negative value, same as in movi_f). Remember to multiply by the resolution.
        cy (float): Principal point y-coordinate (negative value, same as in movi_f). Remember to multiply by the resolution.
        distance_map (np.ndarray): Distance map (originated from movi_f, not a depth map).

    Returns:
        tuple[float, float, float]: 3D point in camera space (x, y, z).
    """
    angle_to_center_x = (u + cx) / fx 
    angle_to_center_y = (v + cy) / fy

    distance = distance_map[int(v), int(u)]
    x = 1 * angle_to_center_x
    y = 1 * angle_to_center_y
    z = 1 / np.sqrt(1 + x**2 + y**2)
    x = distance * x
    y = distance * y
    z = distance * z
    return PROJECTION_FACTOR[0]*x, PROJECTION_FACTOR[1]*y, PROJECTION_FACTOR[2]*z
    

def image_reverse_projection(
    distance_map:np.ndarray,
    fx:float,
    fy:float,
    cx:float,
    cy:float
    ) -> np.ndarray:
    """
    Parameters:
        distance_map (np.ndarray): Distance map (originated from movi_f, not a depth map).
        fx (float): Focal length in the x direction. Remember to multiply by the resolution.
        fy (float): Focal length in the y direction. Remember to multiply by the resolution.
        cx (float): Principal point x-coordinate (negative value, same as in movi_f). Remember to multiply by the resolution.
        cy (float): Principal point y-coordinate (negative value, same as in movi_f). Remember to multiply by the resolution.
    Returns:
        np.ndarray: 3D point cloud in camera space.
    """
    height, width = distance_map.shape
    u_mesh_origin, v_mesh_origin = np.meshgrid(np.arange(width), np.arange(height))
    u_mesh_origin = u_mesh_origin.astype(np.float32)
    v_mesh_origin = v_mesh_origin.astype(np.float32)

    u_mesh = (u_mesh_origin + cx) / fx
    v_mesh = (v_mesh_origin + cy) / fy
    z = distance_map / np.sqrt(1 + u_mesh**2 + v_mesh**2)
    x = z * u_mesh
    y = z * v_mesh
    camera_space_points = np.stack((PROJECTION_FACTOR[0]*x, PROJECTION_FACTOR[1]*y, PROJECTION_FACTOR[2]*z), axis=-1)
    return camera_space_points
