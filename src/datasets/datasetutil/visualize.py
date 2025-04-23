import numpy as np
import open3d as o3d
def visualize_scene_flow(camera_space_points, camera_space_points_next, scene_flow):
    obj = gen_scene_flow_visualize_object(camera_space_points, camera_space_points_next, scene_flow)
    o3d.visualization.draw_geometries(obj, window_name="Scene Flow Visualization")


def gen_scene_flow_visualize_object(camera_space_points, camera_space_points_next, scene_flow):
    """
    Function to visualize the scene flow between two sets of 3D points
    Inputs:
    - camera_space_points: 3D points in the camera space at time t
    - camera_space_points_next: 3D points in the camera space at time t+1
    - scene_flow: 3D vectors representing the flow from points at time t to t+1
    Outputs:
    - Visualizes the scene flow using Open3D, showing start points, end points, and flow vectors
    """

    # Create a point cloud for the starting points
    import open3d as o3d
    pcd_start = o3d.geometry.PointCloud()
    pcd_start.points = o3d.utility.Vector3dVector(camera_space_points)
    pcd_start.paint_uniform_color([0, 1, 0])  # Green color for the starting points
    pcd_end = o3d.geometry.PointCloud()
    pcd_end.points = o3d.utility.Vector3dVector(camera_space_points_next)
    pcd_end.paint_uniform_color([0, 0, 1])  # Blue color for the end points
    # Create lines to represent the scene flow vectors
    lines = []
    colors_blue = []
    colors_green = []
    for i, start_point in enumerate(camera_space_points):
        end_point = start_point + scene_flow[i]
        lines.append([i, len(camera_space_points) + i])  # Line from start to end
        colors_blue.append([0, 0, 1])  # Red color for the flow vectors
        colors_green.append([0, 1, 0])  # Green color for the flow vectors

    # Create a point cloud for the end points
    end_points = camera_space_points + scene_flow
    # pcd_end = o3d.geometry.PointCloud()
    # pcd_end.points = o3d.utility.Vector3dVector(end_points)

    # Combine start and end points
    combined_points = np.vstack((camera_space_points, end_points))
    start_half = np.vstack((camera_space_points, (camera_space_points+end_points)/2))
    end_half = np.vstack(((camera_space_points+end_points)/2,end_points))
    # Create a LineSet for the flow vectors
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(start_half)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors_green)
    line_set_end = o3d.geometry.LineSet()
    line_set_end.points = o3d.utility.Vector3dVector(end_half)
    line_set_end.lines = o3d.utility.Vector2iVector(lines)
    line_set_end.colors = o3d.utility.Vector3dVector(colors_blue)
    # Create a LineSet for the flow vectors
    return [pcd_start, line_set,line_set_end,pcd_end]

def visualize_point_trajectory(global_space_trajectories):
    """
    Function to visualize the point trajectories in 3D space
    Inputs:
    - global_space_trajectories: List of 3D points representing the trajectories
    Outputs:
    - Visualizes the point trajectories using Open3D
    """
    import open3d as o3d
    pcds = []
    for i in range(len(global_space_trajectories)-1):
        begin = global_space_trajectories[i]
        end = global_space_trajectories[i+1]
        scene_flow = end - begin
        obj = gen_scene_flow_visualize_object(begin, end, scene_flow)
        pcds += obj

    # visualize the point cloud
    o3d.visualization.draw_geometries(pcds, window_name="Scene Flow Visualization")


def vis(pcs:list):
    pcds = []
    for i in range(len(pcs)):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcs[i])
        pcds.append(pcd)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    pcds.append(axis)

    # visualize the point cloud
    o3d.visualization.draw_geometries(pcds, window_name="Point Cloud Visualization")