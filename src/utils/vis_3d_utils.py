import numpy as np
import open3d as o3d


def convert_numpy_to_open3d(numpy_points, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(numpy_points)
    if color is not None:
        colors = np.repeat(
            np.array(color).reshape(-1, 3), numpy_points.shape[0], axis=0
        )
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def convert_open3d_to_numpy(open3d_pcd):
    return np.asarray(open3d_pcd.points)


def draw_camera(cam_pose, size=0.1, color=(1, 0, 0), z=1):
    # add camera form
    points = np.array(
        [[0, 0, 0], [-1, -1, z], [-1, 1, z], [1, -1, z], [1, 1, z]]
    ).reshape(-1, 3)
    lines = np.array(
        [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 4], [4, 3], [3, 1]]
    ).reshape(-1, 2)
    points = points * size
    points_in_cam = cam_pose[:3, :3].dot(points.T).T + cam_pose[:3, 3]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_in_cam)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    if color is not None:
        if isinstance(color, np.ndarray):
            line_set.colors = o3d.utility.Vector3dVector(color)
        elif isinstance(color, list):
            new_colors = np.repeat(
                np.array(color).reshape(-1, 3), lines.shape[0], axis=0
            )
            line_set.colors = o3d.utility.Vector3dVector(new_colors)

    # add camera coordinate
    points = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).reshape(-1, 3)
    lines = np.array([[0, 1], [0, 2], [0, 3]]).reshape(-1, 2)
    color_frames = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).reshape(-1, 3)
    points = points * size
    points_in_cam = cam_pose[:3, :3].dot(points.T).T + cam_pose[:3, 3]
    coordinate_set = o3d.geometry.LineSet()
    coordinate_set.points = o3d.utility.Vector3dVector(points_in_cam)
    coordinate_set.lines = o3d.utility.Vector2iVector(lines)
    coordinate_set.colors = o3d.utility.Vector3dVector(color_frames)
    return line_set, coordinate_set


def visualizer(geometries):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for geo in geometries:
        if geo["name"] == "points_cloud":
            if "color" in geo:
                pcd = o3d.geometry.PointCloud(
                    convert_numpy_to_open3d(geo["data"], color=geo["color"])
                )
            else:
                pcd = o3d.geometry.PointCloud(convert_numpy_to_open3d(geo["data"]))
            vis.add_geometry(pcd)
        elif geo["name"] == "camera":
            cam_pose = geo["data"]
            line_set, coordinate_set = draw_camera(cam_pose, color=geo["color"])
            vis.add_geometry(line_set)
            vis.add_geometry(coordinate_set)
        elif geo["name"] == "mesh":
            vis.add_geometry(geo["data"])
    vis.add_geometry(
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    )
    vis.run()
    vis.destroy_window()