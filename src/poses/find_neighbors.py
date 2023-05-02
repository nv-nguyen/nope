import numpy as np
# import open3d as o3d
import os
from src.poses.utils import (
    get_obj_poses_from_template_level,
    get_root_project,
    load_index_level0_in_level2,
    NearestTemplateFinder,
)
import os.path as osp
# from src.utils.vis_3d_utils import convert_numpy_to_open3d, draw_camera

if __name__ == "__main__":
    templates_poses_level0 = get_obj_poses_from_template_level(
        0, "all", return_cam=True
    )
    templates_poses_level2 = get_obj_poses_from_template_level(
        2, "upper", return_cam=True
    )
    finder = NearestTemplateFinder(
        level_templates=2,
        pose_distribution="all",
        return_inplane=True,
    )
    obj_poses_level0 = get_obj_poses_from_template_level(0, "all", return_cam=False)
    obj_poses_level2 = get_obj_poses_from_template_level(2, "all", return_cam=False)
    idx_templates, inplanes = finder.search_nearest_template(obj_poses_level0)
    print(len(obj_poses_level0), len(idx_templates))
    root_repo = get_root_project()
    save_path = os.path.join(root_repo, "src/poses/predefined_poses/idx_all_level0_in_level2.npy")
    np.save(save_path, idx_templates)
