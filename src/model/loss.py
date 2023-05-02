from torch import nn
import torch
from pytorch3d.transforms import so3_relative_angle
from src.poses.utils import load_rotation_transform, convert_openCV_to_openGL_torch
import torch.nn.functional as F

# so3_relative_angle is a function from pytorch3d library but it does not support symmetry objects
# this function is a modified version of so3_relative_angle to support symmetry objects
# symmetry == 0: no symmetry, 1: two symmetry, 2: circle symmetry

roty180 = load_rotation_transform("y", 180)[:3, :3].float()


def so3_relative_angle_with_symmetry(pred, gt, symmetry):
    # This function do not handle inplane rotation
    # handle symmetry by seperating the objects into 3 groups: no symmetry, two symmetry, circle symmetry

    # non symmetry
    idx_non_symmetry = symmetry == 0
    error_non_symmetry = so3_relative_angle(
        pred[idx_non_symmetry], gt[idx_non_symmetry], eps=1e-2
    )
    if torch.sum(idx_non_symmetry) == pred.shape[0]:
        return error_non_symmetry
    else:  # if there exists some symmetry objects
        error = torch.zeros(pred.shape[0], device=pred.device, dtype=pred.dtype)
        error[idx_non_symmetry] = error_non_symmetry

        # symmetry with 180 degrees around Y axis: take the minimum of R and Ry180.dot(R)
        idx_two_symmetry = symmetry == 1
        error[idx_two_symmetry] = so3_relative_angle(
            pred[idx_two_symmetry], gt[idx_two_symmetry], eps=1e-2
        )

        # rotate pred by 180 degrees around Y axis and compare again to take the minimum
        roty180_batch = (
            roty180.unsqueeze(0)
            .repeat(error[idx_two_symmetry].shape[0], 1, 1)
            .to(pred.device)
        )
        pred_two_symmetry_rotated = torch.bmm(
            roty180_batch, pred[idx_two_symmetry].float()
        ).float()
        tmp = so3_relative_angle(
            pred_two_symmetry_rotated.to(torch.float64),
            gt[idx_two_symmetry].to(torch.float64),
            eps=1e-2,
        )
        error[idx_two_symmetry] = torch.minimum(error[idx_two_symmetry], tmp)

        if torch.sum(idx_non_symmetry) + torch.sum(idx_two_symmetry) == pred.shape[0]:
            return error
        else:
            idx_circle_symmetry = symmetry == 2
            pred_circle_openGL = pred[idx_circle_symmetry].clone()
            gt_circle_openGL = gt[idx_circle_symmetry].clone()
            # the trick here is to convert object pose to camera pose and then convert to openGL coordinate system
            pred_cam = pred_circle_openGL[:, :3, :3].inverse()
            gt_cam = gt_circle_openGL[:, :3, :3].inverse()
            pred_circle_openGL = convert_openCV_to_openGL_torch(pred_cam)
            gt_circle_openGL = convert_openCV_to_openGL_torch(gt_cam)
            # to deal with circle symmetry, we convert pose to openGL coordinate system
            pred_circle_openGL[:, :2, :2] = gt_circle_openGL[:, :2, :2]
            cosine_sim_sym = F.cosine_similarity(
                pred_circle_openGL[:, 2, :3],
                gt_circle_openGL[
                    :, 2, :3
                ],  # since we do not consider in-plane rotation, we only compare Z axis
            )
            error[idx_circle_symmetry] = torch.acos(cosine_sim_sym)
        return error


class GeodesicError(nn.Module):
    # credit https://github.com/martius-lab/beta-nll
    def __init__(self, thresholds=[15]):
        super(GeodesicError, self).__init__()
        self.thresholds = thresholds

    def forward(self, predR, gtR, symmetry):
        if len(predR.shape) == 3:  # top 1 Bx3x3
            error = so3_relative_angle_with_symmetry(
                predR.to(torch.float64),
                gtR.to(torch.float64),
                symmetry,
            )
            error = torch.rad2deg(error)
            results = {
                f"top1, accuracy_{self.thresholds[i]}": (error <= self.thresholds[i])
                .float()
                .mean()
                * 100
                for i in range(len(self.thresholds))
            }
            results["top1, median"] = error.median()
            return error, results
        else:  # top k Bxkx3x3
            results = {}
            # assert pred_matrix.shape[1] == len(self.ks), print("Only work with top1, top3, top5")
            errors = torch.zeros((predR.shape[0], predR.shape[1]), device=predR.device)
            for idx_k in range(predR.shape[1]):
                errors[:, idx_k] = so3_relative_angle_with_symmetry(
                    predR[:, idx_k].to(torch.float64),
                    gtR.to(torch.float64),
                    symmetry,
                )
                errors[:, idx_k] = torch.rad2deg(errors[:, idx_k])
                if idx_k in [0, 2, 4]:
                    top_error = torch.min(errors[:, : idx_k + 1], dim=1).values
                    for i in range(len(self.thresholds)):
                        results[f"top{idx_k+1}, accuracy_{self.thresholds[i]}"] = (
                            top_error <= self.thresholds[i]
                        ).float().mean() * 100
                        results[f"top{idx_k+1}, median"] = top_error.median()
            return errors[:, 0], results


if __name__ == "__main__":
    from src.poses.utils import get_obj_poses_from_template_level

    template_poses = get_obj_poses_from_template_level(
        level=0, pose_distribution="upper"
    )
    template_poses = torch.from_numpy(template_poses).cuda()
    loss_func = GeodesicError()
    for _ in range(10):
        feat_query5d = torch.randn(8, 26, 4, 32, 32).cuda()
        pred_feat_templates = torch.randn(8, 26, 4, 32, 32).cuda()
        distance = (feat_query5d - pred_feat_templates) ** 2
        distance = torch.norm(distance, dim=2)
        similarity = -distance.sum(axis=3).sum(axis=2)  # B x N
        inv_distance, pred_index = similarity.topk(k=5, dim=1)  # B x 1
        
        pred_poses = template_poses[pred_index]
        gt_poses = template_poses[range(len(pred_poses))]
        for symmetry in [0]:
            print("index of symmetry", symmetry)
            if symmetry == 0:
                sym = torch.zeros(8).cuda()
            else:
                sym = torch.ones(8).cuda() * symmetry
            err, acc = loss_func(pred_poses[:, :, :3, :3], gt_poses[:, :3, :3], sym)
            print(err)
            print(acc)