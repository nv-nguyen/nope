import numpy as np


class Precomputer(object):
    # credit :  https://github.com/thodan/bop_toolkit/blob/master/bop_toolkit_lib/misc.py#L110
    """
    Caches pre_Xs, pre_Ys for a 30% speedup of depth_im_to_dist_im()
    """
    xs, ys = None, None
    pre_Xs, pre_Ys = None, None
    depth_im_shape = None
    K = None

    @staticmethod
    def precompute_lazy(depth_im, K):
        """Lazy precomputation for depth_im_to_dist_im() if depth_im.shape or K changes
        :param depth_im: hxw ndarray with the input depth image, where depth_im[y, x]
          is the Z coordinate of the 3D point [X, Y, Z] that projects to pixel [x, y],
          or 0 if there is no such 3D point (this is a typical output of the
          Kinect-like sensors).
        :param K: 3x3 ndarray with an intrinsic camera matrix.
        :return: hxw ndarray (Xs/depth_im, Ys/depth_im)
        """
        if depth_im.shape != Precomputer.depth_im_shape:
            Precomputer.depth_im_shape = depth_im.shape
            Precomputer.xs, Precomputer.ys = np.meshgrid(
                np.arange(depth_im.shape[1]), np.arange(depth_im.shape[0])
            )

        if depth_im.shape != Precomputer.depth_im_shape or not np.all(
            K == Precomputer.K
        ):
            Precomputer.K = K
            Precomputer.pre_Xs = (Precomputer.xs - K[0, 2]) / np.float64(K[0, 0])
            Precomputer.pre_Ys = (Precomputer.ys - K[1, 2]) / np.float64(K[1, 1])

        return Precomputer.pre_Xs, Precomputer.pre_Ys


def depth_im_to_dist_im_fast(depth_im, K):
    # credit: https://github.com/thodan/bop_toolkit/blob/529f11135315cffd536d7b5ea44bcf326daa9a6a/
    # bop_toolkit_lib/misc.py#L143
    """Converts a depth image to a distance image.
    :param depth_im: hxw ndarray with the input depth image, where depth_im[y, x]
      is the Z coordinate of the 3D point [X, Y, Z] that projects to pixel [x, y],
      or 0 if there is no such 3D point (this is a typical output of the
      Kinect-like sensors).
    :param K: 3x3 ndarray with an intrinsic camera matrix.
    :return: hxw ndarray with the distance image, where dist_im[y, x] is the
      distance from the camera center to the 3D point [X, Y, Z] that projects to
      pixel [x, y], or 0 if there is no such 3D point.
    """
    # Only recomputed if depth_im.shape or K changes.
    pre_Xs, pre_Ys = Precomputer.precompute_lazy(depth_im, K)

    dist_im = np.sqrt(
        np.multiply(pre_Xs, depth_im) ** 2
        + np.multiply(pre_Ys, depth_im) ** 2
        + depth_im.astype(np.float64) ** 2
    )

    return dist_im


def _estimate_visib_mask(d_test, d_model, delta, visib_mode="bop19"):
    # credit: https://github.com/thodan/bop_toolkit/blob/529f11135315cffd536d7b5ea44bcf326daa9a6a/
    # bop_toolkit_lib/visibility.py#L9
    """Estimates a mask of the visible object surface.
    :param d_test: Distance image of a scene in which the visibility is estimated.
    :param d_model: Rendered distance image of the object model.
    :param delta: Tolerance used in the visibility test.
    :param visib_mode: Visibility mode:
    1) 'bop18' - Object is considered NOT VISIBLE at pixels with missing depth.
    2) 'bop19' - Object is considered VISIBLE at pixels with missing depth. This
         allows to use the VSD pose error function also on shiny objects, which
         are typically not captured well by the depth sensors. A possible problem
         with this mode is that some invisible parts can be considered visible.
         However, the shadows of missing depth measurements, where this problem is
         expected to appear and which are often present at depth discontinuities,
         are typically relatively narrow and therefore this problem is less
         significant.
    :return: Visibility mask.
    """
    assert d_test.shape == d_model.shape

    if visib_mode == "bop18":
        mask_valid = np.logical_and(d_test > 0, d_model > 0)
        d_diff = d_model.astype(np.float32) - d_test.astype(np.float32)
        visib_mask = np.logical_and(d_diff <= delta, mask_valid)

    elif visib_mode == "bop19":
        d_diff = d_model.astype(np.float32) - d_test.astype(np.float32)
        visib_mask = np.logical_and(
            np.logical_or(d_diff <= delta, d_test == 0), d_model > 0
        )

    else:
        raise ValueError("Unknown visibility mode.")

    return visib_mask


def estimate_visib_mask_gt(d_test, d_gt, delta, visib_mode="bop19"):
    # credit: https://github.com/thodan/bop_toolkit/blob/529f11135315cffd536d7b5ea44bcf326daa9a6a/
    # bop_toolkit_lib/visibility.py#L45
    """Estimates a mask of the visible object surface in the ground-truth pose.
    :param d_test: Distance image of a scene in which the visibility is estimated.
    :param d_gt: Rendered distance image of the object model in the GT pose.
    :param delta: Tolerance used in the visibility test.
    :param visib_mode: See _estimate_visib_mask.
    :return: Visibility mask.
    """
    visib_gt = _estimate_visib_mask(d_test, d_gt, delta, visib_mode)
    return visib_gt


def estimate_visib_mask_est(d_test, d_est, visib_gt, delta, visib_mode="bop19"):
    # credit: https://github.com/thodan/bop_toolkit/blob/529f11135315cffd536d7b5ea44bcf326daa9a6a/
    # bop_toolkit_lib/visibility.py#L58
    """Estimates a mask of the visible object surface in the estimated pose.
    For an explanation of why the visibility mask is calculated differently for
    the estimated and the ground-truth pose, see equation (14) and related text in
    Hodan et al., On Evaluation of 6D Object Pose Estimation, ECCVW'16.
    :param d_test: Distance image of a scene in which the visibility is estimated.
    :param d_est: Rendered distance image of the object model in the est. pose.
    :param visib_gt: Visibility mask of the object model in the GT pose (from
      function estimate_visib_mask_gt).
    :param delta: Tolerance used in the visibility test.
    :param visib_mode: See _estimate_visib_mask.
    :return: Visibility mask.
    """
    visib_est = _estimate_visib_mask(d_test, d_est, delta, visib_mode)
    visib_est = np.logical_or(visib_est, np.logical_and(visib_gt, d_est > 0))
    return visib_est
