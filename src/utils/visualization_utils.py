import os
import torch.nn.functional as F

os.environ["MPLCONFIGDIR"] = os.getcwd() + "./tmp/"
import matplotlib.pyplot as plt
import matplotlib
from torchvision import transforms, utils
import torch
from PIL import Image
import numpy as np
import io
from moviepy.video.io.bindings import mplfig_to_npimage
import cv2


inverse_normalize = transforms.Compose(
    [
        transforms.Normalize(
            mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        ),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
    ]
)


def PIL_image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def get_cmap(np_img):
    cmap = matplotlib.colormaps.get_cmap("magma")
    tmp = cmap(np_img)[..., :3]
    return tmp


def put_image_to_grid(list_imgs, adding_margin=True):
    num_col = len(list_imgs)
    b, c, h, w = list_imgs[0].shape
    device = list_imgs[0].device
    if adding_margin:
        num_all_col = num_col + 1
    else:
        num_all_col = num_col
    grid = torch.zeros((b * num_all_col, 3, h, w), device=device).to(torch.float16)
    idx_grid = torch.arange(0, grid.shape[0], num_all_col, device=device).to(
        torch.int64
    )
    for i in range(num_col):
        grid[idx_grid + i] = list_imgs[i].to(torch.float16)
    return grid, num_col + 1


def draw_grid_text(images, texts, save_path, dpi=50):
    B, N = images.shape[:2]
    plt.figure(figsize=(5 * N, 5 * B))
    for b in range(B):
        for n in range(N):
            if n != 2:
                plt.subplot(B, N, b * N + n + 1)
                plt.imshow(images[b, n])
                plt.axis("off")
                if n == 0:
                    plt.title("Query", fontsize=20)
                elif n == 1:
                    plt.title("Reference", fontsize=20)
                else:
                    # put title from the third image
                    plt.title(f"Top {n-2}: {texts[b, n-3]:.03f}", fontsize=30)
    plt.subplots_adjust(wspace=0.1, hspace=0.15)
    plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close("all")


def convert_cmap(tensor, vmin=None, vmax=None):
    b, h, w = tensor.shape
    ndarr = tensor.to("cpu", torch.uint8).numpy()
    output = torch.zeros((b, 3, h, w), device=tensor.device)
    for i in range(b):
        cmap = matplotlib.cm.get_cmap("magma")
        tmp = cmap(ndarr[i])[..., :3]
        data = transforms.ToTensor()(np.array(tmp)).to(tensor.device)
        output[i] = data
    return output


def convert_cmap_slow(tensor, vmin=None, vmax=None):
    b, h, w = tensor.shape
    ndarr = tensor.to("cpu", torch.uint8).numpy()
    output = torch.zeros((b, 3, h, w), device=tensor.device)
    for i in range(b):
        # convert img to cmap of shape (h, w, 3)
        fig = plt.figure(figsize=(5, 5), dpi=50)
        plt.imshow(ndarr[i], cmap="magma", vmin=vmin, vmax=vmax)

        plt.axis("off")
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        data = mplfig_to_npimage(fig)
        data = cv2.resize(data, (h, w), interpolation=cv2.INTER_AREA)
        plt.close("all")

        data = transforms.ToTensor()(np.array(data)).to(tensor.device)
        output[i] = data
    return output


def visualize_uncertainty(uncertainty, img_size, apply_cmap):
    uncertainty = F.interpolate(
        uncertainty, img_size, mode="bilinear", align_corners=False
    )
    uncertainty = torch.norm(uncertainty, dim=1)
    if apply_cmap:
        uncertainty = convert_cmap(uncertainty)
    else:
        uncertainty = uncertainty.unsqueeze(1).repeat(1, 3, 1, 1)
    return uncertainty.to(torch.float16)


def write_text(
    img_path,
    errors_wo_uncertainty,
    errors,
    sample_size=128,
    color=(255, 0, 0),
    font_scale=0.5,
    thickness=1,
    idx_cols=[1],
    gap_between_text=2,
    text_predix="err",
    addtional_info=None,
    addtional_info2=None,
):
    img = Image.open(img_path)
    img_size = np.array(img.size)
    ncol, nrow = int(img_size[0] / sample_size), int(img_size[1] / sample_size)
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_with_text = np.array(img).copy()

    idx_sample = 0
    for idx_row in range(nrow):
        for idx_col in idx_cols:
            # write text for error
            pos = (
                int((idx_col + 0.35) * sample_size),
                int((idx_row + 0.95) * sample_size),
            )
            caption = f"{text_predix}={errors_wo_uncertainty[idx_sample]:.01f}"
            if addtional_info is not None:
                caption += f", err={addtional_info[idx_sample]:.01f}"
            img_with_text = cv2.putText(
                img_with_text,
                caption,
                pos,
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
            if errors is not None:
                # write text for error
                pos = (
                    int((idx_col + gap_between_text + 0.25) * sample_size),
                    int((idx_row + 0.95) * sample_size),
                )
                caption = f"{text_predix}={errors[idx_sample]:.01f}"
                if addtional_info is not None:
                    caption += f", err={addtional_info2[idx_sample]:.01f}"
                img_with_text = cv2.putText(
                    img_with_text,
                    caption,
                    pos,
                    font,
                    font_scale,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
            idx_sample += 1

    img_with_text_PIL = Image.fromarray(img_with_text)
    img_with_text_PIL.save(img_path)
    return img_with_text


def write_text_diffusion(
    img_path,
    errors,
    sample_size=128,
    color=(255, 0, 0),
    font_scale=0.5,
    thickness=1,
):
    img = Image.open(img_path)
    img_size = np.array(img.size)
    ncol, nrow = int(img_size[0] / sample_size), int(img_size[1] / sample_size)
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_with_text = np.array(img).copy()

    idx_sample = 0
    for idx_row in range(nrow):
        for idx_col in [1]:
            # write text for error_wo_uncertainty
            pos = (
                int((idx_col + 0.35) * sample_size),
                int((idx_row + 0.95) * sample_size),
            )
            img_with_text = cv2.putText(
                img_with_text,
                f"err={errors[idx_sample]:.01f}",
                pos,
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
            idx_sample += 1
    img_with_text_PIL = Image.fromarray(img_with_text)
    img_with_text_PIL.save(img_path)
    return img_with_text


def write_text_similarity(
    img_path,
    similarity,
    sample_size=256,
    color=(255, 0, 0),
    font_scale=0.5,
    thickness=1,
):
    img = Image.open(img_path)
    img_size = np.array(img.size)
    ncol, nrow = int(img_size[0] / sample_size), int(img_size[1] / sample_size)
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_with_text = np.array(img).copy()

    for idx_row in range(nrow):
        for idx_col in range(ncol - 2):
            # write text for error_wo_uncertainty
            pos = (
                int((idx_col + 2 + 0.35) * sample_size),
                int((idx_row + 0.95) * sample_size),
            )
            img_with_text = cv2.putText(
                img_with_text,
                f"similarity={similarity[idx_row, idx_col]:.02f}",
                pos,
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
    img_with_text_PIL = Image.fromarray(img_with_text)
    img_with_text_PIL.save(img_path)
    return img_with_text


def draw_pose_axis(cvImg, matrix4x4, intrinsics, radius, thickness):
    R, T = matrix4x4[:3, :3], np.asarray(matrix4x4[:3, 3]).reshape(3, -1)
    aPts = np.array([[0, 0, 0], [0, 0, radius], [0, radius, 0], [radius, 0, 0]])
    rep = np.matmul(intrinsics, np.matmul(R, aPts.T) + T)
    x = np.int32(rep[0] / rep[2] + 0.5)
    y = np.int32(rep[1] / rep[2] + 0.5)
    cvImg = cv2.line(
        cvImg,
        (x[0], y[0]),
        (x[1], y[1]),
        (0, 0, 255),
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    cvImg = cv2.line(
        cvImg,
        (x[0], y[0]),
        (x[2], y[2]),
        (0, 255, 0),
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    cvImg = cv2.line(
        cvImg,
        (x[0], y[0]),
        (x[3], y[3]),
        (255, 0, 0),
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    return cvImg


def project_point_cloud(cvImg, obj_pcd, matrix4x4, intrinsic):
    pts = np.matmul(
        intrinsic,
        np.matmul(matrix4x4[:3, :3], obj_pcd.T) + matrix4x4[:3, 3].reshape(-1, 1),
    )
    xs = pts[0] / (pts[2] + 1e-8)
    ys = pts[1] / (pts[2] + 1e-8)

    for pIdx in range(len(xs)):
        cvImg = cv2.circle(
            cvImg,
            (int(xs[pIdx]), int(ys[pIdx])),
            2,
            (255, 0, 0),
            -1,
        )
    return cvImg


def plot_imgs(imgs, save_path):
    num_imgs = len(imgs)
    plt.figure(figsize=(num_imgs * 5, 5))
    for idx, img in enumerate(imgs):
        plt.subplot(1, num_imgs, idx + 1)
        plt.imshow(img)
        plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)


def resize_tensor(tensor, size):
    return F.interpolate(tensor, size, mode="bilinear", align_corners=True)


if __name__ == "__main__":
    import numpy as np

    img_path = (
        "/home/nguyen/Documents/results/debug_train_obj/media/prediction_rank1.png"
    )
    similarity = np.random.rand(7, 7)
    cmap_sim = get_cmap(similarity)
    print(cmap_sim.shape)
    # img_path_new = write_text_similarity(img_path, similarity)
    # print(img_path)
