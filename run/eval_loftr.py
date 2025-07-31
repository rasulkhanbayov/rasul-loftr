import sys
import os
import json
import argparse
import datetime
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
import numpy as np
import logging
import time

sys.path.append(os.path.join(os.getcwd(), 'lib'))

sys.path.insert(0, os.path.join(os.getcwd(), 'baselines', 'GeoFormer'))
sys.path.insert(0, os.path.join(os.getcwd(), 'baselines', 'SuperRetina'))
from dataset import OCTDataset
from visualization.images import save_image, plot_correspondences, visualize_canvas, create_video_from_images
from models import LoFTRPretrained, LightGluePretrained, GeoFormerPretrained, SuperRetinaPretrained, SIFT
from scipy.ndimage import label

def estimate_rigid_transformation(data_dict, device, conf_threshold):
    kp1, kp2, confidences = data_dict['keypoints0'], data_dict['keypoints1'], data_dict['confidence']

    valid_mask = confidences > conf_threshold
    kp1, kp2 = kp1[valid_mask], kp2[valid_mask]

    kp1 = kp1.cpu().numpy()
    kp2 = kp2.cpu().numpy()

    # model_robust, inliers = ransac((kp2, kp1), EuclideanTransform, min_samples=2, residual_threshold=1, max_trials=2048, rng=233)
    # H = model_robust.params
    # H, mask = pydegensac.findHomography(kp2, kp1, 3.0)

    # Use cv2 to estimate the homography
    # H, mask = cv2.findHomography(kp2, kp1, cv2.RANSAC, 3.0)
    # H, mask = cv2.estimateAffine2D(kp2, kp1, None, cv2.RANSAC, 1.0)

    H, mask = cv2.estimateAffinePartial2D(kp2, kp1, None, cv2.RANSAC, 1.0)
    # H = remove_scaling_shearing(H)
    H = np.concatenate([H, np.array([[0, 0, 1]])], axis=0)
    
    data_dict['pred_H_1to0'] = torch.tensor(np.asarray(H), dtype=torch.float32).to(device)

def calculate_mean_corner_error(corners_pred, corners_gt):
    errors = np.linalg.norm(corners_pred - corners_gt, axis=1)
    mean_error = np.mean(errors)
    return mean_error

def calculate_iou(mask1, mask2):
    if mask1.sum() == 0:
        return 1.0
    intersection = np.logical_and(mask1[0].astype('uint8'), mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def extract_largest_cluster_optimized(segmentation_mask):
    """
    Optimized method to extract the largest cluster from a segmentation mask after applying a median filter.
    
    Parameters:
    - segmentation_mask: A 2D numpy array of shape (H, W) representing the segmentation mask.
    
    Returns:
    - A 2D numpy array of the same shape as `segmentation_mask` with only the largest cluster retained.
    """
    
    # Assuming the median filter is applied outside this function or is no longer needed.
    
    # Label all unique clusters in the mask
    labeled_mask, num_features = label(segmentation_mask)
    if num_features == 0:
        return segmentation_mask
    
    # Use np.bincount to count the size of each cluster efficiently
    # The zeroth bin counts the background, so we skip it using [1:]
    sizes = np.bincount(labeled_mask.flat)[1:]
    
    # Find the label of the largest cluster, adding 1 since we skipped the first bin
    largest_cluster = np.argmax(sizes) + 1
    
    # Create a mask for the largest cluster
    largest_cluster_mask = (labeled_mask == largest_cluster).astype(int)
    
    return largest_cluster_mask

def postprocess_segmentation_mask(data_dict):
    seg_mask = data_dict['seg_mask']
    # softmax
    seg_mask = torch.nn.functional.softmax(seg_mask, dim=1)
    seg_mask = seg_mask.cpu().numpy()
    seg_mask = seg_mask[0]
    # seg_mask = seg_mask.argmax(axis=0)
    # threshold at 0.8
    seg_mask = (seg_mask[1] + 0.0 * data_dict['otsu_mask'][0][0].cpu().numpy()) > 0.1
    seg_mask = seg_mask.astype(np.uint8)
    seg_mask = cv2.medianBlur(seg_mask, 13)
    # seg_mask = cv2.dilate(seg_mask, np.ones((3, 3), np.uint8), iterations=1)
    # seg_mask = cv2.morphologyEx(seg_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)

    # seg_mask = extract_largest_cluster_optimized(seg_mask)
    
    return seg_mask

def initialize_model(model_choice, model_name, device):
    return LoFTRPretrained(device, model_name)
    # return LoFTRPretrained(device, "last.ckpt")
    # return LoFTRPretrained(device, "/home/guests/tony_wang/Documents/hiwi/loftr/logs/tb_logs/outdoor-ds-64/version_6/checkpoints/epoch=7-auc@5=0.246-auc@10=0.412-auc@20=0.557.ckpt")
    return LoFTRPretrained(device, "outdoor")
    # return LoFTRPretrained(device, "indoor")


def create_output_directory(config, take, model_name):
    output_dir = os.path.join(
        "output",
        take,
        f"{datetime.datetime.now().strftime('%m_%d-%H_%M_%S')}_{config.experiment_name.upper()}_{model_name.upper()}"
    )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def evaluate(dataloader, model, device, config, take="eval", visualize=True):
    model.eval()
    mean_corner_errors = []
    ious = []
    json_content = {take: {}}

    os.makedirs(config.output_dir, exist_ok=True)

    for it, data_dict in enumerate(dataloader):
        data_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data_dict.items()}

        model(data_dict)

        if visualize:
            plot_correspondences(data_dict)

        estimate_rigid_transformation(data_dict, device, conf_threshold=config.threshold)

        # Apply predicted H to corners of source image
        original_corners = np.array([
            [0, 0],
            [255, 0],
            [255, 255],
            [0, 255]
        ], dtype=np.float32)

        ones = np.ones((4, 1))
        original_corners_hom = np.hstack([original_corners, ones])
        H = data_dict["first_img_pred_H_1to0"].cpu().numpy()
        transformed_corners_hom = original_corners_hom @ H.T
        transformed_corners = transformed_corners_hom[:, :2] / transformed_corners_hom[:, 2:]

        # Assume gt_img_corners available in data_dict
        gt_corners = data_dict["gt_img_corners"].cpu().numpy()
        error = calculate_mean_corner_error(transformed_corners, gt_corners)
        mean_corner_errors.append(error)

        # IOU computation
        seg_mask = postprocess_segmentation_mask(data_dict)
        data_dict['seg_mask_np'] = seg_mask
        iou = calculate_iou(data_dict['source_image_seg'].cpu().numpy(), seg_mask)
        ious.append(iou)

        # Save overlayed segmentation result
        original_img = data_dict['source_image'].cpu().numpy()[0]
        original_img = (original_img * 255).astype(np.uint8)
        seg_mask_color = (seg_mask * 255).astype(np.uint8)
        seg_mask_color = cv2.cvtColor(seg_mask_color, cv2.COLOR_GRAY2BGR)
        original_img_color = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(original_img_color, 0.5, seg_mask_color, 0.5, 0)
        stacked = np.concatenate((original_img_color, overlay), axis=1)
        cv2.imwrite(os.path.join(config.output_dir, f"{data_dict['id'].item():06}_seg.jpg"), stacked)

        visualize_canvas(data_dict, error)

        json_content[take][f"iter_{it}"] = {
            "pred_H": data_dict['first_img_pred_H_1to0'].tolist(),
            "corner_error": float(error),
            "iou": float(iou)
        }

    # Save metrics to JSON
    json_path = os.path.join(config.output_dir, f"{take}_results.json")
    with open(json_path, "w") as f:
        json.dump(json_content, f, indent=4)

    print(f"Saved evaluation metrics to {json_path}")
    print(f"Mean Corner Error: {np.mean(mean_corner_errors):.4f}")
    print(f"Mean IOU: {np.mean(ious):.4f}")


def main(config, takes, model_choice="loftr", visualize=False):
    # Set device to cuda, mps or cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu" if torch.backends.mps.is_available() else "cpu")

    model_names = list(sorted(os.listdir("loftr_models_new"), key=lambda x: int(x.split('-')[0].split('=')[1])))
    # model_names = []
    model_names.insert(0, "outdoor")

    # Set up logging
    log_filename = os.path.join("logs_1.log")
    logging.basicConfig(level=logging.INFO, filename=log_filename,
                        filemode='w', format='%(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    # model_names = ["outdoor"]
    # model_names = ["finetuned.ckpt"]
    # model_names = ["last.ckpt"]
    # model_names = model_names[13:]
    model_names = ["epoch=46-auc@5=0.153-auc@10=0.339-auc@20=0.529.ckpt"]

    for i, model_name in enumerate(model_names):
        model = initialize_model(model_choice, os.path.join("loftr_models_new", model_name) if model_name != "outdoor" else model_name, device)

        if not isinstance(takes, list):
            takes = [takes]

        take_errors, iou_errors = evaluate(model_name, config, takes, model, device, visualize=visualize)
        avg_error = sum(take_errors) / len(take_errors)
        avg_iou = sum(iou_errors) / len(iou_errors)

        logging.info(f"Model: {model_name}, Average corner error: {avg_error:.4f}, Average IOU: {avg_iou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run image stitching with a PreTrained and OCTDataset.")
    parser.add_argument("--canvas_size", nargs=2, type=int, default=[1000, 1000], help="Size of the canvas as two integers (height width).")
    parser.add_argument("--threshold", type=float, default=0.0, help="Threshold for the key-points.")
    parser.add_argument("--target_image", type=str, choices=['single', 'merged'], default="merged", help="Specify whether to use a single or merged target image.")
    parser.add_argument("--experiment_name", type=str, default="", help="Name of the experiment.")
    parser.add_argument("--model", type=str, default='loftr', choices=['loftr', 'lightglue', 'geoformer', 'superretina'], help="Choose between 'loftr' or 'lightglue' model.")
    parser.add_argument("--data_in_use", type=str, default="percentile_90", help="Specify the data in use.")

    args = parser.parse_args()

    # for take in ['Take_01', 'Take_02', 'Take_03', 'Take_04', 'Take_05', 'Take_06', 'Take_07', 'Take_08', 'Take_09']:
    # for take in ['Take_02', 'Take_03', 'Take_04', 'Take_05', 'Take_06']:
    # for take in ['Take_02']:
    #     args.take = take
    # main(args, takes=[f"Take_0{i}" for i in range(1, 7)], model_choice=args.model, visualize=True)
    main(args, takes=[f"Take_0{i}" for i in range(1, 7)], model_choice=args.model, visualize=False)
    
   
