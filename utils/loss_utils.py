#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np


def surface_loss(mask_image, gt_mask, debug_id=None):
    """
    Compute intra-mask consistency loss and inter-mask separation loss for N-dimensional features.
    Uses cosine similarity to match the PCD export method.

    For each instance label in gt_mask (excluding 0 if present):
    1. Compute intra-mask consistency: minimize cosine distance within each mask region
    2. Compute inter-mask separation: maximize cosine distance between different mask regions
    Lower loss encourages similar features within each mask region and different features between regions.

    Args:
        mask_image (Tensor): shape (N, H, W) - N-dimensional feature vectors per pixel
        gt_mask (Tensor): shape (1, H, W) with integer labels for different masks

    Returns:
        Tensor: scalar loss
    """
    # Ensure mask_image is (N, H, W)
    if mask_image.dim() == 2:
        # (H, W) -> (1, H, W)
        features = mask_image.unsqueeze(0)
    elif mask_image.dim() == 3:
        features = mask_image  # Already (N, H, W)
    else:
        raise ValueError(f"Expected mask_image shape (N, H, W), got {mask_image.shape}")

    # Ensure gt_mask is (1, H, W)
    if gt_mask.dim() == 2:
        labels_map = gt_mask.unsqueeze(0)  # (H, W) -> (1, H, W)
    elif gt_mask.dim() == 3 and gt_mask.size(0) == 1:
        labels_map = gt_mask  # Already (1, H, W)
    else:
        raise ValueError(f"Expected gt_mask shape (1, H, W), got {gt_mask.shape}")

    # Ensure same device
    labels_map = labels_map.to(device=features.device)

    # Safety check: ensure spatial dims match
    if features.size(-2) != labels_map.size(-2) or features.size(-1) != labels_map.size(-1):
        raise ValueError("mask_image and gt_mask spatial dimensions must match")

    # Squeeze to (H, W) for easier processing
    labels_2d = labels_map.squeeze(0)  # (1, H, W) -> (H, W)
    N, H, W = features.shape

    # Normalize features for cosine similarity computation
    features_normalized = torch.nn.functional.normalize(features, dim=0)  # (N, H, W)

    # Get unique instance labels, ignore background label 0 if present
    unique_labels = torch.unique(labels_2d)
    if (unique_labels == 0).any():
        unique_labels = unique_labels[unique_labels != 0]

    # If no valid labels, return zero (preserve graph dtype/device)
    if unique_labels.numel() == 0:
        return features.mean() * 0.0

    # Collect valid masks and their mean feature vectors
    valid_masks = []
    mask_means = []
    
    for label_value in unique_labels:
        mask = (labels_2d == label_value)  # (H, W)
        num_pixels = mask.sum()
        
        # Filter out masks that are too small or too large
        if num_pixels <= 100 or num_pixels > 10000:
            continue
        
        # Check if mask contains the bottom-right corner (unreasonable mask)
        bottom_right_corner = mask[H-1, W-1]  # bottom-right corner
        if bottom_right_corner:
            continue

        # Get feature vectors within mask: shape (num_pixels, N)
        mask_indices = mask.nonzero(as_tuple=False)  # (num_pixels, 2)
        if mask_indices.size(0) == 0:
            continue
            
        # Extract features for this mask: (num_pixels, N)
        mask_features = features_normalized[:, mask_indices[:, 0], mask_indices[:, 1]].T  # (num_pixels, N)
        mean_features = mask_features.mean(dim=0)  # (N,)
        mean_features = torch.nn.functional.normalize(mean_features, dim=0)  # Normalize mean
        
        valid_masks.append(mask)
        mask_means.append(mean_features)

    if len(valid_masks) == 0:
        return features.mean() * 0.0

    # 1. Intra-mask consistency loss (minimize cosine distance within each mask)
    intra_losses = []
    for i, mask in enumerate(valid_masks):
        mask_indices = mask.nonzero(as_tuple=False)
        if mask_indices.size(0) == 0:
            continue
            
        # Extract features for this mask: (num_pixels, N)
        mask_features = features_normalized[:, mask_indices[:, 0], mask_indices[:, 1]].T  # (num_pixels, N)
        mean_features = mask_means[i]  # (N,)
        
        # Compute cosine similarity with mean for each pixel
        # Cosine similarity = dot product for normalized vectors
        similarities = torch.mm(mask_features, mean_features.unsqueeze(1)).squeeze(1)  # (num_pixels,)
        
        # Convert to cosine distance: 1 - cosine_similarity
        cosine_distances = 1.0 - similarities
        intra_loss = cosine_distances.mean()  # Average cosine distance from mean
        intra_losses.append(intra_loss)
    
    if len(intra_losses) == 0:
        return features.mean() * 0.0
        
    intra_loss_total = torch.stack(intra_losses).mean()

    # 2. Inter-mask separation loss (maximize cosine distance between different masks)
    if len(mask_means) > 1:
        mask_means_tensor = torch.stack(mask_means)  # (num_masks, N)
        
        # Compute pairwise cosine similarities between mask means
        # Shape: (num_masks, num_masks)
        pairwise_similarities = torch.mm(mask_means_tensor, mask_means_tensor.T)
        
        # Remove diagonal (self-comparison) and get upper triangle
        mask_upper = torch.triu(torch.ones_like(pairwise_similarities), diagonal=1).bool()
        valid_similarities = pairwise_similarities[mask_upper]
        
        if valid_similarities.numel() > 0:
            # Inter-mask loss: minimize the maximum similarity (maximize minimum distance)
            # This encourages maximum separation between different masks
            max_similarity = valid_similarities.max()
            inter_loss = max_similarity  # Penalty for high similarity between different masks
        else:
            inter_loss = features.new_zeros(())
    else:
        inter_loss = features.new_zeros(())

    # Combine intra-mask consistency and inter-mask separation
    # Weight the losses appropriately
    total_loss = intra_loss_total + 0.1 * inter_loss
    
    return total_loss

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:,1:-1, :-2] + disp[:,1:-1,2:] - 2 * disp[:,1:-1,1:-1])
    grad_disp_y = torch.abs(disp[:,:-2, 1:-1] + disp[:,2:,1:-1] - 2 * disp[:,1:-1,1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

