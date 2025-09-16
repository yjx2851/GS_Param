#!/usr/bin/env python3

import torch
import numpy as np
from typing import Optional, Tuple
import os

def compute_feature_similarity_labels(features: torch.Tensor,
                                    similarity_threshold: float = 0.8,
                                    min_cluster_size: int = 10,
                                    num_hyperplanes: int = 16,
                                    seed: int = 12345) -> torch.Tensor:
    """
    Compute similarity-based labels for features using cosine LSH (random hyperplane hashing).

    Rationale: Pairwise connectivity with a threshold easily percolates into a single
    connected component. Instead, we hash normalized features by the signs of their
    projections onto random hyperplanes. Nearby directions tend to share the same code.

    Args:
        features: (N, D) tensor. Will be normalized along dim=1 if not already.
        similarity_threshold: Unused in LSH mode (kept for API compatibility).
        min_cluster_size: If >0, optionally collapse very small buckets to their nearest large bucket.
        num_hyperplanes: Number of random hyperplanes (bits) for the hash label.
        seed: RNG seed to make labels deterministic across runs.

    Returns:
        labels: (N,) int64 tensor. Label is the integer code from the binary hash.
    """
    assert features.dim() == 2, "features must be (N, D)"
    N, D = features.shape
    device = features.device

    # Normalize to unit length for cosine-based hashing
    feats = torch.nn.functional.normalize(features, dim=1, eps=1e-8)

    # Create random hyperplanes (D, num_hyperplanes)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    hyperplanes = torch.randn((D, num_hyperplanes), generator=gen, device=device)
    hyperplanes = torch.nn.functional.normalize(hyperplanes, dim=0, eps=1e-8)

    # Project and take sign -> binary code
    proj = feats @ hyperplanes  # (N, num_hyperplanes)
    bits = (proj >= 0).to(torch.int64)  # (N, num_hyperplanes) of {0,1}

    # Pack bits into integer labels
    # labels = sum(bits[:,k] << k)
    powers = (2 ** torch.arange(num_hyperplanes, device=device, dtype=torch.int64))
    labels = (bits * powers).sum(dim=1)

    if min_cluster_size > 0:
        # Optionally merge very small buckets to nearest large bucket centroid
        # Compute bucket sizes
        unique, counts = torch.unique(labels, return_counts=True)
        small_buckets = unique[counts < min_cluster_size]
        if small_buckets.numel() > 0:
            # Compute centroids for large buckets
            large_mask = counts >= min_cluster_size
            large_labels = unique[large_mask]
            if large_labels.numel() > 0:
                # Map from label -> centroid
                # Build index for each large label
                centroids = []
                for lbl in large_labels.tolist():
                    idx = (labels == lbl)
                    centroids.append(feats[idx].mean(dim=0, keepdim=True))
                centroids = torch.cat(centroids, dim=0)  # (L, D)
                centroids = torch.nn.functional.normalize(centroids, dim=1, eps=1e-8)

                # For each small bucket point, assign to nearest large centroid by cosine sim
                for lbl in small_buckets.tolist():
                    idx = (labels == lbl)
                    if idx.any():
                        sims = feats[idx] @ centroids.T  # (n_small, L)
                        nearest = sims.argmax(dim=1)
                        labels[idx] = large_labels[nearest]
            else:
                # All buckets are small; keep as-is
                pass

    return labels

def gaussians_to_pcd(gaussian_model, 
                    output_path: str,
                    similarity_threshold: float = 0.8,
                    min_cluster_size: int = 10,
                    default_color: Tuple[float, float, float] = (0.5, 0.5, 0.5)) -> None:
    """
    Convert Gaussian Splatting model to PCD file with similarity-based labels.
    
    Args:
        gaussian_model: Trained GaussianModel instance
        output_path: Path to save the PCD file
        similarity_threshold: Cosine similarity threshold for feature grouping
        min_cluster_size: Minimum cluster size
        default_color: Default RGB color for points (R, G, B) in [0, 1]
    """
    # Extract data from Gaussian model
    xyz = gaussian_model.get_xyz.detach().cpu().numpy()  # (N, 3)
    features = gaussian_model.get_extra_features.detach().cpu()  # (N, D)
    # Also fetch segment as edge attribute
    try:
        edge = gaussian_model.get_segment.detach().cpu().numpy().reshape(-1)
    except Exception:
        # Fallback if property name differs or not available
        edge = np.zeros((xyz.shape[0],), dtype=np.float32)
    
    print(f"Converting {len(xyz)} Gaussians to PCD...")
    print(f"Feature dimension: {features.shape[1]}")
    
    # Compute similarity-based labels
    print("Computing similarity-based labels...")
    labels = compute_feature_similarity_labels(features, similarity_threshold, min_cluster_size)
    labels_np = labels.cpu().numpy()
     
    # Count unique labels
    unique_labels, counts = torch.unique(labels, return_counts=True)
    print(f"Found {len(unique_labels)} unique groups:")
    for label, count in zip(unique_labels.cpu().numpy(), counts.cpu().numpy()):
        print(f"  Group {label}: {count} points")
    
    # Create colors based on labels
    colors = np.tile(default_color, (len(xyz), 1))
    
    # Generate colors for different groups
    if len(unique_labels) > 1:
        # Create a simple color palette
        num_colors = len(unique_labels)
        color_palette = np.random.RandomState(42).rand(num_colors, 3)  # Fixed seed for reproducibility
        
        for i, label in enumerate(unique_labels.cpu().numpy()):
            mask = labels_np == label
            colors[mask] = color_palette[i]
    
    # Create PCD file content
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Write PCD header with feature dimensions
        feature_dim = features.shape[1]
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z rgb label edge")
        for i in range(feature_dim):
            f.write(f" feature_{i}")
        f.write("\n")
        f.write("SIZE 4 4 4 4 4 4")
        for i in range(feature_dim):
            f.write(" 4")
        f.write("\n")
        f.write("TYPE F F F U U F")
        for i in range(feature_dim):
            f.write(" F")
        f.write("\n")
        f.write("COUNT 1 1 1 1 1 1")
        for i in range(feature_dim):
            f.write(" 1")
        f.write("\n")
        f.write(f"WIDTH {len(xyz)}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(xyz)}\n")
        f.write("DATA ascii\n")
        
        # Write point data
        features_np = features.cpu().numpy()
        for i in range(len(xyz)):
            x, y, z = xyz[i]
            r, g, b = colors[i]
            label = int(labels_np[i])
            feature_vector = features_np[i]
            edge_val = float(edge[i])
            
            # Convert RGB to packed integer format (0xRRGGBB)
            rgb_packed = int(r * 255) << 16 | int(g * 255) << 8 | int(b * 255)
            
            # Write coordinates, RGB, label, and feature vector
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {rgb_packed} {label} {edge_val:.6f}")
            for j in range(feature_dim):
                f.write(f" {feature_vector[j]:.6f}")
            f.write("\n")
    
    print(f"PCD file saved to: {output_path}")
    
    # Also save labels as a separate file for reference
    labels_path = output_path.replace('.pcd', '_labels.txt')
    np.savetxt(labels_path, labels_np, fmt='%d')
    print(f"Labels saved to: {labels_path}")
    
    # Save complete feature vectors as a separate file
    features_path = output_path.replace('.pcd', '_features.txt')
    np.savetxt(features_path, features_np, fmt='%.6f')
    print(f"Feature vectors saved to: {features_path}")
    
    # Save feature statistics
    stats_path = output_path.replace('.pcd', '_stats.txt')
    with open(stats_path, 'w') as f:
        f.write(f"Total points: {len(xyz)}\n")
        f.write(f"Feature dimension: {features.shape[1]}\n")
        f.write(f"Similarity threshold: {similarity_threshold}\n")
        f.write(f"Min cluster size: {min_cluster_size}\n")
        f.write(f"Number of groups: {len(unique_labels)}\n")
        f.write(f"Group sizes: {dict(zip(unique_labels.cpu().numpy(), counts.cpu().numpy()))}\n")
        f.write(f"Feature vector file: {features_path}\n")
        f.write(f"Labels file: {labels_path}\n")
    print(f"Statistics saved to: {stats_path}")

def save_gaussian_pcd_with_labels(gaussian_model, 
                                 output_dir: str,
                                 filename: str = "gaussians_with_labels.pcd",
                                 similarity_threshold: float = 0.8,
                                 min_cluster_size: int = 10) -> str:
    """
    Convenience function to save Gaussian model as PCD with labels.
    
    Args:
        gaussian_model: Trained GaussianModel instance
        output_dir: Directory to save the PCD file
        filename: Name of the PCD file
        similarity_threshold: Cosine similarity threshold for feature grouping
        min_cluster_size: Minimum cluster size
        
    Returns:
        Path to the saved PCD file
    """
    output_path = os.path.join(output_dir, filename)
    gaussians_to_pcd(gaussian_model, output_path, similarity_threshold, min_cluster_size)
    return output_path

# Example usage function
def example_usage():
    """Example of how to use the PCD saving functionality."""
    # This would be called from your training script
    # gaussian_model = your_trained_gaussian_model
    # output_path = save_gaussian_pcd_with_labels(
    #     gaussian_model, 
    #     output_dir="./output",
    #     filename="trained_gaussians.pcd",
    #     similarity_threshold=0.7,
    #     min_cluster_size=5
    # )
    pass
