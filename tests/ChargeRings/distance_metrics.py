#!/usr/bin/python

import numpy as np
from scipy.stats import wasserstein_distance
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate2d

def wasserstein_1d_V(img1, img2):
    """
    Calculate 1D Wasserstein distance along voltage (y) axis.
    
    Args:
        img1, img2: Images to compare (shape [nV, nx])
        
    Returns:
        float: Average Wasserstein distance along V direction
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have the same shape. Got {img1.shape} and {img2.shape}")
    
    n_x = img1.shape[1]
    distances = []
    
    # Calculate Wasserstein distance for each column (V-direction)
    for i in range(n_x):
        # Get probability distributions along voltage
        u = img1[:, i]
        v = img2[:, i]
        
        # Normalize to ensure valid probability distributions
        if np.sum(u) > 0:
            u = u / np.sum(u)
        if np.sum(v) > 0:
            v = v / np.sum(v)
        
        # Calculate Wasserstein distance
        dist = wasserstein_distance(u, v)
        distances.append(dist)
    
    # Return average distance
    return np.mean(distances)

def wasserstein_1d_X(img1, img2):
    """
    Calculate 1D Wasserstein distance along position (x) axis.
    
    Args:
        img1, img2: Images to compare (shape [nV, nx])
        
    Returns:
        float: Average Wasserstein distance along X direction
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have the same shape. Got {img1.shape} and {img2.shape}")
    
    n_v = img1.shape[0]
    distances = []
    
    # Calculate Wasserstein distance for each row (X-direction)
    for i in range(n_v):
        # Get probability distributions along position
        u = img1[i, :]
        v = img2[i, :]
        
        # Normalize to ensure valid probability distributions
        if np.sum(u) > 0:
            u = u / np.sum(u)
        if np.sum(v) > 0:
            v = v / np.sum(v)
        
        # Calculate Wasserstein distance
        dist = wasserstein_distance(u, v)
        distances.append(dist)
    
    # Return average distance
    return np.mean(distances)

def wasserstein_combined(img1, img2, v_weight=0.5):
    """
    Calculate combined Wasserstein distance along both x and V directions.
    
    Args:
        img1, img2: Images to compare (shape [nV, nx])
        v_weight: Weight for V-direction (1-v_weight is used for X-direction)
        
    Returns:
        float: Weighted average of distances in both directions
    """
    dist_v = wasserstein_1d_V(img1, img2)
    dist_x = wasserstein_1d_X(img1, img2)
    
    return v_weight * dist_v + (1 - v_weight) * dist_x

def rmse(img1, img2):
    """
    Calculate root mean square error between two images.
    
    Args:
        img1, img2: Images to compare (shape [nV, nx])
        
    Returns:
        float: RMSE between images
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have the same shape. Got {img1.shape} and {img2.shape}")
    
    # Calculate RMSE
    return np.sqrt(np.mean((img1 - img2) ** 2))

def normalized_cross_correlation(img1, img2):
    """
    Calculate normalized cross-correlation between two images.
    
    Args:
        img1, img2: Images to compare (shape [nV, nx])
        
    Returns:
        float: Normalized cross-correlation coefficient (-1 to 1)
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have the same shape. Got {img1.shape} and {img2.shape}")
    
    # Normalize images
    img1_norm = (img1 - np.mean(img1)) / (np.std(img1) + 1e-10)
    img2_norm = (img2 - np.mean(img2)) / (np.std(img2) + 1e-10)
    
    # Calculate correlation
    correlation = np.mean(img1_norm * img2_norm)
    
    # Convert correlation to distance (1 - correlation, scaled to positive values)
    # This way, higher correlation gives lower distance (better match)
    return 1.0 - (correlation + 1) / 2.0

def feature_similarity(img1, img2, sigma=1.0):
    """
    Calculate feature similarity by first applying Gaussian smoothing to highlight features.
    
    Args:
        img1, img2: Images to compare (shape [nV, nx])
        sigma: Standard deviation for Gaussian filter
        
    Returns:
        float: RMSE between filtered images
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have the same shape. Got {img1.shape} and {img2.shape}")
    
    # Apply Gaussian filter to enhance features
    img1_filt = gaussian_filter(img1, sigma=sigma)
    img2_filt = gaussian_filter(img2, sigma=sigma)
    
    # Calculate RMSE on filtered images
    return np.sqrt(np.mean((img1_filt - img2_filt) ** 2))

# Dictionary mapping metric names to functions
AVAILABLE_METRICS = {
    'wasserstein_v':        wasserstein_1d_V,
    'wasserstein_x':        wasserstein_1d_X,
    'wasserstein_combined': wasserstein_combined,
    'rmse':                 rmse,
    'cross_correlation':    normalized_cross_correlation,
    'feature_similarity':   feature_similarity
}

def get_metric_function(metric_name):
    """
    Get a distance metric function by name.
    
    Args:
        metric_name: Name of the metric to use
        
    Returns:
        callable: The corresponding distance metric function
    """
    if metric_name not in AVAILABLE_METRICS:
        valid_metrics = list(AVAILABLE_METRICS.keys())
        raise ValueError(f"Unknown metric '{metric_name}'. Valid options are: {valid_metrics}")
    
    return AVAILABLE_METRICS[metric_name]
