import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def crop_signature(img, x_min, y_min, x_max, y_max):
    """
    Crops the signature region from a document image.
    """
    return img[y_min:y_max, x_min:x_max]

def extract_baseline1_edge_shape(img_crop):
    """
    Baseline 1: Edge & Shape Analysis
    Extracts geometric and edge-based features.
    """
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    
    # 1. Apply adaptive thresholding to get binary representation
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # 2. Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Feature 1: Edge Density
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # Feature 2: Edge Spread (standard deviation of edge coordinates)
    y_indices, x_indices = np.where(edges > 0)
    if len(x_indices) > 0:
        x_std = np.std(x_indices) / edges.shape[1]
        y_std = np.std(y_indices) / edges.shape[0]
    else:
        x_std, y_std = 0, 0
        
    # Feature 3: Contour Irregularity (via Distance Transform)
    # Distance transform calculates the distance of each background pixel to the nearest foreground pixel.
    dist_transform = cv2.distanceTransform(255 - thresh, cv2.DIST_L2, 3)
    mean_stroke_thickness = np.mean(dist_transform[thresh > 0]) if np.sum(thresh > 0) > 0 else 0
    std_stroke_thickness = np.std(dist_transform[thresh > 0]) if np.sum(thresh > 0) > 0 else 0
    
    features = np.array([edge_density, x_std, y_std, mean_stroke_thickness, std_stroke_thickness], dtype=np.float32)
    # Handle NaNs just in case
    features = np.nan_to_num(features)
    
    return features

def extract_baseline2_lbp(img_crop, P=8, R=1):
    """
    Baseline 2: LBP Texture Analysis
    Extracts LBP histogram.
    """
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    
    # Apply LBP (Local Binary Pattern)
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    
    # Compute normalized histogram
    # Uniform LBP with P=8 has 10 bins (P+2 bins)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float32")
    hist /= (hist.sum() + 1e-7)
    
    return hist

def extract_proposed_frequency(img_crop):
    """
    Extracts frequency-domain features using 2D FFT.
    """
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 2D Fast Fourier Transform
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    
    # Magnitude spectrum
    magnitude = np.abs(fshift)
    total_energy = np.sum(magnitude) + 1e-7
    
    # Normalized spectrum for entropy calculations
    prob_spectrum = magnitude / total_energy
    
    # Centroid coordinate (center of frequency space)
    cy, cx = h // 2, w // 2
    
    # Feature 1: High-Frequency Energy Ratio
    # We define a low-frequency radius (e.g., 10% of minimum dimension)
    r_cutoff = 0.1 * min(h, w)
    y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
    mask_low = x**2 + y**2 <= r_cutoff**2
    low_freq_energy = np.sum(magnitude[mask_low])
    high_freq_energy = total_energy - low_freq_energy
    hf_ratio = high_freq_energy / total_energy
    
    # Feature 2: Spectral Entropy
    # Measures the randomness/irregularity of the frequency distribution
    spectral_entropy = -np.sum(prob_spectrum * np.log2(prob_spectrum + 1e-12))
    
    # Feature 3: Radial Frequency Distribution (5 concentric rings)
    r_max = np.sqrt(cx**2 + cy**2)
    radial_features = []
    rings = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0] # fraction of max radius
    for idx in range(len(rings) - 1):
        r_inner = rings[idx] * r_max
        r_outer = rings[idx+1] * r_max
        mask_ring = (x**2 + y**2 > r_inner**2) & (x**2 + y**2 <= r_outer**2)
        ring_energy = np.sum(magnitude[mask_ring]) / total_energy
        radial_features.append(ring_energy)
        
    # Feature 4: Edge-Induced Frequency Spikes
    # Digital insertion creates sharp rectangular boundaries, manifesting as cross-like lines in FFT.
    # We measure energy along the vertical and horizontal axes of the FFT (excluding center).
    vertical_strip = magnitude[max(0, cy-2):min(h, cy+3), :]
    horizontal_strip = magnitude[:, max(0, cx-2):min(w, cx+3)]
    
    axis_energy = np.sum(vertical_strip) + np.sum(horizontal_strip)
    # Exclude central low-frequency area to focus on high-frequency spikes
    y_indices, x_indices = np.ogrid[-cy:h-cy, -cx:w-cx]
    mask_exclude_center = (x_indices**2 + y_indices**2 > (3 * r_cutoff)**2)
    
    axis_mask_y = (y_indices >= -2) & (y_indices <= 2)
    axis_mask_x = (x_indices >= -2) & (x_indices <= 2)
    
    spike_magnitude_y = np.sum(magnitude[axis_mask_y & mask_exclude_center])
    spike_magnitude_x = np.sum(magnitude[axis_mask_x & mask_exclude_center])
    
    avg_rest = np.mean(magnitude[~axis_mask_y & ~axis_mask_x & mask_exclude_center]) + 1e-7
    
    spike_ratio_y = spike_magnitude_y / (avg_rest * np.sum(axis_mask_y & mask_exclude_center) + 1e-7)
    spike_ratio_x = spike_magnitude_x / (avg_rest * np.sum(axis_mask_x & mask_exclude_center) + 1e-7)
    
    freq_features = [hf_ratio, spectral_entropy, spike_ratio_y, spike_ratio_x] + radial_features
    return np.array(freq_features, dtype=np.float32)

def extract_proposed_hybrid(img_crop, P=8, R=1):
    """
    Proposed Method: LBP Texture Features + FFT Frequency Features.
    """
    lbp_feats = extract_baseline2_lbp(img_crop, P, R)
    freq_feats = extract_proposed_frequency(img_crop)
    
    # Concatenate spatial texture features and spectral frequency features
    return np.concatenate([lbp_feats, freq_feats])
