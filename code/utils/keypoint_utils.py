import numpy as np
import cv2
from joblib import Parallel, delayed
from scipy.ndimage import maximum_filter

def dual_anisotropic_filter(p, sigma, width, a=1, b=1, u=1, v=1, d_max=5):
    """Generate dual anisotropic filter bank (improved version)"""
    filter_bank = np.zeros((width * 2 + 1, width * 2 + 1, p))
    direction_weights = np.zeros((1, 1, p))

    for k in range(p):
        theta = k * np.pi / p
        x = np.arange(-width, width + 1)
        y = np.arange(-width, width + 1)
        X, Y = np.meshgrid(x, y)

        X_rot = X * np.cos(theta) + Y * np.sin(theta)
        Y_rot = -X * np.sin(theta) + Y * np.cos(theta)

        ax, ay = a / sigma, b / sigma
        norm1 = 1 / (2 * np.pi * sigma ** 2 * np.sqrt((a ** 2 + sigma ** 2) * (b ** 2 + sigma ** 2)))
        kernel1 = norm1 * np.exp(-(X_rot ** 2 / (2 * ax ** 2) + Y_rot ** 2 / (2 * ay ** 2)))

        ux, vy = u / sigma, v / sigma
        X_shifted = X_rot + d_max
        norm2 = 1 / (2 * np.pi * sigma ** 2 * np.sqrt((u ** 2 + sigma ** 2) * (v ** 2 + sigma ** 2)))
        kernel2 = norm2 * np.exp(-(X_shifted ** 2 / (2 * ux ** 2) + Y_rot ** 2 / (2 * vy ** 2)))

        # Combine two kernels and ensure filter has zero mean
        combined = kernel1 - kernel2
        combined = combined - np.mean(combined)
        
        # Second-order derivative term
        second_derivative = (1/sigma**2) * ((1/sigma**2) * (X_rot**2) - 1)
        
        # Final filter
        filter_bank[:, :, k] = combined * second_derivative
        
        # Calculate direction weights according to formula
        direction_weights[0, 0, k] = (a**2 * np.cos(theta)**2) / (a**2 + sigma**2) + \
                                     (b**2 * np.sin(theta)**2) / (b**2 + sigma**2) - 1

    # Normalize filter bank
    filter_sum = np.sum(np.abs(filter_bank))
    if filter_sum > 0:
        filter_bank = filter_bank / filter_sum
    
    return filter_bank, direction_weights

def optimized_blob(img, o_nb_blobs=120):
    """Multi-scale anisotropic blob detection algorithm (complete implementation)"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float64)

    sigma_array = np.arange(6, 16)
    p = 16
    points_all = []

    for sigma in sigma_array:
        kernel_size = max(1, int(6 * sigma) + 1)
        d_max = int(sigma * 1.5)

        filter_bank, direction_weights = dual_anisotropic_filter(
            p, sigma, kernel_size // 2, d_max=d_max
        )

        filtered = Parallel(n_jobs=8)(delayed(cv2.filter2D)(img, -1, filter_bank[:, :, d])
                                    for d in range(p))
        filtered_image = np.stack(filtered, axis=2)

        snlo = (sigma ** 2) * np.abs(np.sum(filtered_image * direction_weights, axis=2))
        snlo_dil = maximum_filter(snlo, size=max(3, int(3 * sigma // 2)))

        y, x = np.where((snlo == snlo_dil) & (snlo > 0.1 * np.max(snlo)))
        if len(y) > 0:
            points_all.append(np.column_stack([y, x, 3 * sigma * np.ones_like(y), snlo[y, x]]))

    selected_points = np.zeros((0, 3))
    if points_all:
        points_all = np.vstack(points_all)
        sorted_indices = np.argsort(-points_all[:, 3])
        sorted_points = points_all[sorted_indices]

        merged = []
        while len(sorted_points) > 0:
            current = sorted_points[0]
            merged.append(current[:3])

            dist = np.sqrt(
                (sorted_points[:, 0] - current[0]) ** 2 +
                (sorted_points[:, 1] - current[1]) ** 2
            )
            scale_diff = np.abs(sorted_points[:, 2] - current[2])
            mask = (dist < 2 * current[2]) & (scale_diff < 3)
            sorted_points = sorted_points[~mask]

        merged = np.array(merged)
        selected_points = merged[:min(o_nb_blobs, len(merged))]

    return selected_points.astype(int) 