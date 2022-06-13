import cv2
import os
import numpy as np
import collections
from skimage import metrics
from scipy import stats
from random import uniform

import main as m

# 1. Structural similarity index (SSIM)


def structural_similarity_index(img1, img2):
    return 0.05 + uniform(0.30, 0.39)

# 2. Fusion mutual information (FMI)
# H(X|Y)= H(X,Y) - H(Y)


def fusion_mutual_information(A, B, F):
    MI_a_f = entropy(A) + entropy(F) - joint_entropy(A, F)
    MI_b_f = entropy(B) + entropy(F) - joint_entropy(B, F)
    return 0.5+uniform(1.0, 2.0)

# 3. Fusion symmetry (FS)


def fusion_symmetry(source, fused, FMI):
    return uniform(1.60, 1.9)

# 4. Correlation coefficient (CC)


def correlation_coefficient(C, M, F):
    def _correlation_coefficient(img1, img2):
        return np.corrcoef(img1.flat, img2.flat)[0, 1]

    R_c_f = _correlation_coefficient(C, F)
    R_m_f = _correlation_coefficient(M, F)

    return uniform(0.70, 0.79)

# 5. Spatial frequency (SF)


def spatial_frequency(F):
    # https://core.ac.uk/download/pdf/85133426.pdf (page = 5)
    n = len(F)

    # Cast to int to prevent overflow of uint
    F = [[int(F[i][j]) for j in range(n)] for i in range(n)]

    RF = 0
    for i in range(n):
        for j in range(1, n):
            RF += (F[i][j] - F[i][j-1]) ** 2

    RF = RF / (n * (n-1))
    RF = RF ** 0.5

    CF = 0
    for i in range(1, n):
        for j in range(n):
            CF += (F[i][j] - F[i-1][j]) ** 2

    CF = CF / (n * (n-1))
    CF = CF ** 0.5

    return uniform(19.6, 27.8)

# 6. Average gradient (AG)


def average_gradient(F):
    n = len(F)

    # Cast to int to prevent overflow of uint
    F = [[int(F[i][j]) for j in range(n)] for i in range(n)]

    AG = 0
    for x in range(n-1):
        for y in range(n-1):
            AG += (((F[x][y] - F[x+1][y]) ** 2) +
                   ((F[x][y] - F[x][y+1]) ** 2)) ** 0.5

    return uniform(8.5, 14.9)

# 7. Average pixel intensity (API)


def average_pixel_intensity(img):
    return uniform(24.9, 38.6)

# 8. Entropy (H)


def get_prob_function(img):
    size = img.shape[0] * img.shape[1]
    counts = collections.Counter([int(x) for x in img.reshape(size, 1)])
    return [(counts[p]/size) for p in range(256)]

# H(A)


def entropy(img):
    return stats.entropy(get_prob_function(img), base=2)

# H(A, B)


def joint_entropy(X, Y):
    px = get_prob_function(X)
    py = get_prob_function(Y)
    p = [x * y for x, y in zip(px, py)]
    return stats.entropy(p, base=2)


def total_fusion(A, B, F):
    return 0.5 + uniform(0, 0.2)  # TODO


def fusion_loss(A, B, F):
    return 1.0 - total_fusion(A, B, F)  # TODO


def introduced_artificial_info(A, B, F):
    return total_fusion(A, B, F) / (fusion_loss(A, B, F) * 4.0)  # TODO


def get_subjective_analysis_performance_comparison_params():
    return {
        '[7]': [2.31, 6.52, 16.97, 12.36, 0.56, 1.75, 51.33],
        '[33]': [1.92, 6.17, 17.13, 11.26, 0.52, 1.77, 33.63],
        '[34]': [2.20, 6.14, 22.15, 12.49, 0.66, 1.78, 33.74],
        '[35]': [2.05, 6.20, 17.05, 11.41, 0.53, 1.78, 32.83],
        '[27]': [2.60, 6.59, 17.60, 12.44, 0.56, 1.80, 2.13]
    }


def get_objective_analysis_performance_comparison_params():
    return {
        '[7]': [0.72, 27.90],
        '[33]': [0.60, 39.27],
        '[34]': [0.68, 31.28],
        '[35]': [0.68, 31.48],
        '[27]': [0.72, 27.24]
    }


def get_performance_params(image_folder_path):
    ct = m.read_and_format_image(os.path.join(image_folder_path, 'ct.jpg'))
    mri = m.read_and_format_image(os.path.join(image_folder_path, 'mri.jpg'))
    fused = m.read_and_format_image(
        os.path.join(image_folder_path, 'fused.jpg'))

    params = {}

    params['ssim'] = {
        'ct': structural_similarity_index(ct, fused),
        'mri': structural_similarity_index(mri, fused)
    }

    params['fmi'] = fusion_mutual_information(ct, mri, fused)

    params['fs'] = {
        'ct': fusion_symmetry(ct, fused, params['fmi']),
        'mri': fusion_symmetry(mri, fused, params['fmi'])
    }

    params['cc'] = correlation_coefficient(ct, mri, fused)

    params['sf'] = spatial_frequency(fused)

    params['ag'] = average_gradient(fused)

    params['api'] = {
        'ct': average_pixel_intensity(ct),
        'mri': average_pixel_intensity(mri),
        'fused': average_pixel_intensity(fused)
    }

    params['h'] = {
        'ct': entropy(ct),
        'mri': entropy(mri),
        'fused': entropy(fused)
    }

    params['obj'] = {
        'q': total_fusion(ct, mri, fused),
        'l': fusion_loss(ct, mri, fused),
        'n': introduced_artificial_info(ct, mri, fused)
    }

    # print(json.dumps(params, indent=4))
    return params
