import os
import numpy as np
import cv2

def read_and_format_image(img_name):
    image = cv2.imread(img_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    return image

# ===== A. Edge Detection ======

# Returns the appropriate mask kernel for the given direction angle


def get_convolution_mask(degree):
    masks = {
        0: np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
        45: np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]),
        90: np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        135: np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
    }
    return masks.get(degree)

# Returns the edge detected image for the given image by convoluting the image using the given mask


def sobel_edge_detection(image, mask):
    mask = mask / (np.sum(mask) if np.sum(mask) != 0 else 1)
    return cv2.filter2D(image, -1, mask)

# Return the edge detected images at various direction angles as a list


def get_edge_detected_images(image):
    edge_detected_images = []
    for i in range(0, 135+1, 45):
        cur_mask = get_convolution_mask(i)
        edge_detected_image = sobel_edge_detection(image, cur_mask)
        edge_detected_images.append(edge_detected_image)
    return edge_detected_images

# ===== B. Detail image generation ======

# EA(x,y) = max(|A(x,y)0|, |A(x,y)45 |, |A(x,y)90 |, |A(x,y)135 |)
# EB(x,y) = max(|B(x,y)0|, |B(x,y)45 |, |B(x,y)90 |, |B(x,y)135 |)


def get_edge_max_image(images):
    max_image = images[0]
    for i in range(len(images)):
        for j in range(len(images[i])):
            max_image[i][j] = max([abs(image[i][j]) for image in images])
    return max_image

# DA(x,y) = B(x,y) − EA(x,y)
# DB(x,y) = A(x,y) − EB(x,y)


def get_detailed_image_from_edge_max_image(edge_max_image, source_complement):
    return np.subtract(source_complement, edge_max_image)

# ===== C. Weight generation ======

# returns neighbour submatrix for mat[r][c]


def get_M_matrix(mat, r, c):
    # for edge pixels compute the value of their inner pixels
    r = min(r, len(mat)-2)
    c = min(c, len(mat[0])-2)
    r = max(r, 1)
    c = max(c, 1)

    row_start = r - 1
    row_end = r + 1
    col_start = c - 1
    col_end = c + 1
    return mat[row_start:row_end+1, col_start:col_end+1]

# C<x,y> h = 1/(W−1) [M^T M]


def get_Ch_matrix_from_M_matrix(M):
    W = len(M)
    C = (1/(W-1)) * np.dot(np.transpose(M), M)
    return C

# C<x,y> v = 1/(W−1) [M M^T]


def get_Cv_matrix_from_M_matrix(M):
    W = len(M)
    C = (1/(W-1)) * np.dot(M, np.transpose(M))
    return C

# sum of the 3 eigen values of C matrix


def get_sum_of_eigen_values(C):
    eigen_values, _ = np.linalg.eigh(C)
    return eigen_values.sum()


def get_weight_for_given_coord(mat, r, c):
    M = get_M_matrix(mat, r, c)
    Ch = get_Ch_matrix_from_M_matrix(M)
    Cv = get_Cv_matrix_from_M_matrix(M)
    return get_sum_of_eigen_values(Ch) + get_sum_of_eigen_values(Cv)


def get_weight_matrix_from_detailed_image(detailed_image):
    weighted_image = np.array(detailed_image)
    for i in range(len(detailed_image)):
        for j in range(len(detailed_image[i])):
            weighted_image[i][j] = get_weight_for_given_coord(
                detailed_image, i, j)
    return weighted_image

# ===== D. Adaptive pixel fusion ======


def get_Z_score_matrix_from_image(source_image):
    Z = np.array(source_image)
    for i in range(len(source_image)):
        for j in range(len(source_image[i])):
            M = get_M_matrix(source_image, i, j)
            mean = np.mean(M)
            std = np.std(M)
            Z[i][j] = 0 if std == 0 else (source_image[i][j] - mean) / std
    return Z


def get_noise_controlling_coeff_matrix(weighted_matrix, z_scores):
    B = np.array(weighted_matrix)
    for i in range(len(weighted_matrix)):
        for j in range(len(weighted_matrix[i])):
            z = z_scores[i][j]
            B[i][j] = 1 if abs(z) < 3 else weighted_matrix[i][j]/z
    return B


def get_weight_prime_matrix(weighted_matrix, noise_coeff_matrix):
    wt_p = np.array(weighted_matrix)
    for i in range(len(weighted_matrix)):
        for j in range(len(weighted_matrix[i])):
            wt_p[i][j] = weighted_matrix[i][j] * noise_coeff_matrix[i][j]
    return wt_p


def fuse_image(mri_source_image, ct_source_image, weight_prime_a, weight_prime_b):
    fused_image = np.array(mri_source_image)
    for i in range(len(mri_source_image)):
        for j in range(len(mri_source_image[i])):
            num = int(mri_source_image[i][j]) * int(weight_prime_a[i][j])
            num += int(ct_source_image[i][j]) * int(weight_prime_b[i][j])
            den = int(weight_prime_a[i][j]) + int(weight_prime_b[i][j])
            fused_image[i][j] = 1 if den == 0 else num / den  # TODO
    return fused_image


def get_fused_image(mri_source_image, ct_source_image, mri_weighted, ct_weighted):
    Z_a = get_Z_score_matrix_from_image(mri_source_image)
    Z_b = get_Z_score_matrix_from_image(ct_source_image)

    B_a = get_noise_controlling_coeff_matrix(mri_weighted, Z_a)
    B_b = get_noise_controlling_coeff_matrix(ct_weighted, Z_b)

    Wt_p_a = get_weight_prime_matrix(mri_weighted, B_a)
    Wt_p_b = get_weight_prime_matrix(ct_weighted, B_b)

    return fuse_image(mri_source_image, ct_source_image, Wt_p_a, Wt_p_b)

def generate_output_images(input_folder_path, output_folder_path, show_output_image):
    mri_image = read_and_format_image(os.path.join(input_folder_path, 'mri.jpg'))
    ct_image = read_and_format_image(os.path.join(input_folder_path, 'ct.jpg'))

    mri_edges = get_edge_detected_images(mri_image)
    ct_edges = get_edge_detected_images(ct_image)

    mri_max = get_edge_max_image(mri_edges)
    ct_max = get_edge_max_image(ct_edges)

    mri_detailed = get_detailed_image_from_edge_max_image(mri_max, ct_image)
    ct_detailed = get_detailed_image_from_edge_max_image(ct_max, mri_image)

    mri_weighted = get_weight_matrix_from_detailed_image(mri_detailed)
    ct_weighted = get_weight_matrix_from_detailed_image(ct_detailed)

    fused_image = get_fused_image(mri_image, ct_image, mri_weighted, ct_weighted)

    if show_output_image:
        cv2.imshow("Press any key to close", fused_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    names = ['ct', 'mri', \
    'ct-e0', 'ct-e45', 'ct-e90', 'ct-e135', \
    'mri-e0', 'mri-e45', 'mri-e90', 'mri-e135', \
    'ct-max', 'mri-max', 'ct-detailed', 'mri-detailed', 'fused']
    images = [ct_image, mri_image] + ct_edges + mri_edges + \
        [ct_max, mri_max, ct_detailed, mri_detailed, fused_image]

    for name, image in zip(names, images):
        cv2.imwrite(os.path.join(output_folder_path, name + '.jpg'), image)