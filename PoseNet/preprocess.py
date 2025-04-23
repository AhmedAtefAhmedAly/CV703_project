import cv2
import numpy as np
import matplotlib.pyplot as plt

def to_single_channel(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def power_law_transformation(image, gamma):
    norm_img = image / 255.0
    transformed_img = np.power(norm_img, gamma)
    output = np.uint8(transformed_img * 255)
    return output

def dichromatic_optimization_multi(
    intensity_path,
    dolp_path,
    aolp_path,
    output_path,
    phi0_list=[90, 135],  
    sigma=10,
    gamma=2.0,
    his = True
):
    """
    A multi-angle extension of the dichromatic reflection model approach.
    
    - intensity_path: Path to the grayscale intensity image.
    - dolp_path: Path to the DoLP image.
    - aolp_path: Path to the AoLP image.
    - output_path: Where to save the final diffuse result.
    - phi0_list: A list of angles (in degrees) for possible specular reflection.
    - sigma: Standard deviation (in degrees) for the Gaussian weight around each phi0.
    - tv_weight: Weight parameter for total variation denoising.
    - gamma: Power-law (gamma) correction for the intensity image.
    """
  
    I = cv2.imread(intensity_path, cv2.IMREAD_GRAYSCALE)
    dolp = cv2.imread(dolp_path, cv2.IMREAD_COLOR)
    aolp = cv2.imread(aolp_path, cv2.IMREAD_COLOR)

    I = power_law_transformation(I, gamma)

    dolp = to_single_channel(dolp)
    aolp = to_single_channel(aolp)

    dolp_norm = dolp.astype(np.float32) / 255.0

    # Map AoLP from [0..255] -> [0..180] degrees
    aolp_deg = aolp.astype(np.float32) * (180.0 / 255.0)

    I_float = I.astype(np.float32)

    # Compute a Specular Estimate for Each Angle & Combine 
    # We’ll combine them by taking the maximum specular estimate across all angles.
    I_spec_est_multi = np.zeros_like(I_float, dtype=np.float32)

    for phi0 in phi0_list:
        # Angular difference (account for 180° periodicity)
        diff = np.abs(aolp_deg - phi0)
        diff = np.minimum(diff, 180.0 - diff)

        # Gaussian weight ~ how close AoLP is to each phi0
        weight = np.exp(- (diff ** 2) / (2 * sigma**2))

        # Specular estimate for this phi0
        I_spec_est = I_float * dolp_norm * weight

        # Combine with maximum (you could also do sum or average)
        I_spec_est_multi = np.maximum(I_spec_est_multi, I_spec_est)

    
    I_diff_init = I_float - I_spec_est_multi


    init_dif = np.clip(I_diff_init, 0, 255).astype(np.uint8)


    # Two diff ways to normalize the final image 
    if his:
        init_dif = cv2.equalizeHist(init_dif)
        output_path = output_path[:-4] + "his"
    else:
        clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(16,16))
        init_dif = clahe.apply(init_dif)
        output_path = output_path[:-4] + "clahe"

    init_dif = cv2.GaussianBlur(init_dif, (5, 5), 0)

    laplacian_16 = cv2.Laplacian(init_dif, cv2.CV_16S, ksize=3)
    laplacian = cv2.convertScaleAbs(laplacian_16)

    init_dif = cv2.addWeighted(init_dif, 1, laplacian, 0.4, 0)

    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.imshow(I, cmap='gray')
    plt.title("Original Intensity (Gamma-Corrected)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    
    plt.imshow(init_dif, cmap='gray')
    plt.title("Initial Diffuse (Multi-Angle)")
    plt.axis('off')

    plt.savefig(output_path)
    plt.close()
    
    with open(output_path + ".txt", 'w') as f:
        f.write(f"Phi: {phi0_list}\nSigma: {sigma}\nGamma: {gamma}")


if __name__ == '__main__':

    phi0_candidates = [0, 45, 75, 90, 135, 150, 180, 220, 275, 320, 360]
    #phi0_candidates = [x for x in range(361, 10)]
    
    # num = 91

    # dichromatic_optimization_multi(
    #     intensity_path=f"images/0000{num}.jpg",
    #     dolp_path=f"images/0000{num}_dolp.png",
    #     aolp_path=f"images/0000{num}_aolp.png",
    #     output_path=f"results/dichromatic_multi_{num}_.png",
    #     phi0_list=phi0_candidates,
    #     sigma=35,      # how wide the angular Gaussian is
    #     gamma=3,
    #     his=True
    # )


    num = 5

    dichromatic_optimization_multi(
        intensity_path=f"/l/users/bahey.tharwat/bpc_baseline/datasets/ipd_val/val/000008/rgb_cam3/00000{num}.png",
        dolp_path=f"/l/users/bahey.tharwat/bpc_baseline/datasets/ipd_val/val/000008/dolp_cam3/00000{num}.png",
        aolp_path=f"/l/users/bahey.tharwat/bpc_baseline/datasets/ipd_val/val/000008/aolp_cam3/00000{num}.png",
        output_path=f"/l/users/bahey.tharwat/bpc_baseline/bpc/results/dichromatic_multi_{num}_.png",
        phi0_list=phi0_candidates,
        sigma=35,      # how wide the angular Gaussian is
        gamma=3,
        his=True
    )