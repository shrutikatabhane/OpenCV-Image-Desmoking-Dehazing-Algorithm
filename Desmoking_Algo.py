import cv2
import numpy as np

def get_dark_channel(image, size):
    """Get the dark channel prior in an image."""
    min_channel = cv2.min(cv2.min(image[:, :, 0], image[:, :, 1]), image[:, :, 2])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel

def get_atmosphere(image, dark_channel):
    """Estimate the atmospheric light in the image."""
    h, w = image.shape[:2]
    num_pixels = h * w
    num_brightest = int(max(np.floor(num_pixels * 0.001), 1))

    dark_vec = dark_channel.ravel()
    image_vec = image.reshape(num_pixels, 3)
    
    indices = np.argsort(dark_vec)[-num_brightest:]
    atmospheric_light = np.mean(image_vec[indices], axis=0)
    
    return atmospheric_light

def get_transmission(image, atmospheric_light, size):
    """Estimate the transmission map."""
    omega = 0.95
    normed_image = np.empty(image.shape, image.dtype)
    
    for i in range(3):
        normed_image[:, :, i] = image[:, :, i] / atmospheric_light[i]
    
    transmission = 1 - omega * get_dark_channel(normed_image, size)
    
    return transmission

def guided_filter(I, p, radius, eps):
    """Perform guided filtering to refine the transmission map."""
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (radius, radius))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (radius, radius))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (radius, radius))
    cov_Ip = mean_Ip - mean_I * mean_p
    
    mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (radius, radius))
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (radius, radius))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (radius, radius))
    
    q = mean_a * I + mean_b
    return q

def recover_image(image, transmission, atmospheric_light, t0=0.1):
    """Recover the de-hazed image."""
    transmission = np.maximum(transmission, t0)
    
    recovered = np.empty(image.shape, image.dtype)
    for i in range(3):
        recovered[:, :, i] = (image[:, :, i] - atmospheric_light[i]) / transmission + atmospheric_light[i]
    
    recovered = np.clip(recovered, 0, 255).astype(np.uint8)
    
    return recovered

def clahe_equalized(img):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def dehaze(image):
    """De-haze an image and return the result."""
    dark_channel = get_dark_channel(image, 15)
    atmospheric_light = get_atmosphere(image, dark_channel)
    transmission = get_transmission(image, atmospheric_light, 15)
    
    # Refine the transmission map using guided filter
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.0
    transmission_refined = guided_filter(gray_image, transmission, radius=60, eps=1e-3)
    
    recovered_image = recover_image(image, transmission_refined, atmospheric_light)
    
    # Apply CLAHE for additional enhancement
    enhanced_image = clahe_equalized(recovered_image)
    
    return enhanced_image

# Example usage
input_image_path = 'hazy_image.jpg'
output_image_path = 'dehazed_image.jpg'

# Read the input image
image = cv2.imread(input_image_path)

# Apply dehazing
dehazed_image = dehaze(image)

# Save and display the results
cv2.imwrite(output_image_path, dehazed_image)
cv2.imshow('Original Image', image)
cv2.imshow('Dehazed Image', dehazed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
