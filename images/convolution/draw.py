import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from skimage import io, color

# Define the kernel
kernel = np.array([[-1, -1, -1], 
                   [-1,  8, -1], 
                   [-1, -1, -1]])

# Đọc ảnh từ file
original_image = io.imread('youngmonkeys.png')


# Chuyển ảnh sang dạng đen trắng
if len(original_image.shape) == 3:
    image = color.rgb2gray(original_image)

# Thực hiện phép tích chập
convolved_image = convolve(image, kernel)

# Vẽ ảnh gốc và ảnh đặc trưng được trích xuất
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(original_image)
plt.axis('off')

kernel_image = io.imread('youngmonkeys.png')
plt.subplot(1, 3, 2)
plt.imshow(kernel_image)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(convolved_image, cmap='gray')
plt.axis('off')

plt.show()