import os
import cv2
import numpy as np

class Image:
    def __init__(self, name, label_id):
        self.name = name
        self.label_id = label_id

images = [
    Image("dog1.jpeg", 0),
    Image("dog2.jpeg", 0),
    Image("dog3.jpeg", 0),
    Image("cat1.jpeg", 1),
    Image("cat2.jpeg", 1),
    Image("cat3.jpeg", 1)
]

label_name_by_id = {
    0: "Chú chó",
    1: "Chú mèo"
}

def calculate_histogram(image):
    # Chuyển đổi ảnh sang không gian màu RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Tính toán histogram cho từng kênh màu
    hist_r = cv2.calcHist([image_rgb], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image_rgb], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([image_rgb], [2], None, [256], [0, 256])

    # Kết hợp các histogram thành một vector đặc trưng
    hist_features = np.concatenate((hist_r, hist_g, hist_b)).flatten()

    # Chuẩn hóa vector đặc trưng
    hist_features /= np.sum(hist_features)
    return hist_features

def calculate_similarity(hist1, hist2):
    # Tính toán độ tương đồng giữa hai histogram sử dụng khoảng cách Bhattacharyya
    distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    return 1 - distance

similarity_by_label_id = {}

# Đọc dữ liệu ảnh
unknow_label_image_data = cv2.imread(os.path.join('images', "cat4.jpeg"))

# Tính toán histogram cho ảnh
unknow_label_image_hist = calculate_histogram(unknow_label_image_data)

max_similarity = 0
the_most_similar_label_id = 0

for image in images:
    # Đọc dữ liệu ảnh
    image_data = cv2.imread(os.path.join('images', image.name))

    # Tính toán histogram cho ảnh
    image_hist = calculate_histogram(image_data)

    # Tính toán độ tương đồng giữa hai histogram
    similarity = calculate_similarity(unknow_label_image_hist, image_hist)

    # So sánh với độ tương đồng lớn nhất ghi nhận trước đó
    # Nếu lớn hơn thì ghi nhận lại cả giá trị mới và nhãn mới
    if similarity > max_similarity:
        max_similarity = similarity
        the_most_similar_label_id = image.label_id

print("Hình ảnh bạn cung cấp có thể là một " + label_name_by_id[the_most_similar_label_id])
