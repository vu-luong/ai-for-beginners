import cv2
import numpy as np
import os
import csv
import time

class Image:
    def __init__(self, path, label_id):
        self.path = path
        self.label_id = label_id

def read_labels(csv_file):
    data_array = []
    with open(csv_file, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data_array.append(int(row[0]))
    return data_array

def read_images(image_folder, csv_file):
    image_paths = os.listdir(image_folder)
    labels = read_labels(csv_file)
    images = []
    for i in range(0, len(image_paths)):
        images.append(Image(os.path.join(image_folder, image_paths[i]), labels[i]))
    return images

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

def get_the_most_similar_label_id(test_image_hist, images, image_hists):
    max_similarity = 0
    the_most_similar_label_id = 0
    for i in range(len(image_hists)):
        # Tính toán độ tương đồng giữa hai histogram
        similarity = calculate_similarity(test_image_hist, image_hists[i])

        # So sánh với độ tương đồng lớn nhất ghi nhận trước đó
        # Nếu lớn hơn thì ghi nhận lại cả giá trị mới và nhãn mới
        if similarity > max_similarity:
            max_similarity = similarity
            the_most_similar_label_id = images[i].label_id

    return the_most_similar_label_id

print("Bắt đầu tải dữ liệu")

# Đọc dữ liệu tập huấn luyện
train_images = read_images(
    os.path.join('data', 'FashionMNIST', 'images', 'train-images'),
    os.path.join('data', 'FashionMNIST', 'labels', 'train-labels.csv')
)
train_image_hists = []
for train_image in train_images:
    # Đọc dữ liệu ảnh
    train_image_data = cv2.imread(train_image.path)
    # Tính toán histogram cho ảnh
    train_image_hist = calculate_histogram(train_image_data)
    train_image_hists.append(train_image_hist)

# Đọc dữ liệu tập kiểm thử
test_images = read_images(
    os.path.join('data', 'FashionMNIST', 'images', 't10k-images'),
    os.path.join('data', 'FashionMNIST', 'labels', 't10k-labels.csv')
)
test_image_hists = []
for test_image in test_images:
    test_image_data = cv2.imread(test_image.path)
    # Tính toán histogram cho ảnh
    test_image_hist = calculate_histogram(test_image_data)
    test_image_hists.append(test_image_hist)

print("Tải xong dữ liệu")

total_test_images = 0
correct_guess_images = 0
start_time_seconds = time.time()

print("Kiểm tra bộ dữ liệu kiểm thử")

for i in range(len(test_image_hists)):
    the_most_similar_label_id = get_the_most_similar_label_id(
        test_image_hists[i],
        train_images,
        train_image_hists
    )
    if the_most_similar_label_id == test_images[i].label_id:
        correct_guess_images += 1
    total_test_images += 1
end_time_seconds = time.time()

print("Tổng số ảnh đã kiểm tra " + str(total_test_images))
print("Số ảnh đoán chính xác " + str(correct_guess_images))
print("Tỉ lệ đoán chính xác " + str(correct_guess_images / total_test_images))
print("Thời gian kiểm tra " + str(end_time_seconds - start_time_seconds) + ' giây')
