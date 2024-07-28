import cv2
import csv
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import time

# Đọc dữ liệu nhãn
def read_labels(csv_file):
    data_array = []
    with open(csv_file, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data_array.append(int(row[0]))
    return data_array


# Lấy danh sách ảnh trong thư mục.
def read_image_paths(image_folder):
    image_files = os.listdir(image_folder)
    image_paths = []
    for image_file in image_files:
        image_paths.append(os.path.join(image_folder, image_file))
    return sorted(image_paths)


# Tải và xử lý một ảnh.
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image.flatten()

# Tải và xử lý bộ dữ liệu ảnh.
def load_dataset(image_paths):
    images = []
    for path in image_paths:
        images.append(preprocess_image(path))
    return np.array(images)

# Khởi tạo dữ liệu
print("Bắt đầu tải dữ liệu")
X_train = load_dataset(
    read_image_paths(
        os.path.join('data', 'FashionMNIST', 'images', 'train-images')
    )
)
X_test = load_dataset(
    read_image_paths(
        os.path.join('data', 'FashionMNIST', 'images', 't10k-images')
    )
)
y_train = np.array(
    read_labels(
        os.path.join('data', 'FashionMNIST', 'labels', 'train-labels.csv')
    )
)
y_test = np.array(
    read_labels(
        os.path.join('data', 'FashionMNIST', 'labels', 't10k-labels.csv')
    )
)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print("Tải xong dữ liệu")

print("Số chiều dữ liệu huấn luyện: ", X_train.shape)
print("Số chiều dữ liệu kiểm thử: ", X_test.shape)

# classifier - tạo đối tượng phân lớp sử dụng thuật toán Gaussian Naive Bayes.
clf = GaussianNB()

print("Bắt đầu huấn luyện mô hình")
# Huấn luyện mô hình
start_time_seconds = time.time()
clf.fit(X_train, y_train)
end_time_seconds = time.time()
print("Kết thúc huấn luyện mô hình")
print("Thời gian huấn luyện mô hình: " + str(end_time_seconds - start_time_seconds) + ' giây')

# Đánh giá mô hình
print("Bắt đầu kiểm thử mô hình")
start_time_seconds = time.time()
y_pred = clf.predict(X_test)
end_time_seconds = time.time()
print("Kết thúc kiểm thử mô hình")
print("Thời gian dự đoán kết quả trên bộ dữ liệu kiểm thử: " + str(end_time_seconds - start_time_seconds) + ' giây')

accuracy = accuracy_score(y_test, y_pred)

print("Độ chính xác:", accuracy)
