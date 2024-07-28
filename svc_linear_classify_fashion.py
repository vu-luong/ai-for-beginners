import cv2
import csv
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
from sklearn.decomposition import PCA


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


# Tải và xử lý ảnh.
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
X_train /= 255.0
X_test /= 255.0

print("Tải xong dữ liệu")

print("Số chiều dữ liệu huấn luyện: ", X_train.shape)
print("Số chiều dữ liệu kiểm thử: ", X_test.shape)

print("Bắt đầu giảm chiều dữ liệu")
start_time_seconds = time.time()
pca = PCA(n_components=100, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
end_time_seconds = time.time()
print("Kết thúc giảm chiều dữ liệu")
print("Thời gian giảm chiều dữ liệu: " + str(end_time_seconds - start_time_seconds) + ' giây')

print("Số chiều dữ liệu huấn luyện sau khi giảm: ", X_train_pca.shape)
print("Số chiều dữ liệu kiểm thử sau khi giảm: ", X_test_pca.shape)

# Tạo đối tượng huấn luyện
# C=13: Tham số này xác định regularization parameter (tham số điều chuẩn) của mô hình SVC.
# Giá trị của C càng cao, mô hình sẽ cố gắng fit tốt hơn với dữ liệu huấn luyện, nhưng cũng có thể dẫn đến overfitting.
# kernel=linear: Sử dụng nhân tuyến tính để tìm ra siêu phẳng.
svm_model = SVC(C=13, kernel='linear')

print("Bắt đầu huấn luyện mô hình")
start_time_seconds = time.time()
svm_model.fit(X_train_pca, y_train)
end_time_seconds = time.time()
print("Kết thúc huấn luyện mô hình")
print("Thời gian huấn luyện mô hình: " + str(end_time_seconds - start_time_seconds) + ' giây')

# Predict labels for test set
print("Bắt đầu kiểm thử mô hình")
start_time_seconds = time.time()
y_pred = svm_model.predict(X_test_pca)
end_time_seconds = time.time()
print("Kết thúc kiểm thử mô hình")
print("Thời gian dự đoán kết quả trên bộ dữ liệu kiểm thử: " + str(end_time_seconds - start_time_seconds) + ' giây')

# Tính toán độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print("Độ chính xác là:", accuracy)
