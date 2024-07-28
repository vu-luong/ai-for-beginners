import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


# Tải và xử lý một ảnh.
def preprocess_image(image_path, size=(150, 150)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, size)
    return image.flatten()


# Tải và xử lý bộ dữ liệu ảnh.
def load_dataset(image_paths):
    images = []
    for path in image_paths:
        images.append(preprocess_image(path))
    return np.array(images)


# Bộ dữ liệu ảnh dùng cho huấn luyện
X_train = load_dataset([
    os.path.join("images", "cat1.jpeg"),
    os.path.join("images", "cat2.jpeg"),
    os.path.join("images", "cat3.jpeg"),
    os.path.join("images", "dog1.jpeg"),
    os.path.join("images", "dog2.jpeg"),
    os.path.join("images", "dog3.jpeg")
])

# Bộ dữ liệu ảnh dùng cho kiểm thử
X_test = load_dataset([
    os.path.join("images", "cat4.jpeg"),
    os.path.join("images", "dog4.jpeg")
])

print("Số chiều dữ liệu huấn luyện: ", X_train.shape)
print("Số chiều dữ liệu kiểm thử: ", X_test.shape)

# Áp dụng PCA để giảm chiều dữ liệu
print("Bắt đầu giảm chiều dữ liệu")
pca = PCA(n_components=2)  # Chọn số chiều mới là 2
reduced_X_train = pca.fit_transform(X_train)
reduced_X_test = pca.fit_transform(X_test)
print("Kết thúc giảm chiều dữ liệu")

print("Số chiều dữ liệu huấn luyện sau khi giảm: ", reduced_X_train.shape)
print("Số chiều dữ liệu kiểm thử sau khi giảm: ", reduced_X_test.shape)

DOG = 0
CAT = 1

# Các nhãn dùng cho huấn luyện.
y_train = np.array([CAT, CAT, CAT, DOG, DOG, DOG])

# Các nhãn dành cho kiểm thử
y_test = np.array([CAT, CAT])

# Khởi tạo và tiến hành huấn luyện
svm_model = SVC(kernel='linear')
svm_model.fit(reduced_X_train, y_train)

# Đoán nhãn cho bộ ảnh kiểm thử
y_pred = svm_model.predict(reduced_X_test)

print("Các nhãn thực tế: ", y_test)
print("Các nhãn được đoán: ", y_pred)

label_name_by_id = {
    CAT: "Chú mèo",
    DOG: "Chú chó"
}
prediction_strings = [label_name_by_id[prediction] for prediction in y_pred]

print("Dự đoán các bức ảnh lần lượt là:", prediction_strings)

# Đánh giá độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print("Độ chính xác:", accuracy)
