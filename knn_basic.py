import numpy as np
from collections import Counter


# Tính khoảng cách Euclidean giữa hai điểm
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


# Lớp KNN cơ bản
class KNN:
    def __init__(self, k=3):
        self.k = k

    # Huấn luyện KNN bằng cách chỉ cần lưu trữ dữ liệu huấn luyện
    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    # Dự đoán nhãn cho tập dữ liệu mới
    def predict(self, X_test):
        predictions = []
        for point in X_test:
            # Tính khoảng cách Euclidean từ điểm mới tới mọi điểm trong tập huấn luyện
            distances = [euclidean_distance(point, train_point) for train_point in self.X_train]
            # Lấy 'k' điểm gần nhất
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            # Lấy nhãn phổ biến nhất trong 'k' điểm gần nhất
            most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common_label)
        return predictions


# Dữ liệu huấn luyện
X_train = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [6, 7],
    [7, 8],
    [8, 9]
])
y_train = np.array(['A', 'A', 'A', 'B', 'B', 'B'])

# Tạo một đối tượng KNN với 'k' là 3
knn = KNN(k=3)

# Huấn luyện KNN
knn.fit(X_train, y_train)

# Dữ liệu kiểm tra
X_test = np.array([
    [3, 3],
    [7, 7],
    [2, 2]
])

# Dự đoán
predictions = knn.predict(X_test)
print("Dự đoán:", predictions)
