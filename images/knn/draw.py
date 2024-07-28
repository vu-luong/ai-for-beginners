import numpy as np
import matplotlib.pyplot as plt

X_train = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [6, 7],
    [7, 8],
    [8, 9]
])
y_train = np.array(['A', 'A', 'A', 'B', 'B', 'B'])

X_test = np.array([
    [3, 3],
    [7, 7],
    [2, 2]
])

# Vẽ các điểm thuộc lớp A
plt.scatter(X_train[:3, 0], X_train[:3, 1], c='blue', marker='o', label='Dữ liệu huấn luyện - Lớp A', s=50)

# Vẽ các điểm thuộc lớp B
plt.scatter(X_train[3:, 0], X_train[3:, 1], c='red', marker='o', label='Dữ liệu huấn luyện - Lớp B', s=50)

# Thêm chú thích cho từng điểm
for i, label in enumerate(y_train):
    plt.text(X_train[i, 0] + 0.1, X_train[i, 1] + 0.1, f'${label}_{i+1}$', fontsize=9)

# Vẽ các điểm cần dự đoán
plt.scatter(X_test[:, 0], X_test[:, 1], c='green', marker='s', label='Dữ liệu kiểm thử', s=50)

# Thêm chú thích cho từng điểm cần dự đoán
for i in range(len(X_test)):
    plt.text(X_test[i, 0] + 0.1, X_test[i, 1] + 0.1, f'$X_{i+1}$', fontsize=9)

# Thêm nhãn, tiêu đề, và chú thích 
plt.xlabel('x')
plt.ylabel('y')
plt.title('Minh họa thuật toán KNN')
plt.legend()

# Vẽ hình
plt.savefig('draw.png')
plt.show()
