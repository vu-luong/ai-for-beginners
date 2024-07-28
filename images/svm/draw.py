import numpy as np
import matplotlib.pyplot as plt

# Tạo dữ liệu mẫu
X = np.array([
    [1, -2], [2, -1], [4, -1], [4.5, 2.085786438], 
    [5.5, 3.085786438], [6, 0], [3, 6], [4, 4.414213562],
    [4.5, 7], [5, 6], [6, 7]
])
y = np.array([1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1])

# Tìm hai điểm dữ liệu thuộc hai lớp
class1 = X[y == 1]
class2 = X[y == -1]

# Tìm đường lề
w = np.array([0.5, -0.5])  # Vectơ hệ số của đường lề
b = -0.5  # Hệ số chặn của đường lề

# Tìm hai điểm trên đường lề
x1 = np.linspace(0, 7, 100)
x2 = (-w[0] * x1 - b) / w[1]

# Tìm hai đường song song
margin = 1 / np.linalg.norm(w)
parallel_line1 = x2 + margin
parallel_line2 = x2 - margin

# Vẽ dữ liệu
plt.scatter(class1[:, 0], class1[:, 1], color='red', label='Lớp 1')
plt.scatter(class2[:, 0], class2[:, 1], color='blue', label='Lớp 2')

# Vẽ đường lề và đường song song
plt.plot(x1, x2, '-', label='Đường phân lớp')
plt.plot(x1, parallel_line1, 'g--', label='Lề 1')
plt.plot(x1, parallel_line2, 'g--', label='Lề 2')

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

plt.savefig('draw.png')
plt.show()

