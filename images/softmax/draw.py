import numpy as np
import matplotlib.pyplot as plt


def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=0))
    return exp_values / exp_values.sum(axis=0)


# Tạo dữ liệu đầu vào
x = np.linspace(-5, 5, 100)

# Tính toán hàm softmax cho dữ liệu đầu vào
softmax_values = softmax(x)

# Vẽ đồ thị
plt.plot(x, softmax_values)
plt.title('Đồ thị hàm softmax')
plt.xlabel('Đầu vào')
plt.ylabel('Xác suất')
plt.grid(True)
plt.show()
