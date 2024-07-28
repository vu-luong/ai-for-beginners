import numpy as np
import matplotlib.pyplot as plt

# Hàm ReLU
def relu(x):
    return np.maximum(0, x)

# Tạo dữ liệu cho trục x
x = np.linspace(-5, 5, 100)

# Tính giá trị của hàm ReLU cho mỗi giá trị của x
y = relu(x)

# Vẽ đồ thị
plt.plot(x, y, label='ReLU', color='b')
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.title('Đồ thị của hàm ReLU')
plt.grid(True)
plt.legend()
plt.show()
