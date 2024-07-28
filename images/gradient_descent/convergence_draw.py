import torch
import matplotlib.pyplot as plt


# Hàm cần tối ưu
def f(x):
    return x * x - 2 * x + 5


# Tốc độ học
learning_rate = 0.1

# Biến cần tối ưu, với khả năng tính gradient
x = torch.rand(1, requires_grad=True)

# Khởi tạo bộ tối ưu hóa SGD với tốc độ học
optimizer = torch.optim.SGD([x], lr=learning_rate)

# Danh sách để lưu trữ giá trị của x qua các bước
x_values = []

# Số lượng bước lặp
num_iterations = 100

# Vòng lặp tối ưu hóa
for iter in range(num_iterations):
    optimizer.zero_grad()  # Đặt lại gradient
    out = f(x)  # Tính toán giá trị của hàm
    out.backward()  # Tính gradient
    optimizer.step()  # Cập nhật giá trị của x
    x_values.append(x.detach().item())  # Lưu trữ giá trị của x

# Vẽ biểu đồ
plt.plot(range(num_iterations), x_values, marker='o', linestyle='-', color='b', label='x qua từng bước')
plt.axhline(y=1.0, color='r', linestyle='--', label='Điểm mong muốn (x=1.0)')
plt.xlabel('Bước lặp')
plt.ylabel('Giá trị x')
plt.title('Quá trình hội tụ của x trong Tối ưu hóa')
plt.legend()
plt.show()
