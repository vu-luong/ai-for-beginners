import numpy as np
import matplotlib.pyplot as plt

# Hàm số f(x) = x^2 - 2x + 5
def f(x):
    return x**2 - 2*x + 5

# Đạo hàm của f(x)
def f_prime(x):
    return 2*x - 2

# Thực hiện Gradient Descent
learning_rate = 0.5  # Tốc độ học
iterations = 5  # Số lần lặp
x = 2.5  # Điểm khởi đầu

# Để lưu lại các giá trị x qua các lần lặp
x_values = [x]

for _ in range(iterations):
    gradient = f_prime(x)  # Tính gradient
    x_new = x - learning_rate * gradient  # Cập nhật x
    x_values.append(x_new)
    x = x_new

# Giá trị y tương ứng với các giá trị x
y_values = [f(x) for x in x_values]

# Vẽ đồ thị của hàm parabol
x_range = np.linspace(-1, 3, 100)  # Tạo một dải giá trị x để vẽ
y_range = f(x_range)  # Tạo dải giá trị y tương ứng

plt.figure(figsize=(8, 6))
plt.plot(x_range, y_range, label="f(x) = x^2 - 2x + 5")  # Đường parabol
plt.plot(x_values, y_values, 'o-', color='red', label="Gradient Descent")  # Các điểm hội tụ

# Vẽ mũi tên biểu thị hướng của Gradient Descent
for i in range(1, len(x_values)):
    # Vẽ mũi tên từ điểm x_values[i-1] tới x_values[i]
    plt.quiver(x_values[i - 1], y_values[i - 1], 
               x_values[i] - x_values[i - 1], 
               y_values[i] - y_values[i - 1], 
               angles='xy', scale_units='xy', scale=1, color='blue', label='Hướng Gradient Descent' if i == 1 else "")

plt.title("Sử dụng Gradient Descent")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()

