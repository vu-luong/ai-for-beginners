import matplotlib.pyplot as plt
import numpy as np

# Tạo dữ liệu cho hàm liên tục y = x
x_continuous = np.linspace(-10, 10, 400)
y_continuous = x_continuous

# Tạo dữ liệu cho hàm không liên tục y = 1/x, tránh chia cho 0
x_discontinuous = np.linspace(-10, 10, 400)
y_discontinuous = np.where(x_discontinuous != 0, 1/x_discontinuous, np.nan)

# Thiết lập đồ thị
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Đồ thị hàm liên tục y = x
ax1.plot(x_continuous, y_continuous, label="y = x", color='blue')
ax1.axhline(0, color='black',linewidth=0.5)
ax1.axvline(0, color='black',linewidth=0.5)
ax1.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
ax1.set_title('Hàm liên tục y = x')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend()

# Đồ thị hàm không liên tục y = 1/x
ax2.plot(x_discontinuous, y_discontinuous, label="y = 1/x", color='red')
ax2.axhline(0, color='black',linewidth=0.5)
ax2.axvline(0, color='black',linewidth=0.5)
ax2.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
ax2.set_title('Hàm không liên tục y = 1/x')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.legend()

plt.show()
