import numpy as np
import matplotlib.pyplot as plt


# Tạo dữ liệu cho phân phối chuẩn Gaussian
mu, sigma = 0, 1  # Giá trị trung bình và độ lệch chuẩn
# Tạo ra một mảng các giá trị có khoảng cách đều nhau trong một khoảng cố định.
# Bắt đầu từ mu - 4 * sigma
# Dừng lại tại mu + 4 * sigma
x_gaussian = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)

# Tính giá trị của các phân phối chuẩn
y_gaussian = ((1 / (np.sqrt(2 * np.pi * sigma ** 2)))
              * np.exp(-((x_gaussian - mu) ** 2) / (2 * sigma ** 2)))

# Vẽ đồ thị phân phối chuẩn Gaussian
plt.figure(figsize=(10, 6))
plt.plot(x_gaussian, y_gaussian, label='Phân phối chuẩn Gaussian', color='red')
plt.fill_between(x_gaussian, y_gaussian, alpha=0.2, color='red')
plt.title('Phân phối Chuẩn Gaussian')
plt.xlabel('Giá trị')
plt.ylabel('Mật độ Xác suất')
plt.legend()
plt.grid(True)
plt.show()
