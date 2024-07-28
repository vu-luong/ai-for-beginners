import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root="data", # Lưu dữ liệu vào thư mục có tên data.
    train=True, # Training set hay testing set.
    download=True,
    transform=ToTensor() # Sau khi ảnh được đọc xong chúng ta có thể chuyển dữ liệu về dạng Tensor.
)

# Khai báo tên cho các nhãn.
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
# Khởi tạo khung chứa ảnh.
figure = plt.figure(figsize=(8, 8))

# Sẽ có 9 ảnh chia thành 3 hàng 3 cột.
cols, rows = 3, 3

# Duyệt để lấy ngẫu nhiên 9 ảnh.
for i in range(1, cols * rows + 1):

    # Lấy ngẫu nhiên chỉ số của một tensor.
    sample_idx = torch.randint(len(training_data), size=(1,)).item()

    # Lấy thông tin của tensor trong tập dữ liệu training 
    img, label = training_data[sample_idx]
    # In ra số chiều của tensor
    print(img.shape)

    # Thêm ảnh vào khung.
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")

# Hiển thị các ảnh.
plt.show()
