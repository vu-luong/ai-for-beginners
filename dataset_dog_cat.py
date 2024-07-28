from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
import pandas as pd  # Thư viện để đọc tập tin csv
import cv2


# Tạo một lớp data set mới thừa kế lớp Dataset của Pytorch.
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        """
        Khởi tạo các thuộc tính của data set.

        :param annotations_file: Tập tin lưu nhãn.
        :param img_dir: Thư mục chứa ảnh.
        :param transform: Hàm chuyển đổi ảnh.
        :param target_transform: Hàm chuyển đổi nhãn.
        """
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """
        Số lượng quan sát

        :param self: Con trỏ self.
        :return: Trả về số lượng quan sát.
        """
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        Lấy ra quan sát ảnh và nhãn tương ứng.

        :param self: Con trỏ self.
        :param idx: Chỉ số nhãn quan sát.
        :return: Trả về ảnh và nhãn quan sát.
        """
        # Tạo đường dẫn ảnh.
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])

        # Đọc dữ liệu ảnh
        image = cv2.imread(img_path)

        # Lấy nhãn của ảnh
        label = self.img_labels.iloc[idx, 1]

        # Chuyển đổi ảnh nếu có hàm transform được truyền vào.
        if self.transform:
            image = self.transform(image)

        # Chuyển đổi nhãn nếu hàm target_transform được truyền vào.
        if self.target_transform:
            label = self.target_transform(label)

        # Trả về ảnh và nhãn.
        return image, label


# Tạo lớp data set.
dataset = CustomImageDataset(
    annotations_file='dog_cat_labels.csv',
    img_dir='images'
)

# Đặt tên cho nhãn.
labels = {
    0: 'dog',
    1: 'cat'
}

# Tạo ảnh để hiển thị cho trực quan
figure = plt.figure(figsize=(20, 10))
for i in range(dataset.__len__()):
    x, y = dataset[i]
    figure.add_subplot(1, 5, i + 1)
    plt.imshow(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
    plt.title(labels[y])

plt.savefig("images/dog_cat_dataset_overview.png")
plt.show()
