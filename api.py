import torch
from torch import nn
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, HTTPException
import os
import magic


UPLOAD_FOLDER = "upload"

fashion_mnist_labels = {
    0: "T-shirt/top: Áo thun hoặc áo trên",
    1: "Trouser: Quần dài",
    2: "Pullover: Áo len chui đầu",
    3: "Dress: Váy",
    4: "Coat: Áo khoác",
    5: "Sandal: Dép sandal",
    6: "Shirt: Áo sơ mi",
    7: "Sneaker: Giày thể thao",
    8: "Bag: Túi xách",
    9: "Ankle boot: Giày bốt cổ chân"
}


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Sử dụng thiết bị {}'.format(device))

model = CNN()
model.load_state_dict(torch.load('model.pth', map_location=device))
model.to(device)
model.eval()


def predict(image_path):
    image = Image.open(image_path)
    # Chuyển ảnh màu sang grayscale
    image_grayscale = image.convert("L")
    image_grayscale.save("upload/tshirt-gray.jpeg")
    resize_transform = transforms.Resize((28, 28))  # Định kích thước thành 28x28
    image_resized = resize_transform(image_grayscale)  # Áp dụng phép biến đổi
    tensor_transform = transforms.Compose([
        transforms.ToTensor(),  # Chuyển ảnh thành tensor
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Chuẩn hóa tương tự FashionMNIST
        transforms.Lambda(lambda x: x.unsqueeze(0))  # Thêm chiều batch_size
    ])

    image_tensor = tensor_transform(image_resized)  # Áp dụng phép biến đổi
    image_tensor = image_tensor.to(device)  # Chuyển tensor lên gpu hoặc cpu

    with torch.no_grad():
        pred = model(image_tensor)

    # Lấy lớp có xác suất cao nhất
    predicted_class = pred.argmax(1)

    return predicted_class.item()


app = FastAPI()

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.post("/api/v1/predict-image-class")
async def predict_image_class(file: UploadFile):
    # Validate the MIME type using python-magic
    mime = magic.Magic(mime=True)
    mime_type = mime.from_buffer(await file.read(1024))  # Read first 1KB to detect MIME type

    # Reset the file pointer to the beginning
    await file.seek(0)

    allowed_mime_types = ["image/jpeg", "image/png"]

    if mime_type not in allowed_mime_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {mime_type}")

    # Construct the file path
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    # Save the file
    with open(file_path, "wb") as f:
        content = await file.read()  # Read the entire file content
        f.write(content)

    label = predict(file_path)
    return {"label_id": label, "label_name": fashion_mnist_labels[label]}
