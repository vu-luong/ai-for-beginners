import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import time

learning_rate = 1e-3
batch_size = 64
epochs = 5

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

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
model = CNN().to(device)

def train_loop(dataloader, model, loss_fn, optimizer):
    n_samples = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss_item, current = loss.item(), batch * len(X)
            print(f"Mất mát huấn luyện: {loss_item:>7f}  [{current:>5d}/{n_samples:>5d}]")

def test_loop(dataloader, model, loss_fn):
    n_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= n_samples
    print(f"Tỉ lệ dự đoán chính xác: {(100 * correct):>0.1f}%, mất mát trung bình: {test_loss:>8f} \n")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_train_time = 0
total_test_time = 0
for t in range(epochs):
    print(f"Chu kỳ huấn luyện {t + 1}\n-------------------------------")
    start_time_seconds = time.time()
    train_loop(train_dataloader, model, loss_fn, optimizer)
    end_time_seconds = time.time()
    total_train_time += end_time_seconds - start_time_seconds
    start_time_seconds = time.time()
    test_loop(test_dataloader, model, loss_fn)
    end_time_seconds = time.time()
    total_test_time += end_time_seconds - start_time_seconds

print("Hoàn thành!")
print("Thời gian huấn luyện mô hình: " + str(total_train_time) + ' giây')
print("Thời gian dự đoán kết quả trên bộ dữ liệu kiểm thử: " + str(total_test_time / epochs) + ' giây')

torch.save(model.state_dict(), "model.pth")
