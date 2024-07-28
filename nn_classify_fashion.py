import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import time
import matplotlib.pyplot as plt


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

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(28 * 28, 512)
        self.hidden1 = nn.ReLU()
        self.layer2 = nn.Linear(512, 512)
        self.hidden2 = nn.ReLU()
        self.output_layer = nn.Linear(512, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.hidden1(x)
        x = self.layer2(x)
        x = self.hidden2(x)
        out = self.output_layer(x)
        return out


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Sử dụng thiết bị {}'.format(device))
model = NeuralNetwork().to(device)

learning_rate = 1e-3
batch_size = 64

# Khởi tạo hàm mất mát và thuật toán tối ưu
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

train_losses = []
test_losses = []
test_accuracies = []


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
    train_losses.append(loss.item())


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
    test_losses.append(test_loss)
    test_accuracies.append(100 * correct / len(test_dataloader.dataset))
    print(f"Tỉ lệ dự đoán chính xác: {(100 * correct):>0.1f}%, mất mát trung bình: {test_loss:>8f} \n")


epochs = 15
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

# Vẽ đồ thị mất mát qua các epoch
plt.plot(range(1, epochs + 1), train_losses, 'b-', label='Mất mát huấn luyện')
plt.plot(range(1, epochs + 1), test_losses, 'r-', label='Mất mát kiểm thử')
plt.xlabel("Chu kỳ")
plt.ylabel("Mất mát")
plt.title("Mất mát qua các chu kỳ")
plt.legend()
plt.show()

# Vẽ đồ thị độ chính xác qua các epoch
plt.plot(range(1, epochs + 1), test_accuracies, 'g-', label='Độ chính xác kiểm thử')
plt.xlabel("Chu kỳ")
plt.ylabel("Độ chính xác (%)")
plt.title("Độ chính xác qua các chu kỳ")
plt.legend()
plt.show()
