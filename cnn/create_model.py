import torch # нахуй tensorflow багованый

from torch import nn 
from torch.utils.data import DataLoader # загрузчик данных

from torchvision import datasets # импорт публичных датаестов
from torchvision.transforms import ToTensor, Compose, Normalize # totensor преобразует PIL картинку (array [0,255]) в тензор (CxHxW) 
# normalize - приведение в один рендж

from config import *

# загрузка датасета mnist с циферками и предварительное разделение на test/train
# if not os.path.isdir("data/train"):
training_data = datasets.MNIST(
    root="data/train",
    train=True,
    download=True,
    transform=ToTensor()
)

# if not os.path.isdir("data/test"):
test_data = datasets.MNIST(
    root="data/test",
    train=False,
    download=True,
    transform=ToTensor()
)

# ---- загрузка данных и преобразование ----

# выходной датасет в тензор -> нормализация (среднее отклонение и стандартное отклонение MNIST - 0.1307, 0.3081)
trans = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]) 

train_dataset = datasets.MNIST(root=TRAIN_DATA_PATH, train=True, transform=trans, download=True) 
test_dataset = datasets.MNIST(root=TEST_DATA_PATH, train=False, transform=trans)

train_dataloader = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# вывод информации о датасете (размеры и типы данных)
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break # break чтобы не выводило оч много в консоль

# выбор акселератора (cpu/gpu)
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"используем {device}")

# определение модели на основе nn.Module
class CNN(nn.Module):
    # в pytorch слои сетки определяются в __init__
    def __init__(self):
        super(CNN, self).__init__() #создание обьекта базового класса

        # sequential - 
        self.layer1 = nn.Sequential(
            # в Conv2d() первый аргумент - колво входных каналов
            # второй аргумент - колво выходных каналов
            # kernel_size = размер сверточного фильтра
            # stride = шаг свертки
            # padding = ?
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),

            # функция активации
            nn.ReLU(),

            # в MaxPool2d() первый аргумент - размер обьединения
            # stride - шаг обьединения
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            # второй слой self.layer2 определяется также как и первый
            # единственное отличие здесь — вход в функцию Conv2d теперь 32 канальный, а выход — 64 канальный. 
            # следуя такой же логике и учитывая пулинг и даунсемплинг, выход из self.layer2 представляет из себя 64 канала изображения размера 7х7.
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.drop_out = nn.Dropout()
        # слои для предотвращения переобучения
        self.fc1 = nn.Linear(7 * 7  * 64, 1000)
        self.fc2 = nn.Linear(1000,10)
    
    # метод для определения принципов распространения данных по слоям сети
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

model = CNN() # создание экземпляра модели с указанной архитектурой выше
print(model)

# model.state_dict() сохраняет ток веса
# torch.save(model, store_path + 'cnn_model.pt') 

def load_model(wpath: str = "cnn/cnn_model.pt") -> torch.nn.Module:
    model = CNN().to(device)
    state_dict = torch.load(wpath, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

if __name__ == "__main__":
    criterion = nn.CrossEntropyLoss() # функция подсчета потерь

    # выбран метод оптимизации через оптимизатор Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # lr = learning rate

    total_step = len(train_dataloader)
    loss_list = []
    acc_list = []

    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_dataloader):
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            optimizer.zero_grad() # обнуление градиентов
            loss.backward()
            optimizer.step()

            # отслеживание точности
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                    .format(epoch + 1, epochs, i + 1, total_step, loss.item(),
                            (correct / total) * 100))
                
    torch.save(model.state_dict(), store_path + 'cnn_model.pt')

    model.eval() # режим оценки

    # с выключенным автоградиентом тк он не нужен при тесте модели
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"точность модели на 10000 тестовых пикчах: {(correct / total) * 100}%")