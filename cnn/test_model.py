from create_model import *
import torch # нахуй tensorflow багованый

from config import *

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
