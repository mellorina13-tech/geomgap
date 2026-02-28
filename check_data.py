import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
print('Dataset size:', len(train_set))
loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
for i, (x, y) in enumerate(loader):
    print('Batch', i, x.shape, y.shape)
    if i == 2:
        break
print('Data loading successful.')