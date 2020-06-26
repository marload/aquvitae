import torch
import torchvision
from aquvitae import dist, ST

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

trainset = torchvision.datasets.CIFAR100(
    root="./data", train=True, download=True, transform=transform
)
train_ds = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR100(
    root="./data", train=False, download=True, transform=transform
)
test_ds = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2
)


teacher = torchvision.models.resnet50(num_classes=100)
student = torchvision.models.resnet50(num_classes=100)

optimizer = torch.optim.Adam(student.parameters())
st = ST(alpha=0.0, T=2.5)

student = dist(
    teacher=teacher,
    student=student,
    algo=st,
    optimizer=optimizer,
    train_ds=train_ds,
    test_ds=test_ds,
    iterations=50000,
)
