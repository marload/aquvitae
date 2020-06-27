import torch
import torchvision
import torchvision.transforms as transforms
from aquvitae import dist, ST

# TEACHER_WEIGHTS_PATH = ...  # Teacher Weights PATH
BATCH_SIZE = 64
INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10

# Load the dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
train_ds = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
test_ds = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=False, num_workers=2
)

# Load the teacher and student model

# teacher's output activation function must be `None`.
teacher = torchvision.models.resnet50()
# student's output activation function must be `None`.
student = torchvision.models.mobilenet_v2()
# teacher.load_state_dict(torch.load(TEACHER_WEIGHTS_PATH))

student = dist(
    teacher=teacher,
    student=student,
    algo=ST(alpha=0.6, T=2.5),  # Knowledge Distillation Algorithm Instance
    optimizer=torch.optim.Adam(student.parameters()),
    # `train_ds` must be an Instance of `torch.utils.data.DataLoader`.
    train_ds=train_ds,
    # `test_ds` must be an Instance of `torch.utils.data.DataLoader`.
    test_ds=test_ds,
    iterations=3000,
)
