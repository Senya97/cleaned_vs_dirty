from torch.nn import CrossEntropyLoss
from torch.optim import Adam, lr_scheduler
from torchvision import models
from torchvision import transforms

DEFAULT_DATA_PATH = '../data/default/'
DATASET_PATH = '../data/dataset/'

MODEL = models.resnet18(pretrained=True)

HEIGHT = 224
WIDTH = 244

BATCH_SIZE = 8
LEARNING_RATE = 3e-4
NUM_EPOCH = 2
STEP_SIZE = 10
GAMMA = 0.1

LOSS = CrossEntropyLoss()
OPTIMIZER = Adam(MODEL.parameters(), lr=LEARNING_RATE)
SCHEDULER = lr_scheduler.StepLR(OPTIMIZER, step_size=STEP_SIZE, gamma=GAMMA)

TRANSFORM = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
