from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.datasets import VisionDataset

class ArtBench10(CIFAR10):
    base_folder = "artbench-10-batches-py"
    url = "https://artbench.eecs.berkeley.edu/files/artbench-10-python.tar.gz"
    filename = "artbench-10-python.tar.gz"
    tgz_md5 = "9df1e998ee026aae36ec60ca7b44960e"
    train_list = [
        ["data_batch_1", "c2e02a78dcea81fe6fead5f1540e542f"],
        ["data_batch_2", "1102a4dcf41d4dd63e20c10691193448"],
        ["data_batch_3", "177fc43579af15ecc80eb506953ec26f"],
        ["data_batch_4", "566b2a02ccfbafa026fbb2bcec856ff6"],
        ["data_batch_5", "faa6a572469542010a1c8a2a9a7bf436"],
    ]
    test_list = [
        ["test_batch", "fa44530c8b8158467e00899609c19e52"],
    ]
    meta = {
        "filename": "meta",
        "key": "styles",
        "md5": "5bdcafa7398aa6b75d569baaec5cd4aa",
    }
    
class ArtBench10_subclass(ArtBench10):
    def __init__(self, root, train=True, download=True, transform=None, target_transform=None, labels=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.labels = labels
        self.indices = [i for i, label in enumerate(self.targets) if label in self.labels]

    def __getitem__(self, idx):
        index = self.indices[idx]
        image, target = super().__getitem__(index)
        return image, target
    
    def __len__(self):
        return len(self.indices)