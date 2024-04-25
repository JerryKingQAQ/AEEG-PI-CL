import numpy as np
import torch
from torchvision import datasets, transforms

from utils.toolkit import split_images_labels


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self, args):
        train_dataset = datasets.cifar.CIFAR10("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self, args):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self, args):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iTHUPIDatasets(iData):
    use_path = False
    train_trsf = [
        transforms.ToTensor()
    ]
    test_trsf = [
        transforms.ToTensor()
    ]
    common_trsf = [
        transforms.ToTensor()
    ]

    class_order = np.arange(100).tolist()

    def __init__(self, args):
        self.class_order = np.arange(80).tolist()
        self.emotion = args["emotion"]
        self.test_form = args["test_form"]

    def download_data(self, args):
        train_data = torch.load(f'./data/TsinghuaPIDatasets/{self.test_form}/train/{self.emotion}_data.pth')
        train_labes = torch.load(f'./data/TsinghuaPIDatasets/{self.test_form}/train/{self.emotion}_labels.pth')
        test_data = torch.load(f'./data/TsinghuaPIDatasets/{self.test_form}/test/{self.emotion}_data.pth')
        test_labes = torch.load(f'./data/TsinghuaPIDatasets/{self.test_form}/test/{self.emotion}_labels.pth')
        frequency = args['frequency']
        frequencies = {"delta": 0, "theta": 1, "alpha": 2, "beta": 3, "gamma": 4, "broad": 5}
        if frequency == 'all':
            self.train_data = train_data
            self.train_targets = train_labes
            self.test_data = test_data
            self.test_targets = test_labes
        else:
            self.train_data = train_data[:, :, :, frequencies[frequency]]
            self.train_data = self.train_data.reshape(*self.train_data.shape, 1)
            self.train_targets = train_labes
            self.test_data = test_data[:, :, :, frequencies[frequency]]
            self.test_data = self.test_data.reshape(*self.test_data.shape, 1)
            self.test_targets = test_labes


class iTHUPI30Datasets(iData):
    use_path = False
    train_trsf = [
        transforms.ToTensor()
    ]
    test_trsf = [
        transforms.ToTensor()
    ]
    common_trsf = [
        transforms.ToTensor()
    ]

    class_order = np.arange(100).tolist()

    def __init__(self, args):
        self.class_order = np.arange(30).tolist()
        self.emotion = args["emotion"]
        self.test_form = args["test_form"]

    def download_data(self, args):
        train_data = np.load(f'./data/Tsinghua30PIDatasets/{self.emotion}/train_dataset.npy')
        train_labes = np.load(f'./data/Tsinghua30PIDatasets/{self.emotion}/train_label.npy')
        test_data = np.load(f'./data/Tsinghua30PIDatasets/{self.emotion}/test_dataset.npy')
        test_labes = np.load(f'./data/Tsinghua30PIDatasets/{self.emotion}/test_label.npy')
        frequency = args['frequency']
        frequencies = {"delta": 0, "theta": 1, "alpha": 2, "beta": 3, "gamma": 4, "broad": 5}
        if frequency == 'all':
            self.train_data = train_data
            self.train_targets = train_labes
            self.test_data = test_data
            self.test_targets = test_labes
        else:
            self.train_data = train_data[:, :, :, frequencies[frequency]]
            self.train_data = self.train_data.reshape(*self.train_data.shape, 1)
            self.train_targets = train_labes
            self.test_data = test_data[:, :, :, frequencies[frequency]]
            self.test_data = self.test_data.reshape(*self.test_data.shape, 1)
            self.test_targets = test_labes


class iTHU_SEED_PIDatasets(iData):
    use_path = False
    train_trsf = [
        transforms.ToTensor()
    ]
    test_trsf = [
        transforms.ToTensor()
    ]
    common_trsf = [
        transforms.ToTensor()
    ]

    class_order = np.arange(100).tolist()

    def __init__(self, args):
        self.class_order = np.arange(30).tolist()
        self.emotion = args["emotion"]
        self.test_form = args["test_form"]

    def download_data(self, args):
        train_data = np.load(f'./data/TsinghuaSEEDPIDatasets/{self.emotion}/train_dataset.npy')
        train_labes = np.load(f'./data/TsinghuaSEEDPIDatasets/{self.emotion}/train_label.npy')
        test_data = np.load(f'./data/TsinghuaSEEDPIDatasets/{self.emotion}/test_dataset.npy')
        test_labes = np.load(f'./data/TsinghuaSEEDPIDatasets/{self.emotion}/test_label.npy')
        frequency = args['frequency']
        frequencies = {"delta": 0, "theta": 1, "alpha": 2, "beta": 3, "gamma": 4, "broad": 5}
        if frequency == 'all':
            self.train_data = train_data
            self.train_targets = train_labes
            self.test_data = test_data
            self.test_targets = test_labes
        else:
            self.train_data = train_data[:, :, :, frequencies[frequency]]
            self.train_data = self.train_data.reshape(*self.train_data.shape, 1)
            self.train_targets = train_labes
            self.test_data = test_data[:, :, :, frequencies[frequency]]
            self.test_data = self.test_data.reshape(*self.test_data.shape, 1)
            self.test_targets = test_labes
