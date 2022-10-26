import os.path as osp
import json
from PIL import Image
from torchvision import transforms
import torch
import cv2 as cv
from dataset.randaugment import RandAugmentMC


class TransformFixMatch(object):
    def __init__(self, mean, std, resolution=32):
        self.weak = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=resolution,
                                  padding=int(resolution*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=resolution,
                                  padding=int(resolution*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)



normal_mean = [0.485, 0.456, 0.406]
normal_std = [0.229, 0.224, 0.225]


def split_multilabel_data(data):
    pass


def get_voc07(args, root):
    resolution = 224
    transform_labeled = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=1, saturation=0.1, hue=0.5),
        #transforms.RandomCrop(size=resolution,
        #                      padding=int(resolution*0.125),
        #                      padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)
    ])
    transform_val = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)
    ])
    base_dataset = MultiLabelClassification(osp.join(root, 'mlc_voc/train.json'), transform=transform_labeled)

    '''
    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    '''
    test_dataset = MultiLabelClassification(osp.join(root, 'mlc_voc/val.json'), transform=transform_val)

    return base_dataset, [], test_dataset


class MultiLabelClassification:
    """Multi label classification dataset.
    """

    def __init__(self, root='', transform=None, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.data_dir = osp.dirname(self.root)
        self.transform = transform

        data, classes = self.load_annotation(
            self.root,
            self.data_dir,
        )
        self.num_classes = len(classes)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, target = self.data[index]
        targets = torch.zeros(self.num_classes)
        for obj in target:
            targets[obj] = 1

        img = cv.cvtColor(cv.imread(img_path, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        #if self.target_transform is not None:
        #    target = self.target_transform(target)
        return img, targets

    def load_annotation(self, annot_path, data_dir):
        out_data = []
        with open(annot_path) as f:
            annotation = json.load(f)
            classes = sorted(annotation['classes'])
            class_to_idx = {classes[i]: i for i in range(len(classes))}
            images_info = annotation['images']
            img_wo_objects = 0
            for img_info in images_info:
                rel_image_path, img_labels = img_info
                full_image_path = osp.join(data_dir, rel_image_path)
                labels_idx = [class_to_idx[lbl] for lbl in img_labels if lbl in class_to_idx]
                assert full_image_path
                if not labels_idx:
                    img_wo_objects += 1
                out_data.append((full_image_path, tuple(labels_idx)))
        if img_wo_objects:
            print(f'WARNING: there are {img_wo_objects} images without labels and will be treated as negatives')
        return out_data, class_to_idx