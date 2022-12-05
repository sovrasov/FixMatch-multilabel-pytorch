import os.path as osp
import json
from copy import deepcopy
import random
from PIL import Image
from torchvision import transforms
import torch
import cv2 as cv
from dataset.randaugment import RandAugmentMC


class TransformFixMatchMultilabel(object):
    def __init__(self, mean, std, resolution=224):
        self.weak = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ColorJitter(brightness=0.5, contrast=1, saturation=0.1, hue=0.5),
            transforms.RandomHorizontalFlip()])
        self.strong = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=1, saturation=0.1, hue=0.5),
            RandAugmentMC(n=2, m=10)])

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class TransformBarlowTrwinsTwoCrop:
    def __init__(self, mean, std, resolution=224):
        self.strong = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=1, saturation=0.1, hue=0.5),
            RandAugmentMC(n=2, m=10)])

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        data1 = self.strong(deepcopy(x))
        data2 = self.strong(x)
        return self.normalize(data1), self.normalize(data2)


normal_mean = [0.485, 0.456, 0.406]
normal_std = [0.229, 0.224, 0.225]


def get_labels_freq(data, num_classes):
    counters = {i : 0 for i in range(num_classes)}
    freqs = deepcopy(counters)

    for record in data:
        _, labels = record
        for l in labels:
            counters[l] += 1

    for i in range(num_classes):
        freqs[i] = counters[i] / len(data)
    return freqs, counters


def split_small_subset(base_dataset, min_num_images=10):
    min_num_images = int(min_num_images)

    def get_class_records(all_records, class_idx):
        class_records_idx = []
        for i, rec in enumerate(all_records):
            if class_idx in rec[1]:
                class_records_idx.append(i)

        random.shuffle(class_records_idx)
        return class_records_idx

    def get_num_class_samples(records, class_idx):
        num_samples = 0
        for rec in records:
            if class_idx in rec[1]:
                num_samples += 1
        return num_samples

    data_records = base_dataset.data
    random.seed(6)
    sampled_records = []

    for i in range(base_dataset.num_classes):
        class_records_idxs = get_class_records(data_records, i)
        if i > 0:
            num_sampled = get_num_class_samples(sampled_records, i)
            req_samples = min_num_images - num_sampled

            for idx in class_records_idxs:
                if req_samples <= 0:
                    break
                if data_records[idx] not in sampled_records:
                    sampled_records.append(data_records[idx])
                    req_samples -= 1
        else:
            for j in range(min([len(class_records_idxs), min_num_images])):
                sampled_records.append(data_records[class_records_idxs[j]])

    class_counters = {i : get_num_class_samples(sampled_records, i) for i in range(base_dataset.num_classes)}
    print(class_counters)
    sup_dataset = deepcopy(base_dataset)
    sup_dataset.data = sampled_records
    unsup_dataset = deepcopy(base_dataset)
    unsup_dataset.data = list(set(data_records) - set(sampled_records))
    print(len(unsup_dataset), len(sup_dataset), len(base_dataset))

    return sup_dataset, unsup_dataset


def split_multilabel_data(base_dataset, unsupervised_fraction=0.5):
    data_records = base_dataset.data
    random.seed(6)
    sampled_records = random.sample([i for i in range(len(data_records))], int(len(data_records) * unsupervised_fraction))

    supervised_subset = []
    unsupervised_subset = []

    for i in range(len(data_records)):
        if i in sampled_records:
            supervised_subset.append(data_records[i])
        else:
            unsupervised_subset.append(data_records[i])

    sup_dataset = deepcopy(base_dataset)
    sup_dataset.data = supervised_subset
    unsup_dataset = deepcopy(base_dataset)
    unsup_dataset.data = unsupervised_subset

    return sup_dataset, unsup_dataset


def get_multilabel_dataset(args, root, name='mlc_voc', resolution=224):
    transform_labeled = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=1, saturation=0.1, hue=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)
    ])
    transform_val = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)
    ])
    base_dataset = MultiLabelClassification(osp.join(root, f'{name}/train.json'), transform=transform_labeled)
    print(f'Num train images: {len(base_dataset.data)}')
    train_labeled_dataset, train_unlabeled_dataset = split_small_subset(base_dataset, args.frac_labeled)
    if args.use_bt:
        train_unlabeled_dataset.transform = TransformBarlowTrwinsTwoCrop(mean=normal_mean, std=normal_std, resolution=resolution)
    else:
        train_unlabeled_dataset.transform = TransformFixMatchMultilabel(mean=normal_mean, std=normal_std, resolution=resolution)

    test_dataset = MultiLabelClassification(osp.join(root, f'{name}/val.json'), transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


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
                labels_idx = set([class_to_idx[lbl] for lbl in img_labels if lbl in class_to_idx])
                labels_idx = list(labels_idx)
                assert full_image_path
                if not labels_idx:
                    img_wo_objects += 1
                out_data.append((full_image_path, tuple(labels_idx)))
        if img_wo_objects:
            print(f'WARNING: there are {img_wo_objects} images without labels and will be treated as negatives')
        return out_data, class_to_idx