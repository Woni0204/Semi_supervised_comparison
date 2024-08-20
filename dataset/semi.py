from dataset.transform import crop, hflip, normalize, resize, blur, cutout

import math
import os
from PIL import Image
import random
from torch.utils.data import Dataset
from torchvision import transforms


class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size, labeled_id_path=None, unlabeled_id_path=None, pseudo_mask_path=None):
        """
        :param name: dataset name, pascal or cityscapes
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        :param labeled_id_path: path of labeled image ids, needed in train or semi_train mode.
        :param unlabeled_id_path: path of unlabeled image ids, needed in semi_train or label mode.
        :param pseudo_mask_path: path of generated pseudo masks, needed in semi_train mode.
        """
        self.name = name # 데이터셋 이름
        self.root = root # 데이터셋 경로
        self.mode = mode # 학습 모드 (train/label/semi_train/val)
        self.size = size # 학습할 이미지를 어떠한 크기로 자를지
        # labeled_id_path : label 이미지의 경로 -> train / semi_train 모드에 필요.
        # unlabeled_id_path : unlabeled 이미지의 경로 -> semi_train / label 모드에 필요.
        # pseudo_mask_path : 만들 Pseudo mask들의 경로 -> semi_train 모드에 필요.

        self.pseudo_mask_path = pseudo_mask_path

        if mode == 'semi_train': # 모드가 Semi_train 일 때.
            with open(labeled_id_path, 'r') as f:
                self.labeled_ids = f.read().splitlines() # 각 줄마다 id를 읽어 labeled_ids 리스트에 저장
            with open(unlabeled_id_path, 'r') as f:
                self.unlabeled_ids = f.read().splitlines() # 각 줄마다 id를 읽어 unlabeled_ids 리스트에 저장
            # ids 리스트는 labeled_ids를 반복하여 unlabeled_ids 의 길이보다 길거나 같게 만든 후, unlabeled_ids 를 덧붙인 것.
            self.ids = \
                self.labeled_ids * math.ceil(len(self.unlabeled_ids) / len(self.labeled_ids)) + self.unlabeled_ids

        else:
            if mode == 'val': # 모드가 val 일 때
                id_path = 'dataset/splits/%s/val.txt' % name # id_path를 지정.
            elif mode == 'label': # 모드가 label 일 때
                id_path = unlabeled_id_path # 마찬가지로 id_path를 지정.
            elif mode == 'train': # 모드가 train 일 때
                id_path = labeled_id_path # 마찬가지로 id_path를 지정.

            with open(id_path, 'r') as f: # id_path 경로에 있는 파일을 읽기 모드('r')로 열기.
                self.ids = f.read().splitlines() # 파일의 모든 내용을 읽고, 각 줄을 나누어 리스트로 만듦.

    def __getitem__(self, item):
        id = self.ids[item] # item 인덱스에 해당하는 id를 self.ids 리스트에서 가져옴.
        img = Image.open(os.path.join(self.root, id.split(' ')[0])) # id에서 이미지 파일의 경로를 추출하여 이미지를 엶.

        if self.mode == 'val' or self.mode == 'label': # mode 가 val 혹은 label 이면
            mask = Image.open(os.path.join(self.root, id.split(' ')[1])) # id에서 마스크 파일의 경로를 추출하여 마스크 이미지를 엶.
            img, mask = normalize(img, mask) # dataset.transform 에 존재하는 normalize 를 불러와서 이미지와 마스크를 정규화시킴.
            return img, mask, id # 이미지, 마스크, id를 반환

        if self.mode == 'train' or (self.mode == 'semi_train' and id in self.labeled_ids): # 모드가 train 이거나 semi_train 모드인데 labeled_ids에 id가 포함된 경우
            mask = Image.open(os.path.join(self.root, id.split(' ')[1])) # id에서 마스크 경로를 추출하여 마스크 이미지를 엶
        else:
            # 모드가 'semi_train'이고 id가 unlabeled image에 해당하는 경우
            fname = os.path.basename(id.split(' ')[1]) # 마스크 파일 이름을 추출.
            mask = Image.open(os.path.join(self.pseudo_mask_path, fname)) # pseudo_mask 경로에서 해당 마스크를 엶.

        # 모든 이미지에 대해 기본적인 증강 수행.
        base_size = 400 if self.name == 'pascal' else 2048 # pascal 일 시 400, 아니면 2048
        img, mask = resize(img, mask, base_size, (0.5, 2.0)) # base와 scale range(0.5, 2.0) 맞춰 resize
        img, mask = crop(img, mask, self.size) # 이미지와 마스크를 size에 맞춰 crop
        img, mask = hflip(img, mask, p=0.5) # 이미지와 마스크를 50% 확률로 hflip(좌우 반전 -> horizontal flip)

        # 모드가 semi_train이고, id가 unlabeled_ids 에 포함되는 경우 strong data augmentation 진행.
        if self.mode == 'semi_train' and id in self.unlabeled_ids:
            if random.random() < 0.8: # 80% 확률로 colorjitter 수행.
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.2)(img) # 20% 확률로 randomgrayscale 진행.
            img = blur(img, p=0.5) # 50% 확률로 blur 증강 수행.
            img, mask = cutout(img, mask, p=0.5) # 50% 확률로 cutout 증강 수행.

        img, mask = normalize(img, mask) # 증강된 이미지와 마스크를 normalize 진행.

        return img, mask # 최종적인 이미지와 마스크 반환

    def __len__(self):
        return len(self.ids) # ids 의 길이 반환
