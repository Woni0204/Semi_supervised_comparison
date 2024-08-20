import numpy as np
from PIL import Image, ImageOps, ImageFilter
import random
import torch
from torchvision import transforms


def crop(img, mask, size):
    # padding height or width if smaller than cropping size
    w, h = img.size # 이미지의 너비(w)와 높이(h)를 가져옴
    padw = size - w if w < size else 0 # 이미지의 너비가 크롭 사이즈보다 작으면 패딩 값을 계산
    padh = size - h if h < size else 0 # 이미지의 높이가 크롭 사이즈보다 작으면 패딩 값을 계산
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0) # 이미지에 패딩 추가, 너비가 작으면 오른쪽에 패딩, 높이가 작으면 아래쪽에 패딩을 추가. 색상은 검정(0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255) # 마스크에 패딩 추가, 패딩 색상은 흰색(255)

    # cropping
    w, h = img.size # 이미지의 너비(w)와 높이(h)를 가져옴
    x = random.randint(0, w - size) # 크롭 시작 위치(x, y)를 랜덤하게 설정
    y = random.randint(0, h - size) # 크롭 시작 위치(x, y)를 랜덤하게 설정
    img = img.crop((x, y, x + size, y + size)) # 이미지를 크롭하여 지정된 사이즈로 만듦.
    mask = mask.crop((x, y, x + size, y + size)) # 마스크를 크롭하여 지정된 사이즈로 만듦.

    return img, mask # 최종 이미지와 최종 마스크 반환


def hflip(img, mask, p=0.5):
    if random.random() < p: # 주어진 확률 p(default=0.5)에 따라 이미지를 좌우 반전
        img = img.transpose(Image.FLIP_LEFT_RIGHT) # 이미지를 좌우 반전
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT) # 마스크를 좌우 반전
    return img, mask # 최종 이미지와 최종 마스크 반환


def normalize(img, mask=None):
    """
    :param img: PIL image
    :param mask: PIL image, corresponding mask
    :return: normalized torch tensor of image and mask
    """
    img = transforms.Compose([
        transforms.ToTensor(), # 이미지 데이터를 Tensor로 변환
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # 이미지 정규화
    ])(img)
    if mask is not None: # 마스크가 존재하는 경우
        mask = torch.from_numpy(np.array(mask)).long() # 마스크를 numpy 배열로 변환한 후 Tensor로 변환
        return img, mask # 최종 이미지와 최종 마스크 반환
    return img # 최종 이미지 반환


def resize(img, mask, base_size, ratio_range):
    w, h = img.size # 이미지의 너비(w)와 높이(h)를 가져옴
    long_side = random.randint(int(base_size * ratio_range[0]), int(base_size * ratio_range[1])) # 길이 변수를 랜덤하게 설정

    if h > w: # 세로가 더 긴 경우
        oh = long_side # 세로를 길이 변수로 설정
        ow = int(1.0 * w * long_side / h + 0.5) # 가로를 비율에 맞게 조정
    else:
        ow = long_side # 가로를 길이 변수로 설정
        oh = int(1.0 * h * long_side / w + 0.5) # 세로를 비율에 맞게 조정

    img = img.resize((ow, oh), Image.BILINEAR) # 이미지를 BILINEAR 보간법으로 resize
    mask = mask.resize((ow, oh), Image.NEAREST) # 마스크를 NEAREST 보간법으로 resize
    return img, mask # 최종 이미지와 마스크 반환


def blur(img, p=0.5):
    if random.random() < p: # 주어진 확률 p에 따라 이미지를 blur 처리
        sigma = np.random.uniform(0.1, 2.0) # blur 정도를 랜덤하게 설정
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma)) # GaussianBlur를 적용
    return img # 최종 이미지 반환


def cutout(img, mask, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3,
           ratio_2=1/0.3, value_min=0, value_max=255, pixel_level=True):
    if random.random() < p: # 주어진 확률 p에 따라 이미지의 일부분을 제거
        img = np.array(img) # 이미지를 numpy 배열로 변환
        mask = np.array(mask) # 마스크를 numpy 배열로 변환

        img_h, img_w, img_c = img.shape # 이미지 높이, 너비, 채널 수를 가져옴

        while True: # 반복문을 통해 적절한 크기의 영역을 찾음
            size = np.random.uniform(size_min, size_max) * img_h * img_w # 제거할 영역의 크기를 랜덤하게 설정
            ratio = np.random.uniform(ratio_1, ratio_2) # 영역의 비율을 랜덤하게 설정
            erase_w = int(np.sqrt(size / ratio)) # 영역의 너비 계산
            erase_h = int(np.sqrt(size * ratio)) # 영역의 높이 계산
            x = np.random.randint(0, img_w) # 영역의 시작 위치 (x, y)를 랜덤하게 설정
            y = np.random.randint(0, img_h) # 영역의 시작 위치 (x, y)를 랜덤하게 설정

            if x + erase_w <= img_w and y + erase_h <= img_h: # 영역이 이미지 내부에 완전히 포함될 때까지 반복
                break

        if pixel_level: # 픽셀 레벨에서 값을 설정하는 경우
            value = np.random.uniform(value_min, value_max, (erase_h, erase_w, img_c)) # 영역에 채울 값을 랜덤하게 설정
        else:
            value = np.random.uniform(value_min, value_max) # 단일 값으로 설정

        img[y:y + erase_h, x:x + erase_w] = value # 이미지의 해당 영역을 새로운 값으로 대체
        mask[y:y + erase_h, x:x + erase_w] = 255 # 마스크의 해당 영역을 흰색(255)으로 설정

        img = Image.fromarray(img.astype(np.uint8)) # numpy 배열을 다시 이미지로 변환
        mask = Image.fromarray(mask.astype(np.uint8)) # numpy 배열을 다시 이미지(마스크)로 변환

    return img, mask # 최종 이미지와 마스크 반환
