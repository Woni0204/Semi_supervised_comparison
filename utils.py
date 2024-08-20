import numpy as np
from PIL import Image


def count_params(model):
    param_num = sum(p.numel() for p in model.parameters()) # 모델의 모든 파라미터 수를 합산
    return param_num / 1e6 # 파라미터 수를 백만 단위로 변환하여 반환


class meanIOU: # mean Intersection over Union (mIoU) 평가 클래스
    def __init__(self, num_classes):
        self.num_classes = num_classes # 클래스 수 저장
        self.hist = np.zeros((num_classes, num_classes)) # 혼동 행렬 초기화

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes) # 유효한 레이블 마스크 생성
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + # 예측과 실제 레이블의 조합으로 인덱스 생성
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes) # 히스토그램을 계산 한 후 클래스 수로 reshape
        return hist

    def add_batch(self, predictions, gts): # 예측값과 실제값을 추가하여 혼동 행렬 업데이트
        for lp, lt in zip(predictions, gts): # 예측과 실제값을 순회
            self.hist += self._fast_hist(lp.flatten(), lt.flatten()) # 히스토그램을 계산하여 누적

    def evaluate(self): # IoU를 계산하여 반환하는 함수
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist)) # 각 클래스별 IoU 계산
        return iu, np.nanmean(iu) # 각 클래스의 IoU와 전체 평균 IoU 반환

# 데이터셋에 따라 컬러맵을 생성하는 함수
def color_map(dataset='pascal'):
    cmap = np.zeros((256, 3), dtype='uint8') # 256 x 3 크기의 빈 컬러맵 생성(RGB)

    # PASCAL VOC 2012 이나 COCO 데이터셋의 컬러맵 생성
    if dataset == 'pascal' or dataset == 'coco':
        def bitget(byteval, idx): # 특정 비트 값을 추출하는 함수
            return (byteval & (1 << idx)) != 0 # 해당 비트가 1인지 확인

        for i in range(256): # 0부터 255까지
            r = g = b = 0 # 초기 RGB 값
            c = i # 현재 색상 인덱스
            for j in range(8): # 각 비트에 대해
                r = r | (bitget(c, 0) << 7-j) # 비트를 이동시켜 색상 값에 추가
                g = g | (bitget(c, 1) << 7-j) # 비트를 이동시켜 색상 값에 추가
                b = b | (bitget(c, 2) << 7-j) # 비트를 이동시켜 색상 값에 추가
                c = c >> 3 # 다음 세 비트로 이동

            cmap[i] = np.array([r, g, b]) # 색상 값을 cmap 에 저장

    # Cityscapes 데이터셋의 컬러맵 생성
    elif dataset == 'cityscapes':
        cmap[0] = np.array([128, 64, 128]) # 도로
        cmap[1] = np.array([244, 35, 232]) # 보도
        cmap[2] = np.array([70, 70, 70]) # 건물
        cmap[3] = np.array([102, 102, 156]) # 벽
        cmap[4] = np.array([190, 153, 153]) # 울타리
        cmap[5] = np.array([153, 153, 153]) # 기둥
        cmap[6] = np.array([250, 170, 30]) # 교통 신호
        cmap[7] = np.array([220, 220, 0]) # 신호등
        cmap[8] = np.array([107, 142, 35]) # 식물
        cmap[9] = np.array([152, 251, 152]) # 초지
        cmap[10] = np.array([70, 130, 180]) # 하늘
        cmap[11] = np.array([220, 20, 60]) # 사람
        cmap[12] = np.array([255,  0,  0]) # 차
        cmap[13] = np.array([0,  0, 142]) # 트럭
        cmap[14] = np.array([0,  0, 70]) # 버스
        cmap[15] = np.array([0, 60, 100]) # 기차
        cmap[16] = np.array([0, 80, 100]) # 오토바이
        cmap[17] = np.array([0,  0, 230]) # 자전거
        cmap[18] = np.array([119, 11, 32]) # 기타

    return cmap
