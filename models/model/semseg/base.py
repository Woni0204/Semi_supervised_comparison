from model.backbone.resnet import resnet50, resnet101 # resnet50, resnet101 로드

from torch import nn
import torch.nn.functional as F


class BaseNet(nn.Module): # nn.Module 상속 받아 BaseNet이라는 새로운 클래스
    def __init__(self, backbone):
        super(BaseNet, self).__init__() # nn.Module 호출하여 BaseNet 클래스의 인스턴스 초기화
        backbone_zoo = {'resnet50': resnet50, 'resnet101': resnet101} # resnet50, resnet101 두 가지 ResNet 버전을 들고 옴.
        self.backbone = backbone_zoo[backbone](pretrained=True) # 전달받은 'backbone' 이름에 따라 미리 학습된 ResNet (resnet50 or resnet101) 을 가져옴.

    def base_forward(self, x):
        h, w = x.shape[-2:] # 입력 이미지의 너비(w)와 높이(h)를 추출

        x = self.backbone.base_forward(x)[-1] # 입력 텐서 'x'를 forward 하여 특징 맵을 추출하고, 그 중 마지막 특징 맵을 가져옴.
        x = self.head(x) # 추가적으로 특징 맵 'x'를 처리함.
        x = F.interpolate(x, (h, w), mode="bilinear", align_corners=True) # 특징 맵 'x'를 원래 이미지 크기 '(h, w)'로 보간(interpolate)하여 크기를 맞춤.

        return x # 최종 처리 결과 'x'를 반환

    def forward(self, x, tta=False):
        if not tta:
            return self.base_forward(x)

        else:
            h, w = x.shape[-2:] # 입력 이미지의 너비(w)와 높이(h)를 추출
            scales = [0.5, 0.75, 1.0, 1.5, 2.0] # 사용할 다양한 이미지 스케일을 정의 -> 이를 통해 입력 이미지를 여러 크기로 변환

            final_result = None # 최종 저장할 변수를 초기화

            for scale in scales: # 각 스케일에 대하여 반복 진행
                cur_h, cur_w = int(h * scale), int(w * scale) # 현재 스케일에 따라 새로운 너비 'cur_w'와 새로운 높이 'cur_h'를 계산
                cur_x = F.interpolate(x, size=(cur_h, cur_w), mode='bilinear', align_corners=True) # 입력 이미지를 현재 스케일에 맞춰 'bilinear' 보간법으로 크기를 변경

                out = F.softmax(self.base_forward(cur_x), dim=1) # 현재 크기의 이미지를 forward 한 후, 결과에 대하여 softmax 를 적용
                out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True) # forward 한 결과를 원래 이미지 크기 '(h, w)'로 보간하여 맞춤
                final_result = out if final_result is None else (final_result + out) # 'final_result'가 초기값이면 'out'으로 설정하고 그렇지 않으면 'out'을 'final_result'에 더하여 여러 스케일에 대한 결과를 누적함.

                out = F.softmax(self.base_forward(cur_x.flip(3)), dim=1).flip(3) # 수평 반전된 이미지를 forward 하여 softmax를 적용한 후, 다시 수평 반전하여 원래 방향으로 만듦.
                out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True) # 반전된 이미지에 대한 결과를 원래 이미지 크기로 보간.
                final_result += out # 수평 반전된 결과를 final_result에 더함.

            return final_result
