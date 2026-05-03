# 재현 방법 (Reproduction Guide)

이 문서는 위조 영수증 탐지 실험의 전체 재현 절차를 단계별로 설명한다.  
모든 실험은 Google Colab (A100 GPU) 환경에서 진행되었다.

---

## 0. 사전 준비

### 0.1 구글 드라이브 설정
```
구글 드라이브 경로: MyDrive/Colab Notebooks/dataset/findit2/
├── train/        ← 학습 이미지
├── val/          ← 검증 이미지
├── test/         ← 테스트 이미지
├── train.txt     ← 레이블
├── val.txt
├── test.txt
└── checkpoints/  ← 모델 저장 폴더 (자동 생성)
```

### 0.2 패키지 설치
```python
!pip install pytorch-msssim timm
```

### 0.3 경로 설정
```python
BASE_PATH = '/content/drive/MyDrive/Colab Notebooks/dataset/findit2'
SAVE_DIR  = BASE_PATH + '/checkpoints'
```

---

## 1. 베이스라인 실험

EfficientNet-B0 베이스라인.

```python
# 학습 설정
model     = EfficientNet-B0 (pretrained)
loss      = CrossEntropyLoss (class_weight=[1.0, 3.0])
optimizer = Adam (lr=0.0001)
epochs    = 10
```

**저장 경로:** `checkpoints/baseline_best.pt`  
**결과:** Accuracy=0.679, F1=0.244, ROC-AUC=0.593, PR-AUC=0.281

---

## 2. 오토인코더 (비지도학습)

정상 영수증 패치만으로 재구성 오차 기반 이상 탐지.

### 2.1 전처리 (5채널)
```python
# 채널 구성
Ch0: Gray + CLAHE
Ch1: ELA (quality=92, amplify=15)
Ch2: SRM (sigma=3.0)
Ch3: LBP (P=16, R=2)
Ch4: DCT 에너지
```

### 2.2 모델 학습
```python
# 학습 설정
model     = AEv2 (U-Net 스타일 오토인코더, 5채널)
loss      = MSE + SSIM (alpha=0.7)
optimizer = Adam (lr=1e-3)
scheduler = ReduceLROnPlateau (patience=5)
epochs    = 100 (조기종료 patience=10~15)

# 저장
torch.save(ckpt, 'checkpoints')
```

### 2.3 평가
```python
# 패치별 MSE 상위 Top-3 평균으로 이미지 점수 계산
score = mean(top3_patch_mse)
threshold = val set 최적값
```

**결과:** F1=0.308, Recall=0.943, ROC-AUC=0.515, PR-AUC=0.145  
**한계:** 위조 bbox MSE 상위 10% 비율 3.2% → 위조를 실제로 탐지 못함

---

## 3. EfficientNet-B3 지도학습

### 3.1 전처리 캐시 생성 (필수 - 속도 최적화)

학습 속도를 에폭당 6분 → 30초로 단축하기 위해  
전처리 결과를 로컬 SSD에 미리 저장한다.

```python
# 구글 드라이브 → 코랩 SSD 복사
!cp -r "/content/drive/.../checkpoints/preprocessed_efficientnet" "/content/"

# 또는 직접 캐시 생성
LOCAL_CACHE = '/content/preprocessed_efficientnet'

for split in ['train', 'val', 'test']:
    for img in images:
        ch6 = make_6ch(img_rgb)   # 6채널 전처리
        np.save(f'{LOCAL_CACHE}/{split}/{img}.npy', ch6)
```

### 3.2 전처리 함수
```python
def make_6ch(img_rgb):
    # Ch0~2: R/G/B + CLAHE
    # Ch3: ELA (quality=92, amplify=15)
    # Ch4: SRM (sigma=3.0)
    # Ch5: Adaptive Threshold
    return np.stack([ch0,ch1,ch2,ch3,ch4,ch5], axis=-1)
```

### 3.3 Dataset 구성
```python
# 패치 추출 전략
if label == 1 (위조):
    cx, cy = random.choice(forge_coords)  # bbox 중심
else:
    cx = random.randint(W//4, 3*W//4)    # 중앙 50% 랜덤

# Multi-Scale 패치 (12채널 = 6채널 × 2스케일)
patch_high = crop(ch6, cx, cy, 128)      # High-Res 128px
patch_low  = resize(crop(ch6, cx, cy, 256), 128)  # Low-Res
combined   = cat([patch_high, patch_low])  # (12, 128, 128)
```

### 3.4 모델 학습 (RGB+ALL, 200에폭)

```python
model     = ForgeryClassifier(in_channels=12)  # EfficientNet-B3
loss      = FocalLoss(alpha=0.75, gamma=2.0)
optimizer = Adam(lr=1e-4)
scheduler = CosineAnnealingLR(T_max=200, eta_min=1e-6)
sampler   = WeightedRandomSampler  # 클래스 불균형 보정
epochs    = 200
save_criterion = 'val PR-AUC'  # 얼리스탑핑 없음
```

```python
# 학습 루프
for epoch in range(1, 201):
    train_one_epoch(model, train_loader)
    val_pr = evaluate(model, val_loader)
    if val_pr > best_pr:
        torch.save(ckpt, 'efficientnet_best.pt')
```

**저장 경로:** `checkpoints/efficientnet_best.pt` (epoch=154)  
**결과 (Test):** F1=0.582, ROC-AUC=0.848, PR-AUC=0.703

---

## 4. Ablation Study (최우수 모델 재현)

### 4.1 전처리 캐시 생성

```python
LOCAL_ABLATION = '/content/ablation_cache'

# RGB+ELA 캐시
for split in ['train', 'val', 'test']:
    for img in images:
        ch = make_ablation_ch(img_rgb,
                              use_ela=True,
                              use_srm=False,
                              use_thresh=False)
        np.save(f'{LOCAL_ABLATION}/RGB+ELA/{split}/{img}.npy', ch)
```

### 4.2 AblationDatasetFast 사용

```python
# 캐시에서 직접 로드 (빠름)
class AblationDatasetFast(Dataset):
    def __getitem__(self, idx):
        ch = np.load(npy_path).astype(np.float32)
        # ... 패치 추출 및 augmentation
```

### 4.3 RGB+ELA 학습

```python
combo_name = 'RGB+ELA'
n_ch       = 8  # (3+1) * 2스케일

model     = ForgeryClassifier(in_channels=8)
loss      = FocalLoss(alpha=0.75, gamma=2.0)
optimizer = Adam(lr=1e-4)
scheduler = CosineAnnealingLR(T_max=50, eta_min=1e-6)
epochs    = 50
patience  = 15  # 얼리스탑핑
```

**저장 경로:** `checkpoints/ablation_RGB+ELA.pt` (epoch=40)

---

## 5. 저장된 모델로 바로 평가 (재학습 없이)

```python
from google.colab import drive
drive.mount('/content/drive')

import torch
import numpy as np
from sklearn.metrics import (f1_score, roc_auc_score,
                              average_precision_score,
                              confusion_matrix)

DEVICE   = torch.device('cuda')
SAVE_DIR = '/content/drive/MyDrive/.../checkpoints'

# 최우수 모델 로드
ckpt = torch.load(f'{SAVE_DIR}/ablation_RGB+ELA.pt',
                  map_location=DEVICE, weights_only=False)

model = ForgeryClassifier(in_channels=8).to(DEVICE)
model.load_state_dict(ckpt['model'])
model.eval()
print(f"epoch={ckpt['epoch']}  val PR-AUC={ckpt['best_pr']:.4f}")

# Test 평가
te_ds = AblationDatasetFast(
    'test.txt', split='test', augment=False,
    combo_name='RGB+ELA',
    use_ela=True, use_srm=False, use_thresh=False)
te_loader = DataLoader(te_ds, batch_size=32, shuffle=False)

t_probs, t_labels = [], []
with torch.no_grad():
    for x, y in te_loader:
        x = x.to(DEVICE)
        t_probs.extend(torch.sigmoid(model(x)).cpu().numpy().flatten())
        t_labels.extend(y.numpy().flatten())

t_probs  = np.array(t_probs)
t_labels = np.array(t_labels)
t_preds  = (t_probs >= ckpt['threshold']).astype(int)

print(f"F1      : {f1_score(t_labels, t_preds):.4f}")
print(f"ROC-AUC : {roc_auc_score(t_labels, t_probs):.4f}")
print(f"PR-AUC  : {average_precision_score(t_labels, t_probs):.4f}")
print(confusion_matrix(t_labels, t_preds))
```

---

## 주의사항

- `torch.load()` 시 반드시 `weights_only=False` 옵션 추가 (PyTorch 2.6+)
- 전처리 캐시가 없으면 에폭당 5~10분 소요 → 캐시 생성 강력 권장
- 코랩 런타임 재시작 시 `/content/` 하위 캐시 초기화 → 재복사 필요
- GPU 런타임 필수 (CPU로는 학습 불가)
