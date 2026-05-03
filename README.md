# 위조 영수증 탐지 (Receipt Forgery Detection)

데이터 링크:https://www.kaggle.com/datasets/nikita2998/find-it-again-dataset/data
kaggle 코드링크: https://www.kaggle.com/code/sonny12345/receipt-forgery-detection-deep-learning
---

## 프로젝트 개요

FinDit2 데이터셋을 활용하여 이미지 딥러닝 기반의 위조 영수증 탐지 시스템을 구현하였다.  
비지도학습(오토인코더)과 지도학습(EfficientNet-B3) 방식을 비교하고,  
다양한 전처리 채널 조합에 대한 Ablation Study를 수행하였다.

---

## 최종 성능 요약

| 모델 | F1 | Recall | Precision | ROC-AUC | PR-AUC |
|------|-----|--------|-----------|---------|--------|
| 베이스라인 (RGB, 10ep) | 0.244 | 0.294 | 0.207 | 0.593 | 0.281 |
| 오토인코더 (비지도학습) | 0.308 | 0.943 | 0.183 | 0.515 | 0.363 |
| RGB+ELA+SRM (50ep) | 0.6154 | 0.4848 | 0.8338 | 0.865 | 0.6616 |
| RGB+ELA (50ep) ← 최우수 | 0.7616 | 0.8090 | 0.7201 | 0.866 | 0.7816|
| RGB+ALL (200ep) | 0.582 | 0.457 | 0.800 | 0.848 | 0.703 |

---

## 디렉토리 구조

```
findit2/
├── train/                  # 학습 이미지 (577장)
├── val/                    # 검증 이미지 (192장)
├── test/                   # 테스트 이미지 (218장)
├── train.txt               # 학습 레이블
├── val.txt                 # 검증 레이블
├── test.txt                # 테스트 레이블
└── checkpoints/
    ├── ablation_RGB+ELA.pt       # 최우수 모델 (epoch=40, val PR-AUC=0.7816)
    ├── ablation_RGB+ELA+SRM.pt   # (epoch=39, val PR-AUC=0.7528)
    ├── ablation_RGB_only.pt      # (epoch=26, val PR-AUC=0.7157)
    ├── efficientnet_best.pt      # RGB+ALL 200ep (epoch=154, val PR-AUC=0.7796)
    ├── baseline_best.pt          # 베이스라인 (epoch=20, val PR-AUC=0.6616)
    ├── ae_v2_best.pt             # 오토인코더 v2 (best)
    └── ae_roi_best.pt            # ROI 오토인코더
```

---

## 환경 설정

- **플랫폼:** Google Colab (NVIDIA A100-SXM4-40GB)
- **Python:** 3.12
- **주요 라이브러리:**

```bash
pip install torch torchvision timm
pip install scikit-learn opencv-python pillow
pip install scipy tqdm pandas numpy
pip install pytorch-msssim
```

---

## 주요 설계 결정

### 전처리 채널 (6채널 × 2스케일 = 12채널)

| 채널 | 내용 | 역할 |
|------|------|------|
| Ch0~2 | R/G/B + CLAHE | 컬러 정보 유지, IMI 위조의 잉크 색상 차이 포착 |
| Ch3 | ELA (Error Level Analysis) | JPEG 압축 이력 분석 (분리도 1.31×) |
| Ch4 | SRM (Steganalysis Rich Model) | 고주파 노이즈 잔차 (분리도 1.26×) |
| Ch5 | Adaptive Threshold | 글자 뼈대 아티팩트 추출 |

### 샘플링 전략
- 위조 이미지: forgery bbox 중심에서 패치 추출
- 정상 이미지: 이미지 중앙 50% 범위 랜덤 샘플링

### 학습 설정 (최우수 모델 기준)
| 항목 | 값 |
|------|-----|
| 모델 | EfficientNet-B3 (8채널 입력) |
| Loss | Focal Loss (α=0.75, γ=2.0) |
| Optimizer | Adam (lr=1e-4~5e-5) |
| Scheduler | CosineAnnealingLR (T_max=50) |
| Sampler | WeightedRandomSampler |
| 저장 기준 | Val PR-AUC |
| 조기종료 | patience=15 |

---

## 재현 방법

`REPRODUCTION.md` 참고

---

## 참고문헌

- Beatriz Martinez Torne et al., "A Receipt Dataset for Document Forgery Detection," ICDAR 2023.
- Mingxing Tan and Quoc V. Le, "EfficientNet: Rethinking Model Scaling for CNNs," ICML 2019.
