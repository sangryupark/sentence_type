# sentence_type 
DACON 문장 유형 분류 AI 경진대회
## 1. Project Overview
  - 목표
    - 주어진 문장의 유형을 분류하는 AI 모델 개발.
  - 모델
    - [klue/bert-base](https://github.com/KLUE-benchmark/KLUE) fine-tuned model.
  - Data
    - 문장, 유형, 극성, 시제, 확실성과 모든 label을 포함하고 있는 label column이 포함된 데이터 (train: 16541개, test: 7090개).

## 2. Code Structure
``` text
├── data (not in repo)
|   ├── single_label
|   |     ├── train_backtrans.csv(each label)
|   |     └── train_upsample.csv(each label)
|   ├── train.csv
|   ├── train_total_backtrans.csv
|   ├── train_total_upsample.csv
|   └── test.csv  
├── argument.py
├── backtranslation.ipynb
├── dataset.py
├── inference.py
├── loss.py
├── model.py
├── train.py
├── trainer.py
└── utils.py
```

## 3. Detail 
  - Augmentation
    - 네이버 Papago를 이용한 데이터 수가 적은(single label의 경우 최빈 label을 제외한 label, multi-label의 경우 1000개 이하) label을 영어와 일어로 번역 후 한글로 재번역하는 Backtranslation을 이용한 augmentation 진행.
    - 갯수가 적은 label 데이터를 upsampling을 통한 augmentation 진행.
  - Model
    - 전처리 된 데이터를 klue/bert-base에 fine-tunning 과정을 거침.
    - K-Fold(5 Fold)의 교차 검증과정을 통해 보다 정교한 모델 제작.
    - 한 모델안에 다중 loss를 계산하여 유형, 극성, 시제, 확실성의 label loss를 한 train과정에서 계산함.
    - Focal loss를 사용, Imbalance한 데이터 분포에 보다 정교한 loss를 사용하여 모델 성능 향상.
    - Batch size: 16 / Epoch : 20 (Early stopping with patience 2)
  - Inference
    - 각 Fold model의 inference 과정을 거친후 나온 softmax 확률을 soft-voting 과정을 거쳐 최종추론 함.
  - 최종성적
    - 대회기간 중 성적
      - Public Score: 0.72578
      - Private Score: 0.72914
    - 최종성적
      - Public Score: 0.75249
      - Private Score: 0.75180
