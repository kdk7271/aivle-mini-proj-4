# AIVLE SCHOOL 미니프로젝트 4차 

![Image](https://github.com/user-attachments/assets/bf4c7e8b-0ed1-4e60-9cc7-3633aa9811d7)
---

## 이미지 데이터 모델링 및 얼굴인식

> KT AIVLE SCHOOL 미니 프로젝트 4차
> 
> 기간 : 2025-10-28 ~ 2025-11-1


## 프로젝트 소개

   * 얼굴 인식을 위한 모델을 직접 구현하고 이를 통해 주어진 얼굴 데이터셋을 학습하여 개인의 얼굴을 탐지 및 인식하는 시스템을 구현하는 것을 목적으로 합니다.
   
   * 데이터 수집, 전처리, 데이터 증강, 모델 학습 및 평가 등 인공지능 개발의 전체 프로세스를 실습하였습니다.


## 프로젝트 개요

![Image](https://github.com/user-attachments/assets/9843cf43-e1ac-44e2-91ed-eef5459758de)


## 주요 과정

1. 데이터 수집 및 전처리

   * 주어진 얼굴 이미지 데이터를 불러오고 전처리하여 학습 가능한 형태로 준비합니다.
   * 이미지 리사이징 및 정규화를 진행합니다.
    
2. 데이터 증강 및 준비

   * 데이터 부족 문제를 해결하기 위해 이미지 증강을 진행하여 데이터셋의 다양성을 확보합니다.
   * 관련 스크립트(Step0_etc_image_augment.py, Step0_etc_image_rename.py, Step0_etc_image_save.py)를 사용하여 이미지 처리를 진행합니다.
     
3. 모델 구현 및 학습

   * FaceNet 기반 모델(Step1_1_Make_FaceNet_aivler.ipynb)과 MTCNN 모델(Step1_2_Detect_Keras_aivler.py)을 사용하여 얼굴을 탐지하고 인식할 수 있도록 학습을 진행합니다.
     
4. 평가 및 테스트

   * 학습한 모델로 실제 얼굴 이미지 데이터를 활용해 성능을 검증합니다.
   개인의 얼굴을 정확하게 탐지하고 식별할 수 있는지를 평가합니다.

## 프로젝트 결과

![Image](https://github.com/user-attachments/assets/741736ab-0dce-4fa3-9448-0b48b19b9910)
----
![Image](https://github.com/user-attachments/assets/06636a50-fc67-4b64-9633-18fe7cfbf582)
----
![Image](https://github.com/user-attachments/assets/23ce7487-e669-458f-84d1-29f967ebf94b)
----
![Image](https://github.com/user-attachments/assets/e186c22b-0082-4f21-b3dc-9bca7e7290f0)


1. 얼굴 인식 시스템 구축 완료
   
   * FaceNet과 MTCNN 모델 기반의 얼굴 인식 시스템을 성공적으로 구축했습니다.
   * 본인의 얼굴과 타인의 얼굴을 명확하게 구분하여 분류하는 성능을 달성했습니다.
     
2. 데이터 처리 결과

   * 제공된 얼굴 데이터를 전처리하여 효과적으로 학습 가능한 형태로 구성했습니다.
   * 본인의 얼굴 데이터와 타인의 얼굴 데이터를 명확히 분리해 학습셋(Training set)과 테스트셋(Test set)을 구축하였습니다.
     
3. 데이터 증강을 통한 성능 향상

   * 이미지 데이터 증강(Augmentation)을 통해 데이터셋의 크기를 증가시키고 다양성을 확보하여 모델의 일반화 성능을 향상했습니다.
     
4. 모델 성능 평가

   * FaceNet 모델을 통해 개인 얼굴 인식 정확도와 신뢰성을 확보했습니다.
   * MTCNN을 통해 이미지 내 얼굴 탐지 성능을 높여, 더욱 정확한 인식 결과를 얻었습니다.
   * 본인과 타인의 얼굴을 구별하는 정확도가 높게 나타났으며, 실제 활용 가능한 수준의 성능을 확보했습니다.
     


## 느낀점

이번 프로젝트를 통해 얼굴 인식 모델 개발 및 이미지 데이터 모델링 과정을 심도 있게 경험할 수 있었습니다.

특히, 데이터 전처리 과정에서 이미지 증강 기법과 데이터셋 분할 방법이 모델 성능에 매우 큰 영향을 미친다는 것을 직접 확인할 수 있었습니다.

또한, FaceNet과 MTCNN 모델을 직접 구현하면서 이미지 특성 추출과 얼굴 탐지의 원리를 보다 명확히 이해할 수 있었습니다. 다양한 변수와 학습 파라미터의 조정이 모델 정확도 향상에 얼마나 중요한지 알게 되었습니다.

이미지 증강을 통해 데이터를 다양화하는 과정에서, 데이터의 품질과 다양성이 실제 성능 향상에 큰 역할을 한다는 것을 직접 확인할 수 있었습니다. 이 과정에서 반복적 학습과 평가를 통해 모델 성능 개선의 실질적인 방법론을 습득했습니다.
