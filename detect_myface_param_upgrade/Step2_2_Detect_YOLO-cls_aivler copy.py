import cv2
import numpy as np
from ultralytics import YOLO
from facenet_pytorch import MTCNN
import torch

# YOLO 모델 불러오기
model = YOLO('yolov8n-cls_s.pt')

# OpenCV에서 사용하려는 카메라
cap = cv2.VideoCapture(0)

# MTCNN 얼굴 탐지기 초기화
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)  # 모든 얼굴을 탐지하도록 설정

# 카메라 동작 확인용
if not cap.isOpened():
    print('웹캠 실행 불가')
    exit()

# 매 프레임마다 동작시킬 것이므로 무한 반복문
while True:
    ret, frame = cap.read()
    if not ret:
        print('프레임 로드 불가')
        break

    # 카메라 좌우 반전
    frame = cv2.flip(frame, 1)

    # MTCNN으로 얼굴 탐지
    boxes, _ = mtcnn.detect(frame)

    # 얼굴이 탐지되었을 때
    if boxes is not None:
        for box in boxes:
            # box 좌표를 정수로 변환
            x1, y1, x2, y2 = map(int, box)

            # 얼굴 영역 추출 및 RGB로 변환
            face = frame[y1:y2, x1:x2]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            # YOLO 모델을 통한 예측 수행
            results = model.predict(face_rgb)

            # 예측값을 이용한 반복문
            for r in results:
                # 확률이 가장 높은 클래스가 0이라면, 초록색 박스 그림
                if r.probs.top1 == 0:
                    color = (0, 255, 0)
                    prob = float(r.probs.top1conf) * 100
                    label_text = f'My Face prob: {prob:.2f}%'
                # 그렇지 않다면, 빨간색 박스 그림
                else:
                    color = (0, 0, 255)
                    prob = float(r.probs.top1conf) * 100
                    label_text = f'Other Face prob: {prob:.2f}%'

            # 프레임에 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 박스 위에 텍스트 추가
            cv2.putText(frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # 프레임을 확인할 수 있다
    cv2.imshow('Face_Detection', frame)

    # 키보드 q 키를 누르면 반복문 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
