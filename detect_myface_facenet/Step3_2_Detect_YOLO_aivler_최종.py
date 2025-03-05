import cv2
import numpy as np
from ultralytics import YOLO
import torch

# (주의!) 파일 종속성 문제 발생 가능.
# 꼭 가상환경에서 실행.
# pip install facenet-pytorch
from facenet_pytorch import InceptionResnetV1
import pandas as pd
from torchvision import transforms
from scipy.spatial.distance import cosine
###############################################
## 모델 불러오기
model = YOLO('best.pt')

# GPU 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# FaceNet 모델 초기화 (classify=True로 설정 후 분류 레이어 제거)
facenet_model = torch.load('facenet_model003.pth', map_location=device)
facenet_model = facenet_model.to(device)

# 모델을 평가 모드로 전환
facenet_model.eval()
###############################################

# Transform 설정 (FaceNet 모델에 맞게 전처리)
transform = transforms.Compose([
    transforms.ToTensor(),  # numpy 배열을 Tensor로 변환
    transforms.Resize((160, 160)),  # FaceNet 입력 크기에 맞게 조정
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # FaceNet 모델에 맞는 정규화
])

# my_face 임베딩 사전 생성
# 기준이 될 얼굴 사진 경로 지정
my_face_image = cv2.imread('./face/face_img_1657.jpg')  # 사용자의 얼굴 이미지 파일 경로
my_face_image = cv2.cvtColor(my_face_image, cv2.COLOR_BGR2RGB)
my_face_tensor = transform(my_face_image).unsqueeze(0).to(device)

with torch.no_grad():
    my_face_embedding = facenet_model(my_face_tensor).cpu().numpy()

# 유사도 임계값 설정 (0.6 이하를 같은 얼굴로 판단)
similarity_threshold = 0.6

## opencv에서 사용하려는 카메라
cap = cv2.VideoCapture(0)

## 카메라 동작 확인용
if not cap.isOpened() :
    print('웹캠 실행 불가')
    exit()

data_similarity = []
data_labels = []

## 매 프레임마다 동작시킬 것이므로 무한 반복문
while True :
    ret, frame = cap.read()
    if not ret :
        print('프레임 로드 불가')
        break
    
    frame = frame.astype(np.uint8)
    ## 카메라 좌우 전환
    frame = cv2.flip(frame, 1)
    
    ## 예측값 생성 : 무엇이 들어가야 할까요?
    results = model.predict(frame)
    #print(results)
    
    ## 예측한 것을 뜯어봅시다
    for r in results :
        ## r_b는 매 프레임의 바운딩 박스'들'의 정보를 가지고 있음
        r_b = r.boxes
        ## r_b의 클래스 예측값이 없는게 아니라면
        if not r_b.cls == None :
            ## r_b가 클래스 예측한만큼 반복 수행
            for idx in range(len(r_b)) :
                ## 점 2개의 좌표를 가져옴
                x1,y1,x2,y2 = int(r_b.xyxy[idx][0]), int(r_b.xyxy[idx][1]), int(r_b.xyxy[idx][2]), int(r_b.xyxy[idx][3])
                
                face = frame[y1:y2, x1:x2]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_tensor = transform(face).unsqueeze(0).to(device)
                with torch.no_grad():
                     my_face_embedding = facenet_model(my_face_tensor).detach().cpu().numpy().squeeze()
                     face_embedding = facenet_model(face_tensor).detach().cpu().numpy().squeeze()

                # # 코사인 유사도를 통해 내 얼굴과 비교
                # similarity = 1 - cosine(my_face_embedding, face_embedding)
                # # label = "my_face" if similarity >= similarity_threshold else "others"
                # if similarity >= 0.6:
                #     label = "my_face"
                # elif 0.3 < similarity < 0.6:
                #     label = "others"
                # else:
                #     label = "something"
                
                # data_similarity.append(similarity)
                # data_labels.append(label)
                # # 결과 표시
                # color = (0, 255, 0) if label == "my_face" else (0, 0, 255)
                # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 코사인 유사도를 통해 내 얼굴과 비교
                similarity = 1 - cosine(my_face_embedding, face_embedding)

                # similarity에 따른 라벨 설정
                if similarity >= 0.6:
                    label = "my_face"
                elif -0.6 < similarity < 0.6:
                    label = "others"
                else:
                    label = "something"
                sim = str(similarity)

                # 라벨과 유사도 값 저장
                data_similarity.append(similarity)
                data_labels.append(label)

                # similarity가 0.6 이상 또는 -0.6 ~ 0.6 사이일 때만 결과 표시
                # 벽에 생기는 네모 칸을 제거하고, 얼굴 부분에 대한 detect 결과를 보기 위함.
                if similarity >= 0.6 or (-0.6 < similarity < 0.6):
                    color = (0, 255, 0) if label == "my_face" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label , (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(frame, sim , (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


                
                # ## 1. 어쨌건 r_b에 대한 신뢰 점수가 임계값을 넘으면서,
                # ## 2. r_b의 클래스 예측이 0이라면 (즉, 다른 사람 얼굴)
                # if r_b.cls[idx]== 0:
                #     color = (0,0,255)
                #     conf = r_b.conf[idx]*100
                #     label_text = f'Other Face conf-score: {conf:.2f}'
                #     print(r_b)
                    
                # ## 1. 어쨌건 r_b에 대한 신뢰 점수가 임계값을 넘으면서,
                # ## 2. r_b의 클래스 예측이 0이 아니면 (즉, 본인 얼굴)
                # else :
                #     color = (0,255,0)
                #     conf = r_b.conf[idx]*100
                #     label_text = f'My Face conf-score: {conf:.2f}'
                # data_cls.append(r_b.cls[idx])
                # data_conf.append(r_b.conf[idx])
                
                # ## 프레임에 꼭짓점 2개의 좌표를 이용하여 박스를 표현해줌
                # ## r_b는 여럿일 수 있으므로 반복문 안에서 수행 (한 프레임에서 박스 여러 개가 있다고 예측할 수 있으니까)
                # cv2.rectangle(frame, (x1 ,y1), (x2 ,y2 ), color, 2)
                
                # ## 박스 위에 텍스트 추가
                # cv2.putText(frame, label_text, (x1, y1),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                #             color, 2,
                #             )

    ## 프레임을 확인할 수 있다
    cv2.imshow('Face_Detection', frame)
    
    ## 키보드 q 키를 누르면 반복문 종료
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break

data = pd.DataFrame({
    'similarity': data_similarity,
    'labels': data_labels
})

# 저장할 파일 이름 변경
name = 'data003.csv'
data.to_csv(f'{name}', index=False)
print(f"{name} 파일 저장")

cap.release()
cv2.destroyAllWindows()