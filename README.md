## 음성 추 파이프라인
![voice pipeline](https://github.com/Team-Remember/rememberMe-voice_train_AI/blob/main/img/voice%20pipeline.png)
- 모델 : Fast speech2
- 챗봇의 결과로 나온 문자(개인별 챗봇 데이터)를 기반으로 추론함으로써 악용할 수 있는 기회를 제거 하였습니다.
- 개인별 checkpoint는 google cloud에서 관리하며, 사용할 때 파일을 로드하여 추론합니다.
- 음성 추론 시간은 약 3초 정도 소요됩니다.
