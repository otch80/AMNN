해당 레포는 AMNN의 코드를 에러없이 사용할 수 있도록 사용의 편의를 위해 생성되었습니다.

# AMNN
> AMNN: Attention-based Multimodal Neural Network Model for Hashtag Recommendation  
> 이미지와 텍스트를 분석하여 맞춤형 해시태그를 추천하는 모델

<br>

## 폴더 구조
```bash
├─dataset : [모델 학습 폴더]
|   ├─(jpg image files) : [학습 대상 이미지]
|   ├─image_text_tag.csv : [이미지명, 텍스트, 해시태그]
|   └─preprocessed_data
|       ├─...
|       ├─data_parameters.log : [학습한 데이터 관련 정보]
|       ├─id_to_word.p : [id, 해시태그]
|       ├─resnet50_Image_name_to_features.h5 : [추출한 이미지 features]
|       ├─word_to_id.p : [해시태그, id]
|       └─...
├─test_dataset : [예측 폴더] 
|   ├─(jpg image files) : [예측 이미지]
|   └─image_text_tag.csv : [이미지명, 텍스트]
|    └─preprocessed_data
|       ├─...
|       ├─data_parameters.log : [예측 이미지 관련 정보]
|       ├─resnet50_image_name_to_features.h5 : [예측 이미지 feature]
|       └─...
├─Attention.py : [Attention Layer]
├─confing.py : [가공 사전 정보]
├─data_manager.py : [image feature 추출, hashtag 전처리]
├─evaluator.py : [모델 성능 평가]
├─generator.py : [text tokenize, model flow 설정]
├─models.py : [AMNN init]
├─README.md
├─result_accuracy.py : [ACC, Recall, Precision 계산]
├─train.py : [모델 학습]
├─predict.py : [해시태그 예측]
└─...
```

<br>

## 주의사항

- text 파일을 기준으로 작성된 코드를 csv로 활용이 가능하게끔 변경하여, 사용 데이터에 따라 조금의 에러가 발생할 수 있습니다.
- 코드에서 사용하는 단어에 혼동이 있을 수 있습니다. (동작에 이상이 없어 수정하지 않았습니다)
  - text : 토큰화된 텍스트
  - tweets : 사용자 입력 텍스트
  - caption : 해시태그

<br>

## 학습 데이터셋

|Name| Contents|captions|
|:---:|:---|:---|
| 2768320131198970077 | ... 팬케이크가루 조금 계란 하나 섞어서 구워주면 집이 바로 브런치 카페... | ['브런치메뉴', '팬케이크', '바나나팬케이크']|
| 2768372291294523883 | ... 아프지말고 튼튼하게 커주길 ... | ['헬린이', '헬스남', '다이어트', '육아', '육아스타그램', '육아그램'] | 
|2768152909861339732 | 체력고갈 | ['오오티디', '셀카그램', '연애스타그램', '셀기꾼']|
| 2767608636311416933 | ... 내가 가장 좋아하는 아쿠아리움 ...  | ['follow4follow', 'fff', 'selfie', 'likeforlike', 'life']|
| 2759485501676069416 | ... 사진좀 찍어 ... | ['인스타패션', '감성', '마스타', '강아지',  '여행사진', '맛스타그램', 'handmade', '헬스타그램', '선팔하면맞팔'] | 


## 예측 데이터셋

| image_names | tweets |
|:---:|:---|
| 1m아트홀 | 대학로에 있는 1m클래식아트홀은 어린이들을 위한 어린이들만의 클래식 전용 체험관으로 미술관이나 ...
| 40계단 문화관 | 한국전쟁 당시의 역사와 삶의 애환이 담겨있는, 40계단 문화관  1950년 한국전쟁 ...|
| 가톨릭대학교 성신교정도서관 | 한국 교회를 이끌어 갈 성직자 양성과, 가톨릭 교육의 최고 기관 ...| | 갓전시관 | 제주특별자치도는 조선시대 갓 공예의 중심지였던 이미지를 부각시켜 ...|
| 강동아트센터 | 강동아트센터는 강동구 명일근린공원 내에 위치한 전문 공연장으로 자연공원과 문화공간이 경계없이 조우하는... |


## 라이브러리

- keras - 2.7.0
- tensorflow - 2.7.0
- h5py - 3.1.0
- pickle - 4.0

<br>

## 사용방법

> GPU 사용이 가능한 환경에서 실행시키길 권장드립니다. (Colab)  
> 대략 1만장의 경우 총 학습시간은 2시간 정도 걸렸습니다 (코랩기준)  
> 예측시간은 1시간 이내였습니다   

1. 학습, 예측 데이터셋을 준비합니다 (폴더 구조는 위에서 설명하였습니다)
1. train.py 에서 [학습 이미지 경로, CNN layer, 이미지 meta data 저장경로, 모델 저장 경로] 등을 설정합니다.
    - 최대 해시태그 갯수, 토큰 길이 등은 config.py에서 수정이 가능합니다
2. train.py 를 실행시킵니다. 
    - data_manager 생성 부분의 extract_image_features를 통해 초기 이미지 feature 추출 여부를 결정할 수 있습니다.
3. result_accuracy.py 를 실행하여 모델의 성능을 평가합니다
4. predict.py 에서 예측 데이터에 대한 설정을 합니다.
5. predict.py 를 실행시켜 해시태그 예측을 마무리합니다.