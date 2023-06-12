import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np

# 모델 로드
model = load_model('model.h5')

# 입력 특성 리스트
feature = ['측정연령수', '수축기혈압(최고)mmHg', '이완기혈압(최저)mmHg', 'BMI', '체지방율', '악력']

# 진단 데이터 인코딩 이전 내용
diagnosis = {
    0: '트레드밀에서 걷기',
    1: '앉아서 다리 모으기',
    2: '앉아서 다리 벌리기',
    3: '앉아서 다리 펴기',
    4: '앉아서 다리 밀기',
    5: '앉아서 다리 굽히기',
    6: '실내 자전거타기',
    7: '앉아서 뒤로 당기기',
    8: '발 닿기',
    9: '몸통 들어올리기',
    10: '앉아서 당겨 내리기',
    11: '앉아서 모으기',
    12: '앉아서 위로 밀기',
    13: '바벨들어올리기',
    14: '앉아서 밀기',
    15: '거꾸로 누워서 밀기',
    16: '비스듬히 누워서 밀기',
    17: '허리 굽혀 덤벨 들기',
    18: '원판던지기',
    19: '서서 어깨 들어올리기',
    20: '파워클린',
    21: '앉았다 일어서기',
    22: '바벨 들어 팔꿈치 굽히기',
    23: '손목 펴기',
    24: '옆구리늘리기',
    25: '턱걸이',
    26: '덤벨 옆으로 들어올리기',
    27: '엎드려서 균형잡기',
    28: '앉아서 팔꿈치 굽히기',
    29: '윗몸 말아 올리기',
    30: '한발 앞으로 내밀고 앉았다 일어서기',
    31: '앉아서 몸통 움츠리기',
    32: '서서 균형잡으며 몸통 회전하기',
    33: '바벨 끌어당기기',
    34: '누워서 밀기',
    35: '허리 굽혀 덤벨 뒤로 들기',
    36: '짝 운동',
    37: '뒤꿈치 들기',
    38: '몸통 옆으로 굽히기',
    39: '매달려서 다리 들기',
    40: '누워서 머리 위로 팔꿈치 펴기',
    41: '의자 앞에서 앉았다 일어서기',
    42: '엎드려서 다리 차올리기'
}

# Streamlit 앱 제목 설정
st.title('운동 처방 모델')

# 사용자로부터 특성 값 입력받기
col1, col2, col3 = st.columns(3)
user_input = {}
user_input['측정연령수'] = col1.number_input('나이', min_value=0, step=1, value=0)
user_input['수축기혈압(최고)mmHg'] = col2.number_input('최고혈압 / mmHg', min_value=0.0)
user_input['이완기혈압(최저)mmHg'] = col3.number_input('최저혈압 / mmHg', min_value=0.0)

col4, col5, col6 = st.columns(3)
user_input['BMI'] = col4.number_input('BMI', min_value=0.0)
user_input['체지방율'] = col5.number_input('체지방율', min_value=0.0)
user_input['악력'] = col6.number_input('악력', min_value=0.0)

if st.button('처방'):
    input_df = pd.DataFrame([user_input])

    input_features = input_df[feature].values
    input_features = (input_features - np.mean(input_features, axis=0)) / np.std(input_features, axis=0)

    prediction = model.predict(input_features)
    predicted_labels = np.argsort(prediction)[0][-3:][::-1]

    for label in predicted_labels:
        st.write(diagnosis[label])