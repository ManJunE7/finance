import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# --- 1. 역할 및 목표에 따른 기본 설정 ---
# 역할: 데이터 과학 및 머신러닝 애플리케이션 개발 전문가
# 목표: Streamlit을 사용하여 사용자가 데이터를 입력하면 대출 상환 가능 여부(정상 상환/불이행)를 예측하는 웹 애플리케이션 개발

# --- 2. 데이터 로딩 및 전처리 (가상 데이터 생성 포함) ---

# 캐시를 사용하여 데이터 로딩 속도 향상
@st.cache_data
def load_data():
    """
    'loan_data.csv' 파일이 있으면 로드하고, 없으면 가상의 데이터를 생성합니다.
    """
    file_path = 'loan_data.csv'
    if not os.path.exists(file_path):
        st.info(f"'{file_path}'를 찾을 수 없어, 테스트용 가상 데이터를 생성합니다.")
        # 가상 데이터 생성
        data = {
            'Credit Score': np.random.randint(300, 851, size=1000),
            'Annual Income': np.random.randint(20000, 150001, size=1000),
            'Loan Amount': np.random.randint(1000, 50000, size=1000),
            'Loan Term': np.random.choice([5, 10, 15, 30], size=1000),
            'Purpose': np.random.choice(['주택', '자동차', '사업', '교육', '개인'], size=1000),
            'Loan Status': np.random.choice([0, 1], size=1000, p=[0.2, 0.8]) # 0: 불이행, 1: 정상 상환
        }
        df = pd.DataFrame(data)
        # 결측치 시뮬레이션
        for col in ['Credit Score', 'Annual Income']:
            df.loc[df.sample(frac=0.05).index, col] = np.nan
        df.to_csv(file_path, index=False)
    
    df = pd.read_csv(file_path)
    
    # 결측치 처리
    for col in df.select_dtypes(include=np.number).columns:
        df[col].fillna(df[col].mean(), inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
        
    return df

# --- 4. 머신러닝 모델 학습 ---

# 캐시를 사용하여 모델 학습 과정을 반복하지 않도록 설정
@st.cache_resource
def train_model(df):
    """
    데이터프레임을 받아 RandomForestClassifier 모델을 학습하고 반환합니다.
    """
    # 범주형 데이터 변환 (One-Hot Encoding)
    df_processed = pd.get_dummies(df, columns=['Purpose'], drop_first=True)
    
    # 피처(X)와 타겟(y) 분리
    X = df_processed.drop('Loan Status', axis=1)
    y = df_processed['Loan Status']
    
    # 훈련/테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 랜덤 포레스트 모델 초기화 및 학습
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # (선택) 모델 성능 출력
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.sidebar.write(f"테스트 데이터 정확도: {accuracy:.2f}")

    # 학습에 사용된 컬럼 정보도 함께 반환
    return model, X.columns

# --- Streamlit 앱 메인 로직 ---

st.set_page_config(page_title="대출 상환 능력 예측", layout="wide")

# 제목
st.title("🤖 대출 상환 능력 예측 웹 애플리케이션")
st.write("---")

# 데이터 로드 및 표시
df = load_data()
model, trained_columns = train_model(df)

# 데이터 정보 표시
st.header("📊 학습 데이터 정보")
st.write("이 애플리케이션은 아래와 같은 데이터로 학습되었습니다.")
col1, col2 = st.columns(2)
with col1:
    st.subheader("데이터 샘플 (상위 5개)")
    st.dataframe(df.head())
with col2:
    st.subheader("기초 통계 정보")
    st.write(df.describe())

st.write("---")

# --- 3. 사이드바(Sidebar) UI 구성 ---
st.sidebar.header("사용자 정보 입력")
st.sidebar.write("아래 정보를 입력하여 대출 상환 능력을 예측해 보세요.")

# 사용자 입력 받기
credit_score = st.sidebar.slider('신용 점수 (Credit Score)', 300, 850, 650)
annual_income = st.sidebar.number_input('연 소득 (Annual Income)', min_value=0, value=50000, step=1000)
loan_amount = st.sidebar.number_input('대출 금액 (Loan Amount)', min_value=0, value=10000, step=1000)
loan_term = st.sidebar.selectbox('대출 기간(년) (Loan Term)', sorted(df['Loan Term'].unique()))
purpose = st.sidebar.selectbox('대출 목적 (Purpose)', df['Purpose'].unique())

# --- 5. 예측 및 결과 표시 ---

# 사용자 입력을 데이터프레임으로 변환
input_data = {
    'Credit Score': [credit_score],
    'Annual Income': [annual_income],
    'Loan Amount': [loan_amount],
    'Loan Term': [loan_term],
    'Purpose': [purpose]
}
input_df = pd.DataFrame(input_data)

st.header("🔍 예측 결과")

if st.sidebar.button("예측 실행"):
    # 입력 데이터 전처리 (One-Hot Encoding)
    input_processed = pd.get_dummies(input_df, columns=['Purpose'])
    # 학습된 모델의 컬럼 순서에 맞게 재정렬
    input_final = input_processed.reindex(columns=trained_columns, fill_value=0)

    # 예측 수행
    prediction = model.predict(input_final)[0]
    prediction_proba = model.predict_proba(input_final)[0]

    # 결과 표시
    st.subheader("입력한 정보")
    st.table(input_df)
    
    st.subheader("예측 결과")
    if prediction == 1:
        st.success("✅ **대출 상환 가능성이 높습니다.**")
    else:
        st.error("🚨 **대출 불이행 위험이 있습니다.**")

    # 예측 확률 표시
    st.write(f"**정상 상환 확률:** {prediction_proba[1]:.2%}")
    st.write(f"**불이행 확률:** {prediction_proba[0]:.2%}")

    # 확률 시각화
    st.bar_chart(pd.DataFrame({'확률': prediction_proba}, index=['불이행', '정상 상환']))
else:
    st.info("사이드바에 정보를 입력하고 '예측 실행' 버튼을 클릭하세요.")

