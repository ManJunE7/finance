import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

# --- 1. 페이지 설정 변경 ---
st.set_page_config(
    page_title="Lending Club",
    page_icon="💰",
    layout="centered" # 콘텐츠를 중앙에 배치하여 더 깔끔하게 보이도록 설정
)

# --- 4. 디자인 및 스타일링 ---
def set_bg_color():
    """
    CSS를 사용하여 애플리케이션의 배경색을 설정합니다.
    """
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f0fff0; /* honeydew 색상 */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_color()

# --- 데이터 로딩 및 모델 학습 함수 (기존과 동일) ---
@st.cache_data
def load_data():
    file_path = 'loan_data.csv'
    if not os.path.exists(file_path):
        data = {
            'Credit Score': np.random.randint(300, 851, size=1000),
            'Annual Income': np.random.randint(20000, 150001, size=1000),
            'Loan Amount': np.random.randint(1000, 50000, size=1000),
            'Loan Term': np.random.choice([5, 10, 15, 30], size=1000),
            'Purpose': np.random.choice(['주택', '자동차', '사업', '교육', '개인'], size=1000),
            'Loan Status': np.random.choice([0, 1], size=1000, p=[0.2, 0.8])
        }
        df = pd.DataFrame(data)
        for col in ['Credit Score', 'Annual Income']:
            df.loc[df.sample(frac=0.05).index, col] = np.nan
        df.to_csv(file_path, index=False)
    
    df = pd.read_csv(file_path)
    for col in df.select_dtypes(include=np.number).columns:
        df[col].fillna(df[col].mean(), inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    return df

@st.cache_resource
def train_model(df):
    df_processed = pd.get_dummies(df, columns=['Purpose'], drop_first=True)
    X = df_processed.drop('Loan Status', axis=1)
    y = df_processed['Loan Status']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X.columns

# --- 데이터 로딩 및 모델 학습 ---
df = load_data()
model, trained_columns = train_model(df)


# --- Streamlit 앱 메인 로직 ---

# 메인 제목 (중앙 정렬)
st.markdown("<h1 style='text-align: center;'>💰 Lending Club 대출 상환 능력 예측</h1>", unsafe_allow_html=True)
st.write("---")


# --- 2. 레이아웃 변경 (메인 화면에 입력 폼 구성) ---
st.subheader("아래 정보를 입력하여 대출 상환 능력을 예측해 보세요.")

with st.form(key='loan_prediction_form'):
    # 입력 필드들을 2열로 나누어 더 보기 좋게 배치
    col1, col2 = st.columns(2)
    
    with col1:
        credit_score = st.slider('신용 점수 (Credit Score)', 300, 850, 650)
        loan_amount = st.number_input('대출 금액 (Loan Amount)', min_value=0, value=10000, step=1000)

    with col2:
        annual_income = st.number_input('연 소득 (Annual Income)', min_value=0, value=50000, step=1000)
        loan_term = st.selectbox('대출 기간(년) (Loan Term)', sorted(df['Loan Term'].unique()))
    
    purpose = st.selectbox('대출 목적 (Purpose)', df['Purpose'].unique())
    
    # 폼 제출 버튼
    submitted = st.form_submit_button("예측하기")


# --- 예측 및 결과 표시 ---
if submitted:
    # 사용자 입력을 데이터프레임으로 변환
    input_data = {
        'Credit Score': [credit_score],
        'Annual Income': [annual_income],
        'Loan Amount': [loan_amount],
        'Loan Term': [loan_term],
        'Purpose': [purpose]
    }
    input_df = pd.DataFrame(input_data)
    
    # 입력 데이터 전처리 및 예측
    input_processed = pd.get_dummies(input_df, columns=['Purpose'])
    input_final = input_processed.reindex(columns=trained_columns, fill_value=0)
    
    prediction = model.predict(input_final)[0]
    prediction_proba = model.predict_proba(input_final)[0]

    st.write("---")
    st.header("🔍 예측 결과")
    
    # 결과를 2열로 나누어 표시
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        if prediction == 1:
            st.success("✅ **대출 상환 가능성이 높습니다.**")
        else:
            st.error("🚨 **대출 불이행 위험이 있습니다.**")
        
        st.write(f"**정상 상환 확률:** {prediction_proba[1]:.2%}")
        st.write(f"**불이행 확률:** {prediction_proba[0]:.2%}")

    with res_col2:
        # 확률 시각화
        st.bar_chart(pd.DataFrame({'확률': prediction_proba}, index=['불이행', '정상 상환']))
else:
    st.info("화면에 정보를 입력하고 '예측하기' 버튼을 클릭하세요.")
