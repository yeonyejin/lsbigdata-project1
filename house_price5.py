# 원하는 변수를 사용해서 회귀모델을 만들고, 제출할것!
# 원하는 변수 2개 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

## 필요한 데이터 불러오기
house_test = pd.read_csv('data/houseprice/test.csv')
house_train = pd.read_csv('data/houseprice/train.csv')
sub_df = pd.read_csv('sample_submission.csv')

##이상치 탐색(house_train.query("GrLivArea >4500") 그러면 4500인 애들만 나옴 근데 우리는 이상치 빼고 데이터를 알고 싶기 때문에 그 반대 부호로 뽑아줌줌)
house_train=house_train.query("GrLivArea <= 4500") reshape한 이유 세로 형태로 벡터를 만듦
#house_train["GarageArea"]

## 회귀분석 적합(fit)하기 (x 2차원 배열로 만들어줌 y는 집값 추정)
#판다스 시리즈는 칼럼명이 날라감dtype 있음 . 넘파이 벡터..? 여튼 그래서 1차원 벡터임 그래서 2차원 벡터로 만들어 주기 위해 reshape을 해준것  , 판다스 프레임은 dtype없음 칼럼명 있음 , row column 도 있음 
#x = np.array(house_train[["GrLivArea", "GarageArea"]]).reshape(-1, 2) #reshape한 이유 세로 형태로 벡터를 만듦
x = house_train[["GrLivArea", "GarageArea"]]
y = house_train["SalePrice"]

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값을 구해줌 (fit에 minimize가 다 들어가 있음)

#시각화
# 회귀 직선의 기울기와 절편 (위에 다 불러온 다음 기울기값과 절편값을 알게됨)
model.coef_       # 기울기 a
model.intercept_  # 절편 b

slope = model.coef_[0]
slope = model.coef_[1]
intercept = model.intercept_
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")

def my_houseprice(x, y):
    return model.coef_[0] * x + model.coef_[1]* y + model.intercept_

my_houseprice(300, 55)

temp_result=my_houseprice(house_test["GrLivArea"], house_test["GarageArea"])

test_x = house_test[["GrLivArea", "GarageArea"]]
test_x

#결측치 확인
test_x["GrLivArea"].isna().sum()
test_x["GarageArea"].isna().sum()
test_x=test_x.fillna(house_test["GarageArea"].mean())

# 예측값 계산 (train에 있는 집값을 넣음 )
#예측하고 싶은 값이 test.set에만 있음 saleprice를 예측해서 submission에 넣고 싶다.

pred_y=model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/houseprice/sample_submission7.csv", index=False)



# 시각화
# 직선값 계산
y_pred = model.predict(x)

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='data')
plt.plot(x, y_pred, color='red', label='regression')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([0, 5000])
plt.ylim([0, 900000])
plt.legend()
plt.show()
plt.clf()
===========================


y_pred = model.predict(x) 

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='data')
plt.plot(x, y_pred, color='red', label='regression line')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([0, 5000])
plt.ylim([0, 900000])
plt.legend()
plt.show()
plt.clf()

test_x = np.array(house_test["GrLivArea"]).reshape(-1, 1)
test_x

#a=30, b-70
#a * test_x + b 대신 밑의 함수(model.predict()를 쓴다. 빨간색 점의 높이들을 다 구한다..?
# 모델을 학습한다. = a , b 를 찾아봐라라고 하는 말과 같음

pred_y = model.predict(test_x) # test 셋에 대한 집값 -> submission 엑셀 파일에 넣어야함
pred_y

#SalePrice 바꿔치기
sub_df["SalePrice"]=pred_y 
sub_df

#csv 파일 내보내기
sub_df.to_csv("data/houseprice/sample_submission9.csv", index=False)
