# 직선의 방정식
# y= ax+b
# y = x
# y= 2x+3의 그래프를 그려보세요!

my_df= pd.read_csv('data/houseprice/train.csv')

import numpy as np
import matplotlib.pyplot as plt
a = 2
b = 3

x = np.linspace(-5, 5, 100)
y = a * x + b


plt.plot(x, y, color="blue")
plt.axvline(0, color="black")
plt.axhline(0, color="black")
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()
plt.clf()

=================================================
#조별과제========================================

import pandas as pd

# 예제 데이터
x = my_df["BedroomAbvGr"]
y = my_df["SalePrice"]

# 평균 계산
x_mean = np.mean(x)
y_mean = np.mean(y)

# Sxx 계산
Sxx = np.sum((x - x_mean) * (x - x_mean))

# Sxy 계산
Sxy = np.sum((x - x_mean) * (y - y_mean))

# 결과 출력
print(f"Sxx = {Sxx}")
print(f"Sxy = {Sxy}")

# 회귀 계수 계산
beta_1 = Sxy / Sxx
beta_0 = y_mean - beta_1 * x_mean

======================================================
=======================================================

a = beta_1 
b = beta_0 

x = np.linspace(0, 5, 100)
y = a * x + b


house_train = pd.read_csv('data/houseprice/train.csv')
my_df=house_train[["BedroomAbvGr", "SalePrice"]].head(10)
my_df["SalePrice"]=my_df["SalePrice"]/1000
plt.scatter(x=my_df["BedroomAbvGr"], y=my_df["SalePrice"])
plt.plot(x, y, color="blue")
plt.show()
plt.clf()

#값이 작을수록 성능이 좋음 ##거리가 작으면 좋다..
#왜냐면 원래 예측해서 그리려던 직선과 점의 사이가 가깝기 때문에
#y = 70 * x + 10
#방 1개 집값: 8천만
#방 2개 집값: 1억 5천
#방 3개 집값: 2억 2천
#방 4개 집값: 2억 9천
#방 5개 집값: 3억 6천
# 1조: (70, 10) - 0.52186
# 2조: (12, 170) - 0.46138
# 3조: (53, 45) - 0.46803 a방법 1등 값
# 4조: (36, 68) - 0.42609 b방법 1등 값값
# 5조: (80, -30) - 0.69897
# 6조: (47, 63) - 0.46078
# 7조: (63, 100) - 1.376
# 회귀: (16.38, 133.966)
# 1조: 106021410
# 2조: 94512422
# 3조: 93754868 1등
# 4조: 81472836
# 5조: 103493158

# 1조: 106021410
# 2조: 94512422
# 3조: 93754868 
# 4조: 81472836 1등
# 5조: 103493158
# 9459298301338
# 회귀(절대값): 79617742
# 회귀(제곱합): 82373710


a = beta_1 
b = beta_0 

x = np.linspace(0, 5, 100)
y = a * x + b

#테스트 집 정보 가져오기
house_test = pd.read_csv('data/houseprice/test.csv')

a = 47; b= 63
(a*house_test[ "BedroomAbvGr"] + b) * 1000 #우리가 추정하는 집값 1000을 나눠줬기 때문에 1000을 다시 곱해줘야함.


#sub 데이터 불러오기
sub_df = pd.read_csv('data/houseprice/sample_submission.csv')
sub_df

sub_df.drop(columns = ["GrLivArea"], inplace=True)

#SalePrice 바꿔치기
sub_df["SalePrice"]=(a*house_test[ "BedroomAbvGr"] + b) * 1000
sub_df

sub_df.to_csv("data/houseprice/sample_submission5.csv", index=False)


# 직선 성능 평가 
a = 16
b = 117

#y_hat은 어떻게 구할까?
house_train = pd.read_csv('data/houseprice/train.csv')
y_hat = (a * house_train["BedroomAbvGr"] + b) * 1000

# y는 어디에 있는가?
y=house_train["SalePrice"]

# 절대거리
np.abs(y - y_hat)         # 절대 거리리
np.sum(np.abs(y - y_hat)) # 절대값 합 
np.sum((y - y_hat)**2)    # 제곱합 


python
!pip install scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 예시 데이터 (x와 y 벡터)
x = np.array([1, 3, 2, 1, 5]).reshape(-1, 1)  # x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
y = np.array([1, 2, 3, 4, 5])  # y 벡터 (레이블 벡터는 1차원 배열입니다)

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_       # 기울기 a
model.intercept_  # 절편 b

slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")

# 예측값 계산
y_pred = model.predict(x)

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='data')
plt.plot(x, y_pred, color='red', label='regression line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()

==========================================
# 회귀모델을 통한 집값 예측
# house_train을 이용해 a, b 값을 구하기 


# 필요한 패키지 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

## 필요한 데이터 불러오기
house_test = pd.read_csv('data/houseprice/test.csv')
house_train = pd.read_csv('data/houseprice/train.csv')
sub_df = pd.read_csv('sample_submission.csv')

## 회귀분석 적합(fit)하기 (x 2차원 배열로 만들어줌 y는 집값 추정)
x = np.array(house_train["BedroomAbvGr"]).reshape(-1, 1)
y = house_train["SalePrice"]/1000

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값을 구해줌 (fit에 minimize가 다 들어가 있음)

# 회귀 직선의 기울기와 절편 (위에 다 불러온 다음 기울기값과 절편값을 알게됨)
model.coef_       # 기울기 a
model.intercept_  # 절편 b

slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")

# 예측값 계산 (train에 있는 집값을 넣음 )
#예측하고 싶은 값이 test.set에만 있음 saleprice를 예측해서 submission에 넣고 싶다.

y_pred = model.predict(x) 

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='data')
plt.plot(x, y_pred, color='red', label='regression line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()

test_x = np.array(house_test["BedroomAbvGr"]).reshape(-1, 1)
test_x

#a=30, b-70
#a * test_x + b 대신 밑의 함수(model.predict()를 쓴다. 빨간색 점의 높이들을 다 구한다..?
# 모델을 학습한다. = a , b 를 찾아봐라라고 하는 말과 같음

pred_y = model.predict(test_x) # test 셋에 대한 집값 -> submission 엑셀 파일에 넣어야함
pred_y

#SalePrice 바꿔치기
sub_df["SalePrice"]=pred_y * 1000
sub_df

#csv 파일 내보내기
sub_df.to_csv("data/houseprice/sample_submission4.csv", index=False)



=================================
import numpy as np
from scipy.optimize import minimize

# 최소값을 찾을 다변수 함수 정의
def my_f(x):
    return (x[0] - 1) ** 2 + (x[1] - 2) ** 2

# 초기 추정값
initial_guess = [0, 0]

# 최소값 찾기
result = minimize(my_f, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)


# 회귀직선 구하기
# minimize는 최소값 구할 수 있는 
import numpy as np
from scipy.optimize import minimize

def line_perform(par):
    y_hat=(par[0] * house_train["BedroomAbvGr"] + par[1]) * 1000
    y=house_train["SalePrice"]
    return np.sum(np.abs((y-y_hat)))

line_perform([36, 68])

# 초기 추정값
initial_guess = [0, 0]

# 최소값 찾기
result = minimize(line_perform, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

===================================================
===================================================
# ====================
# =====   옵션   =====
# ====================

import numpy as np
from scipy.optimize import minimize

# 최소값을 찾을 다변수 함수 정의
def my_f(x):
    return (x[0] - 1) ** 2 + (x[1] - 2) ** 2

# 초기 추정값
initial_guess = [0, 0]

# 최소값 찾기
result = minimize(my_f, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)


# 회귀직선 구하기

import numpy as np
from scipy.optimize import minimize

def line_perform(par):
    y_hat=(par[0] * house_train["BedroomAbvGr"] + par[1]) * 1000
    y=house_train["SalePrice"]
    return np.sum(np.abs((y-y_hat)))

line_perform([36, 68])

# 초기 추정값
initial_guess = [0, 0]

# 최소값 찾기
result = minimize(line_perform, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

===========================================================
# 24.08.02 아침수업

def my_f(x):
    return x**2+3

my_f(3)

# minimize는 최소값 구할 수 있는 
import numpy as np
from scipy.optimize import minimize

# 초기 추정값
initial_guess = [0]

# 최소값 찾기
result = minimize(my_f, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)
===========================
# z = x^2 + y^2 + 3
def my_f2(x):
    return x[0]**2 + x[1]**2 + 3

my_f2([1,3])

import numpy as np
from scipy.optimize import minimize

# 초기 추정값
initial_guess = [-10,3]

# 최소값 찾기
result = minimize(my_f2, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)
==============================
# z = (x-1)^2 + (y-2)^2 + (z-4)^2 + 7
def my_f3(x):
    return (x[0]-1)**2 + (x[1]-2)**2 + (x[2]-4)**2+7

my_f3([1,2,3])

import numpy as np
from scipy.optimize import minimize

# 초기 추정값
initial_guess = [-10,3,4]

# 최소값 찾기
result = minimize(my_f3, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)
