# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

## 필요한 데이터 불러오기
house_test = pd.read_csv('data/houseprice/test.csv')
house_train = pd.read_csv('data/houseprice/train.csv')
sub_df = pd.read_csv('sample_submission.csv')

## 이상치 탐색
#house_train=house_train.query("GrLivArea <= 4500")


house_train.info() 

## 회귀분석 적합(fit)하기
# house_train["GrLivArea"]   # 판다스 시리즈
# house_train[["GrLivArea"]] # 판다스 프레임
# 숫자형 변수만 선택하기 
x = house_train.select_dtypes(include=[int, float])
x.info()
#필요없는 칼럼 제거하기
x=x.iloc[:,1:-1]
y = house_train["SalePrice"]


#변수별로 결측값 채우기
fill_values = {
    "LotFrontage": x["LotFrontage"].mean(),
    "MasVnrArea" : x["MasVnrArea"].mean(),
    "GarageYrBlt": x["GarageYrBlt"].mean()
}

x=x.fillna(value=fill_values)
x.isna().sum()
#x["LotFrontage"]=x["LotFrontage"].fillna(house_train["LotFrontage"].mean())
#x["MasVnrArea"]=x["MasVnrArea"].fillna(house_train["MasVnrArea"].mean())
#x["GarageYrBlt"]=x["GarageYrBlt"].fillna(house_train["GarageYrBlt"].mean())

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b




# test 데이터 예측
test_x = house_test.select_dtypes(include=[int, float])
test_x.info()
#필요없는 칼럼 제거하기
#test_x=x.iloc[:,1:-1] test 끝에는 SalePrice가 없어서 :로 써줌 

test_x=test_x.iloc[:,1:]

# 결측치 채우기 
fill_values = {
    "LotFrontage": test_x["LotFrontage"].mean(),
    "MasVnrArea" : test_x["MasVnrArea"].mean(),
    "GarageYrBlt": test_x["GarageYrBlt"].mean()
}
test_x = test_x.fillna(value=fill_values)
test_x=test_x.fillna(test_x.mean())

#결측치 확인 
test_x.isna().sum()

#fill_values 위의 코드로 대체할 수 있음 
#test_x["GarageArea"] = test_x["GarageArea"].fillna(house_test["GarageArea"].mean())
#test_x["GarageCars"] = test_x["GarageCars"].fillna(house_test["GarageCars"].mean())
#test_x["GarageYrBlt"] = test_x["GarageYrBlt"].fillna(house_test["GarageYrBlt"].mean())
#test_x["BsmtHalfBath"] = test_x["BsmtHalfBath"].fillna(house_test["BsmtHalfBath"].mean())
#test_x["BsmtFullBath"] = test_x["BsmtFullBath"].fillna(house_test["BsmtFullBath"].mean())
#test_x["TotalBsmtSF"] = test_x["TotalBsmtSF"].fillna(house_test["TotalBsmtSF"].mean())
#test_x["BsmtUnfSF"] = test_x["BsmtUnfSF"].fillna(house_test["BsmtUnfSF"].mean())
#test_x["BsmtFinSF2"] = test_x["BsmtFinSF2"].fillna(house_test["BsmtFinSF2"].mean())
#test_x["BsmtFinSF1"] = test_x["BsmtFinSF1"].fillna(house_test["BsmtFinSF1"].mean())
#test_x["MasVnrArea"] = test_x["MasVnrArea"].fillna(house_test["MasVnrArea"].mean())
#test_x["LotFrontage"] = test_x["LotFrontage"].fillna(house_test["LotFrontage"].mean())


# 테스트 데이터 집값 예측
pred_y=model.predict(test_x) # test 셋에 대한 집값
pred_y


# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/houseprice/sample_submission8.csv", index=False)


===============================================
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
