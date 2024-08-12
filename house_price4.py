# 팀과제
# 원하는 변수를 사용해서 회귀모델을 만들고, 제출할것!
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

## 필요한 데이터 불러오기
house_test = pd.read_csv('data/houseprice/test.csv')
house_train = pd.read_csv('data/houseprice/train.csv')
sub_df = pd.read_csv('sample_submission.csv')

##이상치 탐색(house_train.query("GrLivArea >4500") 그러면 4500인 애들만 나옴 근데 우리는 이상치 빼고 데이터를 알고 싶기 때문에 그 반대 부호로 뽑아줌줌)
house_train=house_train.query("GrLivArea <4500")


## 회귀분석 적합(fit)하기 (x 2차원 배열로 만들어줌 y는 집값 추정)
x = np.array(house_train["GrLivArea"]).reshape(-1, 1)
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

