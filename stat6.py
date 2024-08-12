# y = 2X + 3
import matplotlib.pyplot as plt 
import numpy as np
from scipy.stats import norm

# x 값의 범위 설정
x = np.linspace(0, 100, 400)

# y 값 계산
y = 2 * x + 3

#np.random.seed(20240805)
obs_x=np.random.choice(np.arange(100), 20)
epsilon_i=norm.rvs(loc=0, scale=20, size=20)
obs_y = 2*obs_x + 3 + epsilon_i

# 그래프 그리기
plt.plot(x,y, label="y = 2x + 3", color = "black")
plt.scatter(obs_x, obs_y, color = "blue", s=3)
#plt.show()
#plt.clf()


from sklearn.linear_model import LinearRegression

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
obs_x = obs_x.reshape(-1,1)
model.fit(obs_x, obs_y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_[0]      # 기울기 a hat
model.intercept_ # 절편 b hat

#회귀 직선 그리기
x = np.linspace(0, 100, 400)
y = model.coef_[0] * x + model.intercept_
plt.xlim([0,100])
plt.ylim([0,300])
plt.plot(x, y, color='red') # 회귀직선
plt.show()
plt.clf()

model
summary(model)

#!pip install statsmodels
import statsmodels.api as sm

obs_x = sm.add_constant(obs_x)
model = sm.OLS(obs_y, obs_x).fit()
print(model.summary())


from scipy.stats import norm
(1 - norm.cdf(18, loc=10, scale=1.96)) * 2

## 숙제 p57 신형 자동차의 에너지 소비효율 등급


x=np.array([15.078, 15.752, 15.549, 15.56, 16.098, 13.277, 15.462, 16.116, 15.214, 16.93, 14.118, 14.927,
15.382, 16.709, 16.804])
x.mean()
x.var()
#2. 검정을 위한 가설을 명확하게 서술하시오.
H0 : 뮤 >= 16 vs Ha : 뮤 !=< 16
 
#3. 검정통계량 계산하시오.
t_value = (np.mean(x)-16) / (np.std(x, ddof=1) / np.sqrt(len(x)))
round(t_value, 3)

#4. p‑value을 구하세요.


from scipy.stats import norm
norm.cdf(t_value, loc=0, scale=1)


#6. 현대자동차의 신형 모델의 평균 복합 에너지 소비효율에 대하여 95% 신뢰구간을 구해보세요.

z_005=norm.ppf(0.95, loc=0, scale=1)
z_005
x.mean() + z_005 * 1/ np.sqrt(15)
x.mean() - z_005 * 1/ np.sqrt(15)



