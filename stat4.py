from scipy.stats import uniform
import matplotlib.pyplot as plt
import numpy as np

# X ~ 균일분포 U(a,b)
# loc:a , scale: b-a
uniform.rvs(loc=2, scale=4, size=1)
#유니폼의 높이를 구하는 것 
uniform.pdf(3, loc=2, scale=4)
uniform.pdf(7, loc=2, scale=4)

k=np.linspace(0, 8, 100)
y=uniform.pdf(k, loc=2, scale=4)
plt.plot(k, y, color= "black")
plt.show()
plt.clf()

uniform.cdf(6, loc=2, scale=4) - uniform.cdf(5, loc=2, scale=4)
uniform.ppf(0.93, loc=2, scale=4)


# uniform = 균일분포 P(X<3.25) 
uniform.cdf(3.25, loc= 2, scale=4)

1.25* 0.25 ( (3.25-2)*1/4)<-넓이 구하기


# P(5<X<8.39)
uniform.cdf(8.39, loc=2, scale=4) - uniform.cdf(5, loc=2, scale=4)
uniform.cdf(6, loc=2, scale=4) - uniform.cdf(5, loc=2, scale=4)
#-> 6으로 해도 같은 값이 나옴 그 이유는 유니폼의 넓이를 구하려면 밑변과 높이가 필요한데 밑변이 2에서 5까지 밖에 없어서 6을 해도 같은 값이 아옴


#상위 7% 값은?
uniform.ppf(0.93, loc=2, scale=4) 

#X ~ 균일분포U(a , b)
#loc:a, scale: b-a


#uniform.pdf(x, loc=0, scale=1)
#uniform.cdf(x, loc=0, scale=1)
#uniform.ppf(q, loc=0, scale=1)
#uniform.rvs(loc=0, scale=1, size=None, random_state=None)


# 표본 20개를 뽑고 표본평균
x=uniform.rvs(loc=2, scale=4, size=20*1000,
              random_state=42)
#x.shape
#x.reshape(1000, 20).shape
x = x.reshape(-1, 20)
x.shape
blue_x=x.mean(axis=1)
blue_x

import seaborn as sns
sns.histplot(blue_x, stat='density')
plt.show()
plt.clf()

# 분산 
uniform.var(loc=2, scale=4)
# 기대값
uniform.expect(loc=2, scale=4)

# X bar ~ N(mu, sigma^2/n)
# X bar ~ N(4, 1.33333/20)
from scipy.stats import norm
uniform.var(loc=2, scale=4)
uniform.expect(loc=2, scale=4)


#Plot the normal distribution PDF
xmin, xmax = (blue_x.min(), blue_x.max())
x_values = np.linspace(xmin, xmax, 100)
pdf_values = norm.pdf(x_values, loc=4, scale=np.sqrt(1.33333/20))
plt.plot(x_values, pdf_values, color='red', linewidth=2)
plt.show()
plt.clf()


==================================
# 신뢰구간
# X bar ~ N(mu, sigma^2/n)
# X bar ~ N(4, 1.33333/20)
from scipy.stats import norm

#Plot the normal distribution PDF
x_values = np.linspace(3, 5, 100)
pdf_values = norm.pdf(x_values, loc=4,
                      scale=np.sqrt(1.33333/20))
plt.plot(x_values, pdf_values, color='red', linewidth=2)


#norm.ppf(0.975, loc=0, scale=1) == 1.96

#표본평균(파란벽돌) 점찍기
blue_x=uniform.rvs(loc=2, scale=4, size=20).mean()
a = blue_x + 1.96 * np.sqrt(1.33333/20)
b = blue_x - 1.96 * np.sqrt(1.33333/20)

plt. scatter(blue_x, 0.002, 
            color = 'blue', zorder=10, s=10)
plt.axvline(x=a, color='blue',
            linestyle='--', linewidth=1)
plt.axvline(x=b, color='blue',
            linestyle='--', linewidth=1)
            
#기대값 표현
plt.axvline(x=4, color='green',
            linestyle='-', linewidth=2)
            
plt.show()
plt.clf()


norm.ppf(0.025, loc=4, scale=np.sqrt(1.33333/20))
norm.ppf(0.975, loc=4, scale=np.sqrt(1.33333/20))

norm.ppf(0.0005, loc=4, scale=np.sqrt(1.33333/20)

#4.0에서 얼마나 떨어져 있는지 알아보기 위함
#4-norm.ppf(0.025, loc=4, scale=np.sqrt(1.33333/20))
#4-norm.ppf(0.975, loc=4, scale=np.sqrt(1.33333/20))



import seaborn as sns
sns.histplot(blue_x, stat='density')
plt.show()
plt.clf()

# 분산 
uniform.var(loc=2, scale=4)
# 기대값
uniform.expect(loc=2, scale=4)






