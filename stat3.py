from scipy.stats import bernoulli


#확률질량함수 (pmf)
#확률변수가 갖는 갑에 해당하는 확률을 저장하고 있는 함수
# bernoulli.pmf(k,p)
#P(X=1)
bernoulli.pmf(1, 0.3)

#P(X=0)
bernoulli.cdf(0, 0.3)

#P(X=k | n,p)
#n:베르누이 확률변수 더한 갯수
#p:1이 나올 확률
#binom.pmf(k, n, p)
from scipy.stats import binom
binom.pmf(0, n=2, p=0.3)
binom.pmf(1, n=2, p=0.3)
binom.pmf(2, n=2, p=0.3)

# X~B(n, p)
#list comp.
result=[binom.pmf(x, n=30, p=0.3) for x in range(31)]
result

#numpy
import numpy as np
binom.pmf(np.arange(31), n=30, p=0.3)


import math

math.factorial(54)/(math.factorial(26)*math.factorial(28))
math.comb(54, 26)
# math.comb는 위를 예시로 말하자면, 54!/26!을 계산한 것임



#몰라도됨=======================================
#np.cumprod(np.arange(1,55))[-1] / (np.cumprod(np.arange(1,27)))[-1] * (np.cumprod(np.arange(1,28))[-1]
#fact_54 = np.cumprod(np.arange(1, 55))[-1]
#ln
log(a*b) = log(a) + log(b)
log(1 * 2 * 3 * 4) = log(1) + log(2) + log(3) + log(4)

np.log(24) 
sum(np.log(np.arange(1, 5)))

math.log(math.factorial(54))
sum(np.log(np.arange(1, 55)))

math.log(math.factorial(54)) 
logf_54=sum(np.log(np.arange(1,55)))
logf_26=sum(np.log(np.arange(1,27)))
logf_28=sum(np.log(np.arange(1,29)))
#math.comb(54, 26)
logf_54 - (logf_26 +logf_28)
np.exp(logf_54 - (logf_26 +logf_28))
#===================================

math.comb(2,0) * 0.3**0 * (1-0.3)**2
math.comb(2,1) * 0.3**1 * (1-0.3)**1
math.comb(2,2) * 0.3**2 * (1-0.3)**0

#pmf: probability mass function(확률질량함수)
binom.pmf(0, 2, 0.3)
binom.pmf(1, 2, 0.3)
binom.pmf(2, 2, 0.3)

#X~B(n=10, p=0.36)
#P(X=4) = ?
binom.pmf(4, n=10, p=0.36)

# P(X<= 4)?
np.arange(5)
binom.pmf(np.arange(5), n=10, p=0.36).sum()
np.arange(3,9)
binom.pmf(np.arange(3,9), n=10, p=0.36).sum()

#X~B(30, 0.2)
binom.pmf(np.arange(31), n=30, p=0.2).sum()
#1
a=binom.pmf(np.arange(4), n=30, p=0.2).sum()
#2
b=binom.pmf(np.arange(25, 31), n=30, p=0.2).sum()
#3
a+b
#4
1-binom.pmf(np.arange(4, 25), n=30, p=0.2).sum()

# rvs 함수 (random variates smaple)
# 표본 추출 함수
# X1 ~ Bernulli(p=0.3)
bernoulli.rvs(p=0.3)
# X2 ~ Bernulli(p=0.3)
bernoulli.rvs(0.3)
# X~B(n=2, p=0.3)
bernoulli.rvs(0.3) + bernoulli.rvs(0.3)
binom.rvs(n=2, p=0.3, size=1)
binom.pmf(0, n=2, p=0.3)
binom.pmf(1, n=2, p=0.3)
binom.pmf(2, n=2, p=0.3)

# X ~ B(30, 0.26)
#기대값=30*0.26 = 7.8 
#표본 30개를 뽑아보세요.
binom.rvs(n=30, p=0.26, size=30)

# X~B(30, 0.26)
binom.rvs(n=30, p=0.26, size=30)
30*0.26


import seaborn as sns
prob_x=binom.pmf(np.arange(31), n=30, p=0.26)
sns.barplot(prob_x)
import matplotlib.pyplot as plt
plt.show()
plt.clf()

# 교재 p.207
x= np.arange(31)
import pandas as pd
df = pd.DataFrame({ 'x': x, 'prob':prob_x})
df

sns.barplot(data= df, x ="x", y="prob")
plt.show()


# cdf: cumulative dist. function 
#(누적확률분포 함수)
# F_(X=x) = P(X<= x)
binom.cdf(4, n=30, p=0.26)
binom.cdf(18, n=30, p=0.26) - binom.cdf(4, n=30, p=0.26)
binom.cdf(19, n=30, p=0.26) - binom.cdf(13, n=30, p=0.26)

import numpy as np
import seaborn as sns
x_1 = binom.rvs(n=30, p=0.26, size=10)
x_1
x= np.arange(31)
prob_x=binom.pmf(x, n=30, p=0.26)
sns.barplot(prob_x, color="blue")

# Add a point at (2, 0)
plt. scatter(x_1, np.repeat(0.002, 10), color = 'red', zorder=10, s=3)

#기대값 표현
plt.axvline(x=7.8, color= 'green', linestyle='--', linewidth=2)
plt.show()
plt.clf()

30*0.26

binom.ppf(0.5, n=30, p=0.26)
binom.cdf(8, n=30, p=0.26)
binom.cdf(7, n=30, p=0.26)

binom.ppf(0.7, n=30, p=0.26)
binom.cdf(8, n=30, p=0.26)
binom.cdf(9, n=30, p=0.26)


1/np.sqrt(2*math.pi)
from scipy.stats import norm

norm.pdf(0, loc=0, scale=1)
norm.pdf(5, loc=3, scale=4)

# 정규분포 pdf 그리기
k=np.linspace(-5, 5, 100)
y=norm.pdf(k, loc=0, scale=1)

plt.plot(k, y, color= "black")
plt.show()
plt.clf()

## mu (loc): 분포의 중심 결정하는 모수
k=np.linspace(-5, 5, 100)
y=norm.pdf(k, loc=0, scale=1)

plt.plot(k, y, color= "black")
plt.show()
plt.clf()

## sigma(scale): 분포의 퍼짐을 결정하는 모수(표준편차)
k=np.linspace(-5, 5, 100)
y=norm.pdf(k, loc=0, scale=1)
y2=norm.pdf(k, loc=0, scale=2)
y3=norm.pdf(k, loc=0, scale=0.5)

plt.plot(k, y, color= "black")
plt.plot(k, y2, color= "red")
plt.plot(k, y3, color= "blue")
plt.show()
plt.clf()


norm.cdf(0, loc=0, scale=1)
norm.cdf(100, loc=0, scale=1)


norm.cdf(0.54, loc=0, scale=1) - norm.cdf(-2, loc=0, scale=1)

# 정규분포: normal distribution
# X~N(3, 5^2)
# P( 3< X <5) =? 15.54%
norm.cdf(5, 3, 5) - norm.cdf(3,3,5)
#위 확률변수에서 표본 1000개 뽑아보자!

x=norm.rvs(loc=3, scale=5, size=1000)
sum((x > 3) & (x < 5))/1000

#평균:0, 표준편차: 1
#표본 1000개 뽑아서 0보다 작은 비율 확인
x=norm.rvs(loc=0, scale=1, size=1000)
np.mean(x<0)
(x<0).mean()
len(x<0)
sum(x<0)/1000

sns.histplot(x)


x=norm.rvs(loc=3, scale=2, size=1000)
x
sns.histplot(x, stat='density')

#Plot the normal distribution PDF
xmin, xmax = (x.min(), x.max())
x_values = np.linspace(xmin, xmax, 100)
pdf_values = norm.pdf(x_values, loc=3, scale=2)
plt.plot(x_values, pdf_values, color='red', linewidth=2)
plt.show()
plt.clf()



## 숙제 Qmd
#숙제1번 import scipy.stat import norm 사용금지
# 1. 정규분포 pdf 값을 계산하는 자신만의 
# 파이썬 함수를 정의하고,
# 정규분포 mu = 3, sigma = 2 의 pdf를 그릴 것.

import numpy as np
import matplotlib.pyplot as plt
import math


def my_pdf (x, mu, sigma):
    part_1 = 1 / (sigma * math.sqrt(2 * math.pi))
    part_2 = math.exp(-0.5 * ((x - mu) ** 2) / (sigma ** 2))
    return part_1 * part_2

mu = 3
sigma = 2
x_values=np.linspace(-3, 10, 100)
pdf_values = [my_pdf(x, mu, sigma) for x in x_values]
plt.plot(x_values, pdf_values, color="black")
plt.show()
plt.clf()

# 2. 파이썬 scipy 패키지 사용해서 다음과 같은 확률을 구하시오.
# X ~ N(2, 3^2)
from scipy.stats import binom
# 1) P(X < 3)
norm.cdf(3, 2, 3) = 0.63
# 2) P(2< X < 5)
norm.cdf(5, 2, 3) - norm.cdf(2,2,3) = 0.341
# 3) P(X < 3 or X > 7)
1-(norm.cdf(8, 2, 3) - norm.cdf(4,2,3)) = 0.77



# 3. LS 빅데이터 스쿨 학생들의 중간고사 점수는
# 평균이 30이고, 분산이 4인 정규분포를 따른다.
# 상위 5%에 해당하는 학생의 점수는?
import scipy.stats as stats
data = stats.norm(loc=30, scale=4)
data.ppf(0.95) = 36.57








