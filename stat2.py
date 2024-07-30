import numpy as np
import matplotlib.pyplot as plt

#예제 넘파이 배열 생성
data = np.random.rand(10)

#히스토그램 그리기
plt.clf()
plt.hist(data, bins=4, alpha=0.7, color='blue')
plt.title('Hisytogram of Numpy Vector')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# 0~1 사이 숫자 5개 발생
#표본 평균 계산하기
# 1,2 단계를 10000번 반복한 결과를 벡터로 만들기
#히스토그램 그리기

x = np.random.rand(50000).reshape(-1,5).mean(axis=1)
# 위에 코드 다른 방법 np.random.rand(10000,5).mean(axis=1)


x = np.random.rand(50000).reshape(-1,5).mean(axis=1)
plt.hist(x, bins=100, alpha=0.7, color='blue')
plt.title('Hisytogram of Numpy Vector')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(False)
plt.show()


import numpy as np
x=np.arange(33)
sum(x)/33
sum((x-16)*1/33)
(x-16)**2
np.unique((x-16)**2)

np.unique((np.arange(33) - 16)**2)*(2/33)
sum(np.unique((np.arange(33) - 16)**2)*(2/33))

(np.arange(33) - 16)**2)

# E[X^2]
sum(x**2*(1/33))

# Var(x) = E[X^2] - (E[X])^2
sum(x**2*(1/33)) - 16**2


sum((x-1.5)**1/6)

## Example
x=np.arange(4)
x
pro_x=np.array([1/6, 2/6, 2/6, 1/6])
pro_x

#기대값
Ex=sum(x * pro_x)
Exx=sum(x**2 * pro_x)

# 분산
Exx - Ex**2
sum((x - Ex)**2 * pro_x)


## Example2
x=np.arange(99)
# 1-50-1벡터터
x_1_50_1 = np.concatenate((np.arange(1,51),np.arange(49, 0, -1)))
pro_x=x_1_50_1/2500
sum(x*pro_x)
sum(x**2*pro_x)


a=sum(np.arange(1,51))
b=sum(np.arange(49, 0, -1))


import numpy as np

x=np.arange(99)
sum(x)



Ex=sum(x * pro_x)
Exx=sum(x**2 * pro_x)


#Example3 x: 0, 2, 4, 6
x=np.arange(4)*2
x
pro_x=np.array([1/6, 2/6, 2/6, 1/6])
pro_x

#기대값
Ex=sum(x * pro_x)
Exx=sum(x**2 * pro_x)

# 분산
Exx - Ex**2
sum((x - Ex)**2 * pro_x)

4*0.916

9.52**2/16
np.sqrt(9.52**2/16)


