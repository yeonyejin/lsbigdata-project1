#균일분포 (3, 7)에서 20개의 표본을 뽑아서 분산을 2가지 방법으로 추정해보세요.
#X ~ 균일분포U(a , b)
#loc:a, scale: b-a
from scipy.stats import uniform
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

x=uniform.rvs(loc=3, scale=4, size=20*1000)
x = x.reshape(-1, 20)
x

s_2=np.var(x, ddof=1, axis=1)

sns.histplot(s_2)
plt.show()
plt.clf()


#n-1로 나눈 것을 s_2, n으로 나눈 것을 k_2로 정의하고,
x=uniform.rvs(loc=3, scale=4, size=20*1000)
x = x.reshape(-1, 20)
x
s_2=np.var(x, ddof=1)
s_2

x=uniform.rvs(loc=3, scale=4, size=20*1000)
x = x.reshape(-1, 20)
x.shape
k_2=np.var(x)
k_2


#s_2의 분포와 k_2의 분포를 그려주세요! (10000개 사용)

from scipy.stats import uniform
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

x=uniform.rvs(loc=3, scale=4, size=20*1000)
x = x.reshape(-1, 20)
x.shape
s_2=np.var(x, ddof=1, axis=1)

sns.histplot(s_2)
plt.show()
plt.clf()



from scipy.stats import uniform
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

x=uniform.rvs(loc=3, scale=4, size=20*1000)
x = x.reshape(-1, 20)
x.shape
k_2=np.var(x, ddof=0, axis=1)

sns.histplot(k_2)
plt.show()
plt.clf()


#각 분포 그래프에 모분산의 위치에 녹색 막대를 그려주세요.

from scipy.stats import uniform
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

x=uniform.rvs(loc=3, scale=4, size=20*1000)
x = x.reshape(-1, 20)
x.shape
s_2=np.var(x, ddof=1, axis=1)

sns.histplot(s_2)
plt.axvline(x.var(), color='green',
            linestyle='-', linewidth=2)
plt.show()
plt.clf()



from scipy.stats import uniform
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

x=uniform.rvs(loc=3, scale=4, size=20*1000)
x = x.reshape(-1, 20)
x.shape
k_2=np.var(x)

sns.histplot(k_2)
plt.axvline(x.var(), color='green',
            linestyle='-', linewidth=2)
plt.show()
plt.clf()

#결과를 살펴보고, 왜 n-1로 나눈 것을 
#분산을 추정하는 지표로 사용하는 것이 타당한지 써주세요!
