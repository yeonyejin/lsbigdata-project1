# 행렬

import numpy as np


#두 개의 벡터를 합쳐 행렬 생성


matrix = np.column_stack(
    (np.arange((1,5),
    np.arange(12, 16))
)
print("행렬:\n", matrix)

matrix = np.column_stack((np.arange(1, 5),
np.arange(12, 16)))
print("행렬:\n", matrix)


np.zeros(5)
np.zeros([5,4])
np.arange(1,5).reshape((2,2))
np.arange(1,7).reshape((2,3))
# -1 통해서 크기를 자동으로 결정할 수 있음
np.arange(1,7).reshape((2,-1))

# Q. 0에서부터 99까지 수 중 랜덤하게 50개 숫자를 뽑아서
# 5 by 10 행렬 만드세요. (정수수)
np.random.seed(2024)
a = np.random.randint(0,100, 50).reshape((5,-1))
a
mat_a = np.arange(1,21).reshape((4,5), order="F")
mat_a

#인덱싱
mat_a[0,0]
mat_a[1,1]
mat_a[0:2,3]
mat_a[1:3,1:4]


#행자리, 열자리 비어있는 경우 전체 행, 또는 열 선택
mat_a[3,]
mat_a[3,:]
mat_a[3,::2]

#짝수 행만 선택하려면?
mat_b = np.arange(1,101).reshape((20,-1))
mat_b[1::2,:]


mat_b[[1,4,6,14],]

x = np.arange(1, 11).reshape((5, 2)) * 2
filtered_elements = x[[True, True, False, False, True], 0]


mat_b[:,1]                 #벡터
mat_b[:,1].reshape((-1,1)) #행렬
mat_b[:,(1,)]              #행렬
mat_b[:,[1]]
mat_b[:,1:2]

#필터링
mat_b[mat_b[:,1] % 7 == 0, :]



#사진
import numpy as np
import matplotlib.pyplot as plt

#난수 생성하여 3x3 크기의 행렬 생성
np.random.seed(2024)
img1 = np.random.rand(3,3)
print("이미지 행렬 img1:\n", img1)

plt.imshow(img1, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()


a = np.random.randint(0,256, 20).reshape((4,-1))
a/255
plt.imshow(a)



x = np.arange(1, 11).reshape((5, 2)) * 2
print("원래 행렬 x:\n", x)


transposed_x = x.transpose()


import urllib.request

img_url = "https://bit.ly/3ErnM2Q"
urllib.request.urlretrieve(img_url, "img/jelly.png")

#!pip install imageio
import imageio
import numpy as np
# 이미지 읽기
jelly = imageio.imread("img/jelly.png")
print("이미지 클래스:", type(jelly))
print("이미지 차원:", jelly.shape)
print("이미지 첫 4x4 픽셀, 첫 번째 채널:\n", jelly[:4, :4, 0])

len(jelly)
jelly.shape
jelly[:, :, 0].shape
jelly[:, :, 1]
jelly[:, :, 2]
jelly[:, :, 3]

jelly[:, :, 0].shape
jelly[:, :, 0].transpose().shape


plt.imshow(jelly)
plt.imshow(jelly[:,:,0].transpose())
plt.imshow(jelly[:,:,0]) #R
plt.imshow(jelly[:,:,1]) #G
plt.imshow(jelly[:,:,2]) #B
plt.imshow(jelly[:,:,3]) #투명도도
plt.axis('off') #축 정보 없애기기
plt.show()
plt.clf()

#3차원 배열

# 두 개의 2x3 행렬 생성
mat1 = np.arange(1, 7).reshape(2, 3)
mat2 = np.arange(7, 13).reshape(2, 3)

my_array = np.array([mat1, mat2])
my_array.shape

first_slice = my_array[0, :, :]
first_slice

filtered_array = my_array[:, :, :-1]
filtered_array

filtered_array = my_array[:, :, [0,2]]

filtered_array = my_array[:, 0, :]

filtered_array = my_array[0, 1, 1:3 ]
filtered_array = my_array[0, 1, [1,2] ]


mat_x = np.arange(1,101).reshape((5, 5, 4))
mat_y = np.arange(1,100).reshape((-1, 3, 3))
mat_y
len(mat_y)


my_array2 = np.array([my_array, my_array])
my_array[0, :, :, :]
my_array2
my_array2.shape


#넘파이 배열 메서드
a = np.array([[1, 2,3], [4, 5, 6]])

a.sum()
a.sum(axis=0)
a.sum(axis=1)


a.mean()
a.mean(axis=0)
a.mean(axis=1)

mat_b=np.random.randint(0, 100, 50).reshape((5, -1))
mat_b

#가장 큰 수는?
mat_b.max()

#행별로 가장 큰 수는?
mat_b.max(axis=1)

#열별 가장 큰 수는?
mat_b.max(axis=0)

a=np.array([1, 3, 2, 5]).reshape((2,2))
a.cumsum()
a.cumprod()
a

mat_b.sum()
mat_b.cumsum(axis=1)
mat_b.cumprod(axis=1)

mat_b.reshape((2, 5, 5)).flatten()
mat_b.flatten()

d = np.array([1, 2, 3, 4, 5])
d.clip(2,4)
d.tolist()



