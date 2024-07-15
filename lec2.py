#Ctrl+Enter
#shift+ 화살표 : 블록록
a=1
a

# 파워쉘 명령어 리스트
# ls: 파일목록
# cd: 폴더 이동
# .  현재 폴더
##.. 상위 폴더

#Tab\Shift Tab 자동완성\옵션 

#Show Folder in New Window 
# 해당위치 탐색기

# cls: 화면 정리
a = 10

a = "안녕하세요!"
a
a = '"안녕하세요!"라고 아빠가 말했다.'
a
a=[1,2,3]
a

var1 = [1, 2, 3]
var1
var2 = [4, 5, 6]
var2
var1 + var2

a = '안녕하세요!'
a

b = 'LS빅데이터 스쿨!'
b

a + b
a + ' ' +  b

print(a)


1num = 3
num1 = 3
num2 = 5
num1 + num2 

a = 10
b = 3.3

print("a + b", a + b)
print("a - b", a- b )
print("a * b", a * b)
print("a / b", a / b)
print("a % b", a % b)
print("a // b", a // b)
print("a ** b =", a ** b)


#Shift + Alt + 아래화살표: 아래로 복사
#Ctrl + Alt + 아래화살표: 커서 여러 개 

(a ** 3) // 7
(a ** 3) // 7
(a ** 3) // 7
(a ** 3) // 7


a == b 
a != b
a < b
a > b
a <=b
a >= b


#2에 4승과 12453을 7로 나눈 몫을 더 해서 8로 나눴을 때 나머지
#9의 7승을 12로 나누고, 36452를 253로 나눈 나머지에 곱한 수 
# 중 큰 것은?

a = ((2**4) + (12453// 7))%8 
b = ((9**7) / 12) * (36452 % 253)
a < b

user_age = 14
is_adult = user_age >= 18
print("성인입니까?", is_adult)

#False = 3
#True = 2

a = "True"
b = TRUE
c = true
d = True

# True, False 

a = True
b = False

a and b
a or b

# True: 1
# False: 0
True + True 
True + False
False + False

# and 연산자
True and False
True and True 
False and False
False and True

# and 는 곱셈으로 치환가능
True * False
True * True 
False * False
False * True


# or 연산자
True or True
True or False
False or True
False or False

a = False
b = False
a or b
min(a + b, 1) 

a = 3
# a = a + 10
a += 10
a

a -= 4
a
a %= 3
a

a +=12
a

a **= 2
a

a //= 7
a



str1 = "hello"
str1 + str1
str1 * 3

str1 = "Hello! "

# 문자열 반복
repeated_str = str1 * 3
print("Repeated string:", repeated_str)

str1 * -2

# 정수: int(eger)
# 실수: float (double)


# 단항 연산자
x = -3
x
+x
-x
~x

# binary 
bin(1)
bin(~-4)
bin(~-3)
~0
~5

max(3, 4)
var1 =[1, 2, 3]
sum(var1)


!pip install pydataset

import pydataset
pydataset.data()

df = pydataset.data("AirPassengers")
df

#데이터
