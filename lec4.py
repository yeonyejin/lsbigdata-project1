a = (1, 2, 3) # a = 1,2,3
a
a=1,2,3
a

a = [1, 2, 3]

b = a
b

a[1] = 4
a

b

id(a)
id(b)

# deep copy
a = [1, 2, 3]
a

id(a)


b = a [:]
b = a.copy()

id(b)

a[1] = 4
a
b


