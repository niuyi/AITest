print('hello py_test')

print('==============================================================================')
print('\\\t\\')
print(r'\\\t\\')
print('==============================================================================')
print('this is \r\n a test')
print('''this is
a test''')
print('==============================================================================')
if True:
    print('t')
else:
    print('f')
print('==============================================================================')

if None is None:
    print('is none')
print('==============================================================================')
print('10/3', 10/3)
print('10//3', 10//3)
print('10%3', 10%3)
print('==============================================================================')
#一种有序列表叫元组：tuple。tuple和list非常类似，但是tuple一旦初始化就不能修改，比如同样是列出同学的名字：
classmates = ('Michael', 'Bob', 'Tracy')
#所以，只有1个元素的tuple定义时必须加一个逗号,，来消除歧义：
t = (1)
print(t)
t = (1,)
print(t)
print('==============================================================================')
a = 1000
if a > 100:
    print('bigger')

b = '1000'
if int(b) > 100:
    print('bigger')
print('==============================================================================')
for x in range(5):
    print(x)
print('==============================================================================')
map = {
    'a': 0,
    'b': 1
}
#要避免key不存在的错误，有两种办法，一是通过in判断key是否存在：
#二是通过dict提供的get()方法，如果key不存在，可以返回None，或者自己指定的value：
print(map['a'])
print('c' in map)
print(map.get('c'))
print('==============================================================================')
list = [1, 1, 2, 2, 3, 3]
s = set(list)
print(s)
print('==============================================================================')
a = 5
if a > 18:
    pass
else:
    print('test')
print('==============================================================================')
a = 100
print(isinstance(a, int))
print(isinstance(a, float))
print(isinstance(a, (int, float)))
print(isinstance(a, str))
print('==============================================================================')
#函数可以同时返回多个值，但其实就是一个tuple。
print('==============================================================================')
def sum(*number):
    result = 0
    for x in number:
        result = result + x

    return result

print(sum(1))
print(sum(1, 2))
print(sum(1, 2, 5))
list = [1,2,3]
#*nums表示把nums这个list的所有元素作为可变参数传进去。这种写法相当有用，而且很常见。
print(sum(*list))
print('==============================================================================')
def show_info(name, age, **key):
    print('name', name, 'age', age, 'key', key)

show_info('mike', 18, type='a', id = 100)

def show_info2(name, age, *, type, id):
    print(name, age, type, id)

def show_info3(*, type, id):
    print(type, id)

show_info2('mike', 18, type='newtype', id='1000')
# show_info2('mike', 18, type='oldtype') #??
show_info3(type='mytype', id=10000)
print('==============================================================================')
map = {
    'a':100,
    'b':200,
    'c':300
}

for k,v in map.items():
    print(k, v)

from collections import Iterable
print(isinstance(map, Iterable))

for i, value in enumerate(['a', 'b', 'c']):
    print(i, value)
print('==============================================================================')
print([x*x for x in range(1, 11)])
print([x*x for x in range(1, 11) if x%2 == 0])

g = (x*x for x in range(1, 11))
for n in g:
    print(n)