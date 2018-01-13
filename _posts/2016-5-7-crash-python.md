---
layout: single
title: "Python速成"
subtitle: "摘自《Data Science from Scratch》"
date: 2016-5-7
author: "Anyinlover"
category: 笔记
tags:
  - 编程语言
  - 语法
  - Python
---
## 得到python
从python官网可以下载python，对于数据科学家来说，更方便的是直接下载Anaconda分发包。

pip是python包管理工具，利用它可以很方便的寻找安装python包。

Ipython是一个增强版的python shell。和它配套使用的还有一个jupyter notebook，有很好的交互界面，是我目前使用python的主力战场。

## python之禅
下面描述了python之禅：python的设计原则，

>Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!

其中最常见得一条就是在python中只有一种最好的实现方法，常常被称为Pythonic。

## 空格格式化
python中用缩进来表示代码块。

```python
for i in [1, 2, 3, 4, 5]:
    print(i)
    for j in [1, 2, 3, 4, 5]:
        print(j)
        print(i+j)
    print(i)
print("done looping")
```
这让python代码非常具有可读性，但也因此需要注意空格的使用。空格在括号内是被忽略的，比如用在下面的长代码中：

```python
long_winded_computation=(1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11
                        + 12 + 13 + 14 + 15 + 16 + 17 + 18 + 19 + 20)
```

或者让代码变得更加可读：

```python
list_of_lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
easier_to_read_list_of_lists = [ [1, 2, 3],
                                 [4, 5, 6],
                                 [7, 8, 9] ]
```

反斜杠可以用来断行，但一般很少用：

```python
two_plus_three = 2 + \
                 3
```

使用空格的一个缺陷是代码复制时会变形，在Ipython中有一个函数`%paste`可以解决这个问题。

## 模块

python的很多功能都需要导入模块来实现。

一种简单的方法是把模块整个导入：

```python
import re
my_regex = re.compile("[0-9]+", re.I)
```
这样导入后只能通过加前缀`re.`来调用模块内的函数。

如果代码中这个标识符已经被使用，可以用别名来替代：

```python
import re as regex
my_regex = regex.compile("[0-9]+", regex.I)
```
当模块名称不清晰或者过于冗长时，也可以用别名：

```python
import matplotlib.pyplot as plt
```

如果只需要使用到模块中一些特定的子模块或函数，可以显性的导入：

```python
from collections import defaultdict, Counter
lookup = defaultdict(int)
my_counter = Counter()
```

另外也有一种做法就是把模块内的值全都导入，但这样会有命名冲突的风险，不推荐这样处理，比如下面`re.match`覆盖了原来的`match`。

```python
match = 10
from re import *
print match
```

## 算术

在python2中除法是整数除法，到python3中已经是浮点数除法了。`5/2`再也不会等于2了。

## 函数

一个函数把输入转化成输出，在python中使用`def`来定义函数：

```python
def double(x):
    """this is where you put an optional docstring
    that explains what the function does.
    for example, this function multiplies its input by 2"""
    return x * 2
```
在python中函数是原始类的，也就是函数可以被赋给变量，也可以作为参数传给另一个函数。

```python
def apply_to_one(f):
    """calls the function f with 1 as its argument"""
    return f(1)

my_double = double # refers to the previously defined function
x = apply_to_one(my_double) # equals 2
```
使用lambda语法可以创造匿名函数：

```python
y = apply_to_one(lambda x: x + 4) # equals 5
```
lambda函数也能赋给变量，但多数人推荐下面的传统做法：

```python
another_double = lambda x: 2 * x # don't do this
def another_double(x): return 2 * x # do this instead
```
函数参数可以指定默认值：

```python
def my_print(message="my default message"):
    print(message)

my_print("hello") # prints 'hello'
my_print()		 # prints 'my default message'
```
传参时带上变量名有时候也很有用：

```python
def subtract(a=0, b=0):
    return a - b

subtract(10, 5) # returns 5
subtract(0, 5)  # returns -5
subtract(b=10)  # same as previous
```

## 字符串

字符串可以用单引号或双引号来定义：

```python
single_quoted_string='data science'
double_quoted_string="data science"
```

python使用反斜杠来编码特殊字符：

```python
tab_string = "\t" # represents the tab character
len(tab_string)
```
如果仍然想让反斜杠保持为反斜杠，可以用`r""`创造原始字符串：

```python
not_tab_string=r"\t" # represents the characters '\' and 't'
len(not_tab_string)
```
可以三个双引号来创建多行的字符串：

```python
muti_line_string = """This is the first line.
and this is the second line
and this is the third line"""
```

## 异常

在python中可以用`try`和`except`来处理异常，防止程序报错：

```python
try:
    print(0/0)
except ZeroDivisionError:
    print("cannot divide by zero")
```
虽然在很多语言中异常被认为是不好的，但在python中偶尔的使用异常可让代码保持整洁。


## 列表

列表是python中最基础的数据结构。列表是一个有序集合。

```python
integer_list = [1, 2, 3]
heterogeneous_list = ["string", 0.1, True]
list_of_lists = [ integer_list, heterogeneous_list, []]

list_length = len(integer_list) # equals 3
list_sum = sum(integer_list) # equals 6
```
可以用方括号来对第n个元素取值或赋值

```python
x = list(range(10)) # is the list [0, 1, ..., 9]
zero = x[0] # equals 0, lists are 0-indexed
one = x[1] # equals 1
nine = x[-1] # equals 9, 'Pythonic' for last element
eight = x[-2] # equals 8, 'Pythonic' for next-to-last element
x[0] = -1 # now x is [-1, 1, 2, 3, ..., 9]
```
方括号也可以用来切片：

```python
first_three = x[:3] # [-1, 1, 2]
three_to_end = x[3:] # [3, 4, ..., 9]
one_to_four = x[1:5] # [1, 2, 3, 4]
last_three = x[-3:] # [7, 8, 9]
without_first_and_last = x[1:-1] # [1, 2, ..., 8]
copy_of_x = x[:] # [-1, 1, 2, ..., 9]
```

python有一个`in`操作来验证元素是否在列表中：

```python
1 in [1, 2, 3] # True
0 in [1, 2, 3] # False
```
这个操作会依次去进行比较，除非列表比较短，否则会很费时。

把列表串联起来是容易的：

```python
x = [1, 2, 3]
x.extend([4, 5, 6]) # x is now [1,2,3,4,5,6]
```

如果不想改变x那么可以采用列表加法：

```python
x = [1, 2, 3]
y = x + [4, 5, 6] # y is [1,2,3,4,5,6]; x is unchanged
```

最常见的还是每次添加一个元素：

```python
x = [1, 2, 3]
x.append(0) # x is now [1, 2, 3, 0]
y = x[-1] # equals 0
z = len(x) # equals 4
```
如果知道列表元素个数还可以使用解包：

```python
x, y = [1, 2] # now x is 1, y is 2
```

对于不需要使用的元素可以用下划线替代：

```python
_, y = [1, 2] # now y==2, didn't care about the first element
```

## 元组
元组与列表类似，但是不可变的。列表很多不涉及到修改元素的操作都可以用在元组上。元组用小括号来表示：

```python
my_list = [1, 2]
my_tuple = (1, 2)
other_tuple = 3, 4
my_list[1] = 3 # my_list is now [1, 3]

try:
    my_tuple[1] = 3
except TypeError:
    print("cannot modify a tuple")
```
元组可以被用来从函数返回多值。

```python
def sum_and_product(x, y):
    return (x+y), (x*y)

sp = sum_and_product(2, 3) # equals (5, 6)
s, p = sum_and_product(5, 10) # s is 15, p is 50
```
元组和列表都可以用来赋多值：

```python
x, y = 1, 2 # now x is 1, y is 2
x, y = y, x # Pythonic way to swap variables; now x is 2, y is 1
```

## 字典

字典是另一种基本数据结构，将键和值相对应，方便快速查找。

```python
empty_dict = {} # Pythonic
empty_dict2 = dict() # less Pythonic
grades = {"Joel": 80, "Tim": 95} # dictionary literal
```
可以通过键来查找值。

```python
joels_grade = grades["Joel"] # equals 80
```
假如键不在字典中，就会报错`KeyError`：

```python
try:
    kates_grade = grades["Kate"]
except KeyError:
    print("no grad for Kate!")
```

可以用下面的方法检查键是否存在：

```python
joel_has_grade = "Joel" in grades # True
kate_has_grade = "Kate" in grades # False
```
字典有一个`get`方法可以在键不存在时返回一个默认值：

```python
joels_grade = grades.get("Joel", 0) # equals 80
kates_grade = grades.get("Kate", 0) # equals 0
no_ones_grade = grades.get("No One") # default default is None
```
用中括号可以对键赋值：

```python
grades["Tim"] = 99 # replaces the old value
grades["Kate"] = 100 # adds a third entry
num_students = len(grades) # equals 3
```
字典可以很方便的表示结构性数据：

```python
tweet = {
    "user" : "joelgrus",
    "text" : "Data Science is Awesome",
    "retweet_count" : 100,
    "hashtags" : ["#data", "#science", "#datascience",
                  "#awesome", "#yolo"]
}
```
除了查找特定的键，还可以查找下面的值：

```python
tweet_keys = tweet.keys() # list of keys
tweet_values = tweet.values() # list of values
tweet_items = tweet.items() # list of (key, value) tuples

"user" in tweet_keys # True, but uses a slow list in
"user" in tweet # more Pythonic, uses faster dict in
"joelgrus" in tweet_values
```
字典的键必须是不可变得，因此，不能将列表作为键，如果需要多键，可以使用元组或者把键转化成字符串。


### 默认字典

假如需要对文档的单词进行计数，常规的方法如下：

```python
document = "Hello how are you. Every one has a mi ring"
word_counts = {}
for word in document:
    if word in word_counts:
        word_counts[word] += 1
    else:
        word_counts[word] = 1
word_counts
```
或者使用异常机制来处理首次添加：

```python
word_counts = {}
for word in document:
    try:
        word_counts[word] += 1
    except KeyError:
        word_counts[word] = 1
```

第三种方法是使用`get`方法：

```python
word_counts = {}
for word in document:
    previous_count = word_counts.get(word, 0)
    word_counts[word] = previous_count + 1
```
但这三种方法看起来都有些别扭，更好的方法是使用`defaultdict`。

```python
from collections import defaultdict

word_counts = defaultdict(int) # int() produces 0
for word in document:
    word_counts[word] += 1

```
除了应用在整数上，还能应用在列表或字典，甚至自己的函数上：

```python
dd_list = defaultdict(list) # list() produces an empty list
dd_list[2].append(1) # now dd_list contains {2: [1]}

dd_dict = defaultdict(dict) # dict() produces an empty dict
dd_dict["Joel"]["City"] = "Seattle" # {"Joel": {"City":"Seattle"}}

dd_pair = defaultdict(lambda: [0, 0])
dd_pair[2][1] = 1 # {2: [0,1]}
```
所以这是一种很方便的省却检查键是否存在的解决方案。

### 计数

计数构造了一个字典，并将键映射到个数：

```python
from collections import Counter
c = Counter([0, 1, 2, 0]) # c is (basically) {0:2, 1:1, 2:1}
```
因此这可以方便的解决我们的单词统计问题：

```python
word_counts = Counter(document)
```
计数示例有一个`most_common`方法很有效：

```python
# print the 10 most common words and their counts
for word, count in word_counts.most_common(10):
    print(word, count)
```


## 集合

集合是另一种数据结构，表达了离散元素的集合。

```python
s = set()
s.add(1) # s is now { 1 }
s.add(2) # s is now {1, 2}
s.add(2) # s is still { 1, 2}
x = len(s) # equals 2
y = 2 in s # equals True
z = 3 in s # equals False
```
使用集合主要有两个理由。第一个是`in`操作在集合中应用非常快，类似于字典中键的查找：

```python
stopwords_list = ["a","an","at"] + ["manyotherwords"] + ["yet", "you"]
"zip" in stopwords_list # False, but have to check every element
stopwords_set = set(stopwords_list)
"zip" in stopwords_set # very fast to check
```
第二个理由是找到集合中不同项：

```python
item_list = [1, 2, 3, 1, 2, 3]
num_items = len(item_list) # 6
item_set = set(item_list)  # {1，2，3}
num_distinct_items = len(item_set) # 3
distinct_items = list(item_set) # [1, 2, 3]
```
集合比字典和列表使用的少得多。

## 控制流

像大部分编程语言，python可以使用`if`：

```python
if 1 > 2:
    message = "if only 1 were greater than two..."
elif 1 > 3:
    message = "elif stands for 'else if'"
else:
    message = "when all else fails use else (if you want to)"
```
有时候还可以写成一行：

```python
parity = "even" if x % 2 == 0 else "odd"
```
Python还有一个`while`循环：

```python
x = 0
while x < 10:
    print(x, "is less than 10")
    x += 1
```
但更常用的是`for`和`in`：

```python
for x in range(10):
    print(x, "is less than 10")
```
如果要控制更复杂的逻辑，可以使用`continue`和`break`：

```python
for x in range(10):
    if x == 3:
        continue # go immediately to the next iteration
    if x == 5:
        break # quit the loop entirely
    print(x)
```

## 真值

布尔型在python中也和其他语言一样，除了是大写的：

```python
one_is_less_than_two = 1 < 2 # equals True
true_equals_false = True == False # equals False
```
`None`表示一个不存在的值，类似于其他语言的`null`：

```python
x = None
print(x==None) # prints True, but is not Pythonic
print(x is None) # prints True, and is Pythonic
```
Python在它期望布尔值时允许任何值，下面的值都被认为是假值：

* False
* None
* [](空列表）
* {}（空字典）
* ""
* set()
* 0
* 0.0

其他任何值都被认为是真值，这在`if`中测试空字符串或空字典时格外有用

```python
s = some_function_that_returns_a_string()
if s:
    first_char = s[0]
else:
    first_char = ""
```
一种更简单的方法是：

```python
first_char = s and s[0]
```
因为`and`操作会在第一个值为真时返回第二个值，第一个值为假时返回第一个值。下面用了类似的技巧：

```python
safe_x = x or 0
```
python还有一个`all`函数，输入一个列表，当列表所有元素是真时返回真，还有一个`any`函数，至少有一个元素是真时返回真：

```python
all([True, 1, {3}]) # True
all([True, 1, {}]) # False, {} is falsy
any([True, 1, {}]) # True, True is truthy
all([]) # True, no falsy elements in the list
any([]) # False, no truthy elements in the list
```

## 排序

每个python列表都有一个`sort`方法，如果不想改变原列表，也可以使用`sorted`函数，会返回一个新列表：

```python
x = [4,1,2,3]
y = sorted(x) # is [1,2,3,4], x is unchanged
x.sort() # now x is [1,2,3,4]
```
默认情况下，列表排序会从小到大进行，如果想要从大到小排序，可以指定`reverse=True`参数。除了比较元素本身，还可以用参数`key`指定函数。

```python
# sort the list by absolute value from largest to smallest
x = sorted([-4,1,-2,3], key=abs, reverse=True) # is [-4,3,-2,1]

# sort the words and counts from highest count to lowest
wc = sorted(word_counts.items(),
           key=lambda pair: pair[1],
           reverse=True)
```

## 列表生成器

列表生成器可以很方便的将列表转换成另一个列表，这是典型的Pythonic方法。

```python
even_numbers = [x for x in range(5) if x % 2 == 0] # [0, 2, 4]
squares = [x * x for x in range(5)] # [0, 1, 4, 9, 16]
even_squares = [x * x for x in even_numbers] # [0, 4, 16]
```
这种用法也能把列表转换成字典或集合：

```python
square_dict = { x: x * x for x in range(5)} # {0:0, 1:1, 2:4, 3:9, 4:16}
squre_set = ( x * x for x in [1, -1]) # {1}
```
如果不需要使用到列表中的值，就用下划线替代：

```python
zeroes = [0 for _ in even_numbers] # has the same length as even_numbers
```
列表生成器还能使用多个`for`：

```python
pairs = [(x, y)
        for x in range(10)
        for y in range(10)]
```
后面的`for`可以应用前面的`for`结果：

```python
increasing_pairs = [(x, y) # only pairs with x < y
                   for x in range(10) # range(lo, hi) equals
                   for y in range(x + 1, 10)] # [lo, lo + 1, ..., hi - 1]
```

## 生成器和迭代器
列表有一个问题就是容易变得很大，如果每次只需要使用很少的元素，计算整个列表就会很低效，还可能耗尽内存。

生成器是一种可以迭代（常常用`for`），但值只有在需要使用时才计算。

```python
def lazy_range(n):
    """a lazy version of range"""
    i = 0
    while i < n:
        yield i
        i += 1
```
下面的循环就是依次使用yield里的值直到结束：

```python
for i in lazy_range(10):
    # do_something_with(i)
    print(i)
```
在python3中，`range`本身就是个生成器，这意味着生成器可以创建无限的序列：

```python
def natural_numbers():
    """returns 1, 2, 3, ..."""
    n = 1
    while True:
        yield n
        n += 1
```
当然在实际应用中需要和`break`配合使用。

生成器的一个缺点就是只能迭代一次，如果需要迭代多次，那就需要新建生成器或者使用列表。

另一种创建生成器的方法是使用小括号的列表生成器：

```python
lazy_evens_below_20 = (i for i in lazy_range(20) if i % 2 == 0)
```
前面提到`dict`有一个`items()`方法可以返回键值对。更常用的我们使用它的`iteritems()`方法，这就是一个生成器方法。

## 随机数
使用`random`模块，可以很方便的生成随机数。

```python
import random
four_uniform_randoms = [random.random() for _ in range(4)]

# [0.8444218515250481, # random.random() produces numbers
#  0.7579544029403025, # uniformly between 0 and 1
#  0.420571580830845, # it's the random function we'll use
#  0.25891675029296335] # most often
```
`random`模块实际上产生的是根据内部状态产生伪随机数，如果想要得到可重复结果，可以使用`random.seed`：

```python
random.seed(10) # set the seed to 10
print(random.random()) # 0.57140259469
random.seed(10) # reset the seed to 10
print(random.random()) # 0.57140259469 again
```
有时也会用到`random.randrange`，可以传1或2个参数，返回一个从`range()`中随机抽取的值：

```python
random.randrange(10) # choose randomly from range(10) = [0, 1, ..., 9]
random.randrange(3, 6) # choose randomly from range(3, 6) = [3, 4, 5]
```
另外还有一些方法也很有用，`random.shuffle`将列表元素随机排列：

```python
up_to_ten = list(range(10))
random.shuffle(up_to_ten)
print(up_to_ten)
# [2, 5, 1, 9, 7, 3, 8, 6, 4, 0]
```
想要随机从列表中抽取一个元素，可以使用`random.choice`：

```python
my_best_friend = random.choice(["Alice", "Bob", "Charlie"])
```
如果想要随机抽取一组不重复样本，使用`random.sample`

```python
lottery_numbers = range(60)
winning_numbers = random.sample(lottery_numbers, 6)
```
如果想要抽取重复样本，还是使用`random.choice`：

```python
four_with_replacement = [random.choice(range(10)) for _ in range(4)]
```

## 正则表达式

正则表达式在搜索文本时很有用，但其相当的复杂，这里只简单的列举几例：

```python
import re

print(all([ # all of these are true, because
        not re.match("a", "cat"), # * 'cat' doesn't start with 'a'
        re.search("a", "cat"), # * 'cat' has an 'a' in it
        not re.search("c", "dog"), # * 'dog' doesn't have a 'c' in it
        3 == len(re.split("[ab]", "carbs")), # * split on a or b to ['c', 'r', 's']
        "R-D-" == re.sub("[0-9]", "-", "R2D2") # * replace digits with dashes
    ])) # print True
```

## 面向对象编程
和其他语言一样，python允许你创建类来封装数据和函数。类的使用可以让我们的代码变得简洁。解释类最好的方法还是亲自构造一个。

假设现在我们没有内置的集合，我们来自己构造一个集合类。

我们的集合类需要有些什么成员函数？我们需要去添加元素，删除元素，验证元素是否存在，因此类构造如下：

```python
# by convention, we give classes PascalCase names
class Set:

    # thest are the member functions
    # every one takes a first parameter "self" (another convention)
    # that refers to the particular Set object being used

    def __init__(self, values=None):
        """This is the constructor.
        It gets called when you create a new Set.
        You would use it like
        s1 = Set() # empty set
        s2 = Set([1,2,3,4]) # initialize with values"""

        self.dict = {} # each instance if Set has its own dict property
        					# which is what we'll use to track memberships
        if values is not None:
            for value in values:
                self.add(value)

    def __repr__(self):
        """this is the string representation of a Set object
        if you type is at the Python prompt or pass it to str()"""
        return "Set: " + str(self.dict.keys())

    # we'll represent membership by being a key in self.dict with
    # value True
    def add(self, value):
        self.dict[value] = True

    # value is in the Set if it's a key in the dictionary
    def contains(self, value):
        return value in self.dict

    def remove(self, value):
        del self.dict[value]
```
构建完类后，我们可以像内置类那样进行调用：

```python
s = Set([1,2,3])
s.add(4)
print(s.contains(4)) # True
s.remove(3)
print(s.contains(3)) # False
```

## 函数工具

有时候我们想要部分应用函数来创建一个新函数：

```python
def exp(base, power):
    return base ** power
```
现在想创建一个`two_to_the`函数，输入`power`，输出`exp(2, power)`。用下面这种方法可行，但看起来怪怪的：

```python
def two_to_the(power):
    return exp(2, power)
```
另一种方法是使用`functools.partial`

```python
from functools import partial
two_to_the = partial(exp, 2) # is now a function of one variable
print(two_to_the(3)) # 8
```
也可以通过指定参数名来填充后一个参数：

```python
square_of = partial(exp, power=2)
square_of(3) # 9
```
我们偶尔也会使用到`map`,`reduce`和`filter`来提供与列表生成器类似的功能。

```python
def double(x):
    return 2 * x

xs = [1, 2, 3, 4]
twice_xs = [double(x) for x in xs] # [2, 4, 6, 8]
twice_xs = map(double, xs) # same as above
list_doubler = partial(map, double) # *function* that doubles a list
twice_xs = list_doubler(xs) # again [2, 4, 6, 8]
```
`map`也能使用多列表来来传递多参：

```python
def multiply(x, y): return x * y
products = map(multiply, [1, 2], [4, 5]) # [1 * 4, 2 * 5] = [4, 10]
```
`filter`提供了列表生成器中类似`if`的功能

```python
def is_even(x):
    """True if x is even, False if x is odd"""
    return x % 2 == 0

x_evens = [x for x in xs if is_even(x)] # [2, 4]
x_evens = filter(is_even, xs) # same as above
list_evener = partial(filter, is_even) # *function* that filters a list
x_evens = list_evener(xs) # again [2, 4]
```
在python3中，`reduce`已经不是内置函数了，需要从`functools`中导入，更推荐用列表生成器来处理：

```python
from functools import reduce
x_product = reduce(multiply, xs)
list_product = partial(reduce, multiply)
x_product = list_product(xs)
```
此外，在python3中，这三个函数都产生生成器而不是之前的列表了。

## 遍历

很多时候，经常需要同时遍历列表中的元素及其序号：

```python
# not Pythonic
for i in range(len(documents)):
    document = documents[i]
    do_something(i, document)

# also not Pythonic
i = 0
for document in documents:
    do_something(i, document)
    i += 1

Pythonic式的解决办法是是使用`enumerate`:

# Pythonic
for i, document in enumerate(documents):
    do_something(i, document)
```
类似的，如果我们仅仅想使用序号：

```python
for i in range(len(documents)): do_something(i) # not Pythonic
for i, _ in enumerate(documents): do_something(i) # Pythonic
```

## 打包与解压

我们常常需要把两个列表打包起来，`zip`提供了将多个列表转换成一个含元组的列表的功能：

```python
list1 = ['a', 'b', 'c']
list2 = [1, 2, 3]
zip(list1, list2) # is [('a', 1), ('b', 2), ('c', 3)]
```
如果列表的长度不一致，那就在最短列表处停止。

还可以解压回来，代码看起来比较奇怪：

```python
pairs = [('a', 1), ('b', 2), ('c', 3)]
letters, numbers = zip(*pairs)
```
星号实现了参数解压的功能，把每个列表元素作为`zip`的参数传入，其相当于下面这个调用：

```python
zip(('a', 1), ('b', 2), ('c', 3))
```
最终返回`[('a','b','c'),('1','2','3')]`

这种技巧可以应用在任何函数上：

```python
def add(a, b): return a + b

add(1,2) # returns 3
add([1,2]) # TypeError!
add(*[1,2]) # returns 3
```
有没有好奇怪的感觉~~


## 无名参数和关键字参数

首先来创建一个高阶函数：

```python
def doubler(f):
    def g(x):
        return 2 * f(x)
    return g
```

可以调用如下：

```python
def f1(x):
    return x + 1

g = doubler(f1)
print(g(3)) # 8 (==(3+1)*2)
print(g(-1)) # 0 (==(-1+1)*2)
```

但假如f不止一个参数时，传入g时就会报错：

```python
def f2(x, y):
    return x + y

g = doubler(f2)
print(g(1,2)) # TypeError: g() takes exactly 1 argument (2 given)
```

我们需要一种方法来传递任意参数给函数，我们还是用上了前面的参数解压的技巧：

```python
def magic(*args, **kwargs):
    print("unnamed args:", args)
    print("keyword args:", kwargs)

magic(1, 2, key="word", key2="word2")

# prints
# unnamed args: (1, 2)
# keyword args: {'key': 'word', 'key2': 'word2'}
```
当我们这样定义时，`args`代表的是无名参数构成的元组，`kwargs`代表的是关键字参数构造的字典。另一方面，如果用列表或元组以及字典来传参，函数也可以构造如下：

```python
def other_way_magic(x, y, z):
    return x + y + z

x_y_list = [1, 2]
z_dict = {"z": 3}
other_way_magic(*x_y_list, **z_dict) # 6
```
这种技巧我们只在构造高阶函数时使用到，下面是正确的g函数：

```python
def doubler_correct(f):
    """works no matter what kind of inputs f expects"""
    def g(*args, **kwargs):
        """whatever arguments g is supplied, pass them"""
        return 2 * f(*args, **kwargs)
    return g

g = doubler_correct(f2)
print(g(1, 2)) # 6
```
