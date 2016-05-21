---
layout: post
title: "通配符过滤"
subtitle: "《Sams Teach Yourself SQL in 10 Minutes》第六讲"
date: 2016-5-18
author: "Anyinlover"
catalog: true
tags:
  - 编程语言
  - SQL
---
这一讲主要学习使用通配符和`LIKE`操作符来过滤数据。

## 使用`LIKE`操作符
通配符和`LIKE`操作符一块使用，可以进行模糊搜索。

### 百分号`%`通配符

`%`通配符可以表示任意数量的任意字符。注意`%`也能代表0个字符，但它匹配不到`NULL`值。

~~~sql
SELECT prod_id, prod_name
FROM Products
WHERE prod_name LIKE 'Fish%';

SELECT prod_id, prod_name
FROM Products
WHERE prod_name LIKE '%bean bag%';
~~~

### 下划线`_`通配符

`_`通配符只能表示一个字符。

~~~sql
SELECT prod_id, prod_name
FROM Products
WHERE prod_name LIKE '__ inch teddy bear';
~~~

### 方括号`[]`通配符

~~~sql
SELECT cust_contact
FROM Customers
WHERE cust_contact RLIKE '^[JM]'
ORDER BY cust_contact;
~~~

## 使用通配符的技巧

* 不要过度使用
* 尽量不要在开头使用
* 注意通配符放置位置