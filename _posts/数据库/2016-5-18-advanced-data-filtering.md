---
layout: single
title: "高级数据过滤"
subtitle: "《Sams Teach Yourself SQL in 10 Minutes》第五讲"
date: 2016-5-18
author: "Anyinlover"
category: 数据库
tags:
  - 编程语言
  - SQL
---
本讲主要学习通过组合`WHERE`条件来创建复杂的搜索条件。同时还会用到`NOT`和`IN`操作符。

## `WHERE`条件组合

通过`AND`和`OR`这两个逻辑操作符，我们可以把多个`WHERE`条件组合起来。

### 使用`AND`操作符

~~~sql
SELECT prod_id, prod_price, prod_name
FROM Products
WHERE vend_id = 'DLL01' AND prod_price <= 4;
~~~

### 使用`OR`操作符

~~~sql
SELECT prod_name, prod_price
FROM Products
WHERE vend_id='DLL01' OR vend_id = 'BRS01';
~~~

### 理解操作符顺序

就像大部分其他语言一样，SQL中`AND`操作符的优先级高于`OR`操作符，要指定`OR`在前面，需要使用括号：

~~~sql
SELECT prod_name, prod_price
FROM Products
WHERE vend_id='DLL01' OR vend_id = 'BRS01'
  AND prod_price >= 10;

SELECT prod_name, prod_price
FROM Products
WHERE (vend_id='DLL01' OR vend_id='BRS01')
  AND prod_price >= 10;
~~~

### 使用`IN`操作符

~~~sql
SELECT prod_name, prod_price
FROM Products
WHERE vend_id IN('DLL01','BRS01')
ORDER BY prod_name;
~~~

### 使用`NOT`操作符

~~~sql
SELECT prod_name
FROM Products
WHERE NOT vend_id = 'DLL01'
ORDER BY prod_name;
~~~
