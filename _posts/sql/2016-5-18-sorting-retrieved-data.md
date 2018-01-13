---
layout: single
title: "排序数据"
subtitle: "《Sams Teach Yourself SQL in 10 Minutes》第三讲"
date: 2016-5-18
author: "Anyinlover"
category: 笔记
tags:
  - 编程语言
  - SQL
---

这一讲主要学习使用`SELECT`语句中的`ORDER BY`条件来对获取到的数据进行排序。

## 数据排序

~~~sql
SELECT prod_name
FROM Products
ORDER BY prod_name;
~~~

## 多列排序

~~~sql
SELECT prod_id, prod_price, prod_name
FROM Products
ORDER BY prod_price, prod_name;
~~~

## 通过列位置排序

~~~sql
SELECT prod_id, prod_price, prod_name
FROM Products
ORDER BY 2, 3;
~~~

通过列位置排序可以省略列名，但同时也会有误选，更改，不能指定`SELECT`外列名等缺点。

## 指定排序方向

默认的排序方向是从A到Z，如果想要反方向排序，可以使用关键字`DESC`。

~~~sql
SELECT prod_id, prod_price, prod_name
FROM Products
ORDER BY prod_price DESC;
~~~

如果要对多列指定排序方向，记得`DESC`只对前一个列名生效，也就是说，要让多列都降序排列，必须每一列都指定`DESC`：

~~~sql
SELECT prod_id, prod_price, prod_name
FROM Products
ORDER BY prod_price DESC, prod_name;
~~~
