---
layout: single
title: "统计函数"
subtitle: "《Sams Teach Yourself SQL in 10 Minutes》第九讲"
date: 2016-5-21
author: "Anyinlover"
category: 笔记
tags:
  - 编程语言
  - SQL
---

本讲主要学习SQL统计函数来总结整表数据。

## 使用统计函数

SQL提供了五种统计函数，用于对数据进行一个总结性的了解。

| 函数 | 作用 |
|:---|:--|
|`AVG()`|平均值|
|`COUNT()`|行数|
|`MAX()`|最高值|
|`MIN()`|最低值|
|`SUM()`|和值|

### `AVG()`函数

~~~sql
SELECT AVG(prod_price) AS avg_price
FROM Products;

SELECT AVG(prod_price) AS avg_price
FROM Products
WHERE vend_id = 'DLL01';
~~~

`NULL`值会被`AVG()`忽略不计。

### `COUNT()`函数

使用`COUNT(*)`会计算所有的行数，包括`NULL`值的。

~~~sql
SELECT COUNT(*) AS num_cust
FROM Customers;
~~~

使用`COUNT(column)`则会把`NULL`值排除在外。

~~~sql
SELECT COUNT(cust_email) AS num_cust
FROM Customers;
~~~

### `MAX()`函数

~~~sql
SELECT MAX(prod_price) AS max_price
FROM Products;
~~~

除了应用在数字数据上，`MAX()`也可以应用在文本数据上。

### `MIN()`函数

~~~sql
SELECT MIN(prod_price) AS min_price
FROM Products;
~~~

和`MAX()`类似，`MIN()`也能应用在文本数据上。

### `SUM()`函数

~~~sql
SELECT SUM(quantity) AS items_ordered
FROM OrderItems
WHERE order_num=20005;

SELECT SUM(item_price*quantity) AS total_price
FROM OrderItems
WHERE order_num=20005;
~~~

## 独特值统计

如果要计算独特值得统计数据，可以使用`DISTINCT`关键字，但注意只能配合列名使用：

~~~sql
SELECT AVG(DISTINCT prod_price) AS avg_price
FROM Products
WHERE vend_id='DLL01';
~~~

## 联合统计函数

~~~sql
SELECT COUNT(*) AS num_items,
		MIN(prod_price) AS price_min,
      	MAX(prod_price) AS price_max,
        AVG(prod_price) AS price_avg
FROM Products;
~~~