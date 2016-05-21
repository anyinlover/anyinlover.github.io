---
layout: post
title: "过滤数据"
subtitle: "《Sams Teach Yourself SQL in 10 Minutes》第四讲"
date: 2016-5-18
author: "Anyinlover"
catalog: true
tags:
  - 编程语言
  - SQL
---
这一讲主要学习`SELECT`语句中的`WHERE`条件来指定搜索条件。

## 使用`WHERE`条件

~~~sql
SELECT prod_name, prod_price
FROM Products
WHERE prod_price = 3.49;
~~~

虽然数据可以从应用层面过滤，但强烈建议从数据库层面过滤。

注意`WHERE`条件需要放在`ORDER BY`条件前面。

## `WHERE`条件操作符

### 检查单值

~~~sql
SELECT prod_name, prod_price
FROM Products
WHERE prod_price < 10;

SELECT prod_name, prod_price
FROM Products
WHERE prod_price <= 10;
~~~

### 检查不匹配

~~~sql
SELECT vend_id, prod_name
FROM Products
WHERE vend_id <> 'DLL01';

SELECT vend_id, prod_name
FROM Products
WHERE vend_id != 'DLL01';
~~~

### 检查值范围

~~~sql
SELECT prod_name, prod_price
FROM Products
WHERE prod_price BETWEEN 5 AND 10;
~~~

### 检查空值

~~~sql
SELECT cust_name
FROM Customers
WHERE cust_email IS NULL;
~~~