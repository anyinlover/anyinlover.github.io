---
layout: single
title: "获取数据"
subtitle: "《Sams Teach Yourself SQL in 10 Minutes》第二讲"
date: 2016-5-18
author: "Anyinlover"
category: 笔记
tags:
  - 编程语言
  - SQL
---

本章中主要学习`SELECT`的用法来从表中获取数据。

## `SELECT`语句

SQL语句由英语术语构成，这些术语是SQL语言的保留字，被称为关键字，不能拿来用作表名或列名。

最简单的`SELECT`语句由两部分构成，取什么，从哪取。

## 从单列读取

~~~sql
SELECT prod_name
FROM Products;
~~~

一个简单的`SELECT`语句仅仅是取回了所有的行，数据既不筛选，也不排序。

## 从多列读取

~~~sql
SELECT prod_id, prod_name, prod_price 
FROM Products;
~~~

## 读取所有列

~~~sql
SELECT * 
FROM Products;
~~~

## 读取不同值的行

~~~sql
SELECT distinct vend_id 
FROM Products;
~~~

## 限制返回的数量

只想返回头五行：

~~~sql
SELECT prod_name 
FROM Products 
LIMIT 5;
~~~

要读取下五行，还可以设置偏移：

~~~sql
SELECT prod_name 
FROM Products 
LIMIT 5 OFFSET 5;
~~~

MySQL还支持偏移的简写，前面的是限制数，后面的是偏移量：

~~~sql
SELECT prod_name 
FROM Products 
LIMIT 5,5;
~~~

## 使用评论

下面是行内注释的用法：

~~~sql
SELECT prod_name  -- this is a comment
FROM Products;
~~~

下面是多行注释的用法：

~~~sql
/* SELECT prod_name, vend_id
FROM Products;*/
SELECT prod_name
FROM Products;
~~~
