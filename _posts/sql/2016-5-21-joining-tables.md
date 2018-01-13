---
layout: single
title: "连接表"
subtitle: "《Sams Teach Yourself SQL in 10 Minutes》第十二讲"
date: 2016-5-21
author: "Anyinlover"
category: 笔记
tags:
  - 编程语言
  - SQL
---

本讲主要学习连接表的作用和创建。

## 理解连接

连接表是SQL语句的一个重要特性，也是`SELECT`中主要的操作。理解连接和连接语法是学习SQL中非常重要的一部分。

在理解连接之前，有必要先了解关系表和关系型数据库的设计。

### 理解关系表
关系表被设计成用来将信息分割成多张表，每张条对应一种数据类型。

### 为何使用连接
将数据分散在不同表里可以更高效的存储，更简单的操作，以及更好的扩展。

但当要获取的数据分散在不同表里时，就需要使用连接使得`SELECT`操作可以从多表中获取数据。

## 创建连接

~~~sql
SELECT vend_name, prod_name, prod_price
FROM Vendors, Products
WHERE Vendors.vend_id=Products.vend_id;
~~~

### `WHERE`分句的重要性

`WHERE`分句指定了两张表的关系。

### 内连接

上述的这种连接也被称为内连接，连接是建立在两表的相同性上的。内连接还可以用一种`ON`分句来替代`WHERE`分句，作用是一致的：

~~~sql
SELECT vend_name, prod_name, prod_price
FROM Vendors INNER JOIN Products
	ON Vendors.vend_id=Products.vend_id;
~~~

### 连接多表

连接表的数量是不受限制的，只需要指定关系即可：

~~~sql
SELECT prod_name, vend_name, prod_price, quantity
FROM OrderItems, Products, Vendors
WHERE Products.vend_id=Vendors.vend_id
	AND OrderItems.prod_id=Products.prod_id
    AND order_num=20007;
~~~

回顾上一讲的子查询，我们可以发现用内连接也可以实现它的功能，下面两个SQL语句的功能就是一致的，具体哪个更高效视情况而定，需要测试。

~~~sql
SELECT cust_name, cust_contact
FROM Customers
WHERE cust_id IN (SELECT cust_id
				  FROM Orders
                  WHERE order_num IN (SELECT order_num
									  FROM OrderItems
                                      WHERE prod_id='RGAN01')); 
                                         
SELECT cust_name, cust_contact
FROM Customers, Orders, OrderItems
WHERE Customers.cust_id=Orders.cust_id
	AND OrderItems.order_num=Orders.order_num
    AND prod_id='RGAN01';
~~~