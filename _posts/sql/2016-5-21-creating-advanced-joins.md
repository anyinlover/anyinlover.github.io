---
layout: single
title: "创建高级连接"
subtitle: "《Sams Teach Yourself SQL in 10 Minutes》第十三讲"
date: 2016-5-21
author: "Anyinlover"
category: 笔记
tags:
  - 编程语言
  - SQL
---

本讲主要学习其他连接类型，以及如何使用表别名和在连接表中使用统计函数。

## 使用表别名

~~~sql
SELECT cust_name, cust_contact
FROM Customers AS C, Orders AS O, OrderItems AS OI
WHERE C.cust_id=O.cust_id
	AND OI.order_num=O.order_num
    AND prod_id='RGAN01';
~~~

## 使用不同类型的连接
上一讲的连接被称为内连接或者等连接。下面介绍另外三种连接类型：自连接，自然连接和外连接。

### 自连接
下面两个SQL实现相同的功能：

~~~sql
SELECT cust_id, cust_name, cust_contact
FROM Customers
WHERE cust_name = (SELECT cust_name
						FROM Customers
						WHERE cust_contact = 'Jim Jones');

SELECT c1.cust_id, c1.cust_name, c1.cust_contact
FROM Customers AS c1, Customers AS c2
WHERE c1.cust_name=c2.cust_name
	AND c2.cust_contact='Jim Jones';
~~~

第二个SQL使用了自连接，看似两张表，其实只用了一张表的数据。

很多数据库自连接跑的比子查询更高效，因此自连接值得一试。
   
### 自然连接

内连接会返回所有的列，即使数据是重复的。自然连接会把重复的列去除。而实现的方法就是手工指定~

~~~sql
SELECT C.*, O.order_num, O.order_date, OI.prod_id, OI.quantity, OI.item_price
FROM Customers AS C, Orders AS O, OrderItems AS OI
WHERE C.cust_id=O.cust_id
	AND OI.order_num=O.order_num
    AND prod_id='RGAN01';
~~~
    
### 外连接
连接中大部分行都会在另一张表中有对应的行。但有时也会发生没有对应行的情况。这时候就需要使用外连接。

~~~sql
SELECT Customers.cust_id, Orders.order_num
FROM Customers LEFT OUTER JOIN Orders
	ON Customers.cust_id=Orders.cust_id;

SELECT Customers.cust_id, Orders.order_num
FROM Customers RIGHT OUTER JOIN Orders
	ON Customers.cust_id=Orders.cust_id;
~~~

外连接有两种基本类型，左外连接和右外连接。
分别指定哪张表选择所有的行。

### 连接的统计函数

~~~sql
SELECT Customers.cust_id, COUNT(Orders.order_num) AS num_ord
FROM Customers INNER JOIN Orders
	ON Customers.cust_id=Orders.cust_id
GROUP BY Customers.cust_id;
~~~

## 使用连接和连接条件

总结一下使用连接的几个关键点：

* 注意连接的类型，大部分情况下使用内连接。有时也会用到外连接。
* 不同数据库管理系统的连接语法略有不同。
* 要使用正确的连接条件。
* 一定要提供连接条件。
* 可以多表连接，甚至采取不同连接方式。但建议做分布测试，方便定位。