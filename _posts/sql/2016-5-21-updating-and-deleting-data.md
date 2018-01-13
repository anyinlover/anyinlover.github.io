---
layout: single
title: "更新和删除数据"
subtitle: "《Sams Teach Yourself SQL in 10 Minutes》第十六讲"
date: 2016-5-21
author: "Anyinlover"
category: 笔记
tags:
  - 编程语言
  - SQL
---

本讲主要学习`UPDATE`和`DELETE`语句来操作数据。

## 更新数据

`UPDATE`主要有两种用法：

* 更新特定行
* 更新所有行

~~~sql
UPDATE Customers
SET cust_email='kim@thetoystore.com'
WHERE cust_id='1000000005';

UPDATE Customers
SET cust_contact='Sam Roberts',
	cust_email='sam@toyland.com'
WHERE cust_id='1000000006';

UPDATE Customers
SET cust_email=NULL
WHERE cust_id='1000000005';
~~~

## 删除数据

~~~sql
DELETE FROM Customers
WHERE cust_id='1000000008';
~~~

注意如果要删除的行中存在其他表的外键，那么删除操作是无法完成的。

## 更新删除数据指南

下面是一些实用的更新删除数据指南：

* 永远不要在没有`WHERE`分句的情况下更新和删除，除非你真的想操作所有行。
* 确保每张表都有一个主键，尽可能在`WHERE`分句中使用它。
* 在用`WHERE`分句更新删除前，先用`SELECT`测试查看。
* 使用数据库的强制参考完整性特性，即上面提到的外键。
* 考虑DBA强制在更新删除时使用`WHERE`。
