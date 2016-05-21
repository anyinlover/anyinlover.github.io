---
layout: post
title: "组合查询"
subtitle: "《Sams Teach Yourself SQL in 10 Minutes》第十四讲"
date: 2016-5-21
author: "Anyinlover"
catalog: true
tags:
  - 编程语言
  - SQL
---

本讲主要学习使用`UNION`操作符来联合多`SELECT`语句。

## 理解组合查询

有两种场景需要使用组合查询。


* 从不同表中返回相似结构的数据
* 从单表中执行多次查询，这种情况和使用多`WHERE`分句的效果是一致的。

## 创建组合查询

SQL查询通过`UNION`操作符组合在一起。

### 使用`UNION`
`UNION`简单的把两个`SELECT`区分开来：

~~~sql
SELECT cust_name, cust_contact, cust_email
FROM Customers
WHERE cust_state IN ('IL','IN','MI')
UNION
SELECT cust_name,cust_contact,cust_email
FROM Customers
WHERE cust_name='FUN4All';
~~~

### `UNION`规则

`UNION`有一些规则需要遵循：

* 一个`UNION`必须有两或多个`SELECT`构成。
* 每个查询都必须包含相同的列，表达式或统计函数。
* 列数据类型必须可兼容。

### 包含或省略重复行

`UNION`默认会省略重复行，类似于多`WHERE`分句。如果要包含重复行，使用`UNION ALL`:

~~~sql
SELECT cust_name, cust_contact, cust_email
FROM Customers
WHERE cust_state IN ('IL','IN','MI')
UNION ALL
SELECT cust_name,cust_contact,cust_email
FROM Customers
WHERE cust_name='FUN4All';
~~~

### 组合查询结果排序

组合查询的排序必须放在最后一个`SELECT`之后：

~~~sql
SELECT cust_name, cust_contact, cust_email
FROM Customers
WHERE cust_state IN ('IL','IN','MI')
UNION
SELECT cust_name,cust_contact,cust_email
FROM Customers
WHERE cust_name='FUN4All'
ORDER BY cust_name, cust_contact;
~~~

组合查询在组合不同表的数据时非常有用，即使列名不同，也可以通过别名来保持一致。