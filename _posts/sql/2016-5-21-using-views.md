---
layout: single
title: "使用视图"
subtitle: "《Sams Teach Yourself SQL in 10 Minutes》第十八讲"
date: 2016-5-21
author: "Anyinlover"
category: 笔记
tags:
  - 编程语言
  - SQL
---

本讲主要学习视图，原理及何时使用。可以看到视图能够简化操作。

## 理解视图

视图是虚拟的表，它没有实体的数据，只是包含动态查询的结果。

### 为何使用视图

视图有以下几个常见的用法：

* 重用SQL语句
* 简化复杂SQL操作
* 导出部分表
* 保护数据
* 更改数据的样式和表达

当视图创建之后，它可以执行和表一样的操作。

### 视图的规则和限制

* 视图的命名也必须是唯一的。
* 视图的数量没有限制。
* 创建视图需要有安全权限。
* 视图可以嵌套。
* 视图不可被索引，不能有触发器或默认值。

## 创建视图

使用`CREATE VIEW`语句创建视图。

### 使用视图来简化复杂连接

~~~sql
CREATE VIEW ProductCustomers AS
SELECT cust_name, cust_contact, prod_id
FROM Customers, Orders, OrderItems
WHERE Customers.cust_id=Orders.cust_id
AND OrderItems.order_num=Orders.order_num;

SELECT cust_name, cust_contact
FROM ProductCustomers
WHERE prod_id='RGAN01';
~~~

### 使用视图更改数据样式

~~~sql
Create VIEW VendorLocations AS
SELECT CONCAT(vend_name, ' (', vend_country, ')')
AS vend_title
FROM Vendors;

SELECT * FROM vendorlocations;
~~~

## 使用视图过滤不必要的数据

~~~sql
CREATE VIEW CustomerEMailList AS
SELECT cust_id, cust_name, cust_email
FROM Customers
WHERE cust_email IS NOT NULL;

SELECT * FROM CustomerEMailList;
~~~

## 使用视图表示计算字段

~~~sql
CREATE VIEW OrderItemsExpanded AS
SELECT order_num,
		prod_id,
        quantity,
        item_price,
        quantity*item_price AS expanded_price
FROM OrderItems;

SELECT * FROM OrderItemsExpanded
WHERE order_num=20008;
~~~