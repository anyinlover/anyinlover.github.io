---
layout: post
title: "创建操作表"
subtitle: "《Sams Teach Yourself SQL in 10 Minutes》第十七讲"
date: 2016-5-21
author: "Anyinlover"
catalog: true
tags:
  - 编程语言
  - SQL
---

本讲主要学习表的创建，更新和删除的基本操作。

## 创建表

一般来说有两种方式可以创建表，DBMS自身的交互工具和SQL语句。

### 基本表创建

~~~sql
CREATE TABLE MyProducts
(
prod_id	CHAR(10)	NOT NULL,
vend_id	CHAR(10)	NOT NULL,
prod_name	CHAR(254)	NOT NULL,
prod_price	DECIMAL(8,2)	NOT NULL,
prod_desc	VARCHAR(1000)	NULL
);
~~~

### 使用`NULL`值

如果指定了`NOT NULL`，那么这一列的数据就不能为空。默认值是`NULL`，可以省略：

~~~sql
CREATE TABLE MyVendors
(
vend_id	CHAR(10)	NOT NULL,
vend_name	CHAR(50)	NOT NULL,
vend_address	CHAR(50),
vend_city	CHAR(50),
vend_state	CHAR(5),
vend_zip	CHAR(10),
vend_country	CHAR(50)
);
~~~

### 指定默认值

~~~sql
CREATE TABLE MyOrderItems
(
order_num	INTEGER	NOT NULL,
order_item	INTEGER	NOT NULL,
prod_id	CHAR(10)	NOT NULL,
quantity	INTEGER	NOT NULL	DEFAULT 1,
item_price	DECIMAL(8,2)	NOT NULL
);
~~~

## 更新表
~~~sql
ALTER TABLE Vendors
ADD vend_phone CHAR(20);

ALTER TABLE Vendors
DROP COLUMN vend_phone;
~~~

更新表同样是一个危险的操作，完整的步骤如下：

1. 创建一张新表结构的表
2. 使用`INSERT SELECT`复制数据
3. 验证新表可用
4. 重命名旧表
5. 重命名新表
6. 新建触发器，存储过程，索引和外键

## 删除表

~~~sql
DROP TABLE MyVendors;
~~~
## 重命名表

~~~sql
RENAME TABLE MyOrderItems to YouOrderItems;
~~~