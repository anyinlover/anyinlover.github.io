---
layout: post
title: "插入数据"
subtitle: "《Sams Teach Yourself SQL in 10 Minutes》第十五讲"
date: 2016-5-21
author: "Anyinlover"
catalog: true
tags:
  - 编程语言
  - SQL
---
本讲主要学习使用`INSERT`语句插入数据。

## 理解数据插入

`SELECT`毫无疑问是最常用的SQL语句，但还有其他三个常用SQL语句，第一个就是`INSERT`。

`INSERT`用来在数据库表中插入行，有三种用途：

* 插入完整单行
* 插入部分单行
* 插入查询结果

### 插入完整单行

~~~sql
INSERT INTO Customers
VALUES('1000000006',
			'Toy Land',
            '123 Any Street',
            'New York',
            'NY',
            '11111',
            'USA',
            NULL,
            NULL);
~~~

上面这种插入方式严重依赖于列次序，是很不安全的操作，建议用下面的方式插入：

~~~sql            
INSERT INTO Customers(cust_id,
					cust_name,
					cust_address,
					cust_city,
					cust_state,
					cust_zip,
					cust_country,
					cust_contact,
					cust_email)
VALUES('1000000007',
			'Toy Land',
            '123 Any Street',
            'New York',
            'NY',
            '11111',
            'USA',
            NULL,
			NULL);
~~~

### 插入部分单行

如果数据库允许某一列的值为空，或有指定的默认值，可以在插入时省略该行：

~~~sql
INSERT INTO Customers(cust_id,
					cust_name,
                    cust_address,
                    cust_city,
                    cust_state,
                    cust_zip,
                    cust_country)
VALUES('1000000008',
			'Toy Land',
            '123 Any Street',
            'New York',
            'NY',
            '11111',
            'USA');
~~~

### 插入获取到的数据

`INSERT`还可以用来插入`SELECT`获取到的数据，被称为`INSERT SELECT`。

~~~sql
INSERT INTO Customers(cust_id,
					cust_name,
                    cust_address,
                    cust_city,
                    cust_state,
                    cust_zip,
                    cust_country)
			SELECT cust_id,
					cust_name,
                    cust_address,
                    cust_city,
                    cust_state,
                    cust_zip,
                    cust_country
			FROM CustNew;
~~~

`INSERT`通常只能插入单行数据，但`INSERT SELECT`是个例外，其可以一次性插入多行数据。

## 复制表

~~~sql
CREATE TABLE CustCopy AS
SELECT * FROM Customers;
~~~

复制表操作在测试新的SQL语句时很有用。	