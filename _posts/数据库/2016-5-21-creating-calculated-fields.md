---
layout: single
title: "创建计算字段"
subtitle: "《Sams Teach Yourself SQL in 10 Minutes》第七讲"
date: 2016-5-21
author: "Anyinlover"
category: 数据库
tags:
  - 编程语言
  - SQL
---

本讲主要学习什么是计算字段，如何创建，以及如何使用别名。

## 理解计算字段

有时候存储在数据库的字段不能很好的对应于应用中，这时候就需要对字段进行加工组合，形成计算字段。

## 串联字段

```sql
SELECT CONCAT(vend_name, ' (', vend_country, ')')
FROM Vendors
ORDER BY vend_name;
```

如果字段后面有尾缀空格，可以用`RTRIM`去除：

```sql
SELECT CONCAT(RTRIM(vend_name), ' (', RTRIM(vend_country), ')')
FROM Vendors
ORDER BY vend_name;
```

## 使用别名

```sql
SELECT CONCAT(vend_name, ' (', vend_country, ')')
   AS vend_title
FROM Vendors
ORDER BY vend_name;
```

别名也可以使用在别的地方，比如原始列名包含非法字符或者过于冗长或误读。

## 执行数学运算

```sql
SELECT prod_id, quantity, item_price
FROM OrderItems
WHERE order_num = 20008;

SELECT prod_id,
  quantity,
       item_price,
       quantity*item_price AS expanded_price
FROM OrderItems
WHERE order_num=20008;
```

`SELECT`还能用来测试函数和计算，这时候可以省略`FROM`。比如`SELECT 3 * 2`返回 6，`SELECT curdate()`返回当下的日期和时间。
