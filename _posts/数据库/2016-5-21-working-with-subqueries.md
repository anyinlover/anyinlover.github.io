---
layout: single
title: "使用子查询"
subtitle: "《Sams Teach Yourself SQL in 10 Minutes》第十一讲"
date: 2016-5-21
author: "Anyinlover"
category: 数据库
tags:
  - 编程语言
  - SQL
---

本讲主要学习子查询及其使用。

## 理解子查询

前面的`SELECT`语句都是简单的查询，SQL 还允许在查询中嵌套查询，这就是子查询。

## 使用子查询过滤

```sql
SELECT cust_id
FROM Orders
WHERE order_num IN (SELECT order_num
     FROM OrderItems
     WHERE prod_id = 'RGAN01');

SELECT cust_name, cust_contact
FROM Customers
WHERE cust_id IN (SELECT cust_id
      FROM Orders
                  WHERE order_num IN (SELECT order_num
           FROM OrderItems
                                      WHERE prod_id='RGAN01'));
```

注意子查询虽然能达成目的，但经常不是最有效的方式。后续会讲到连接表也能达到同样的效果。

## 使用子查询作为计算字段

```sql
SELECT cust_name, cust_state,
 (SELECT COUNT(*)
     FROM Orders
     WHERE Orders.cust_id=Customers.cust_id) AS orders
FROM Customers
ORDER BY cust_name;
```
