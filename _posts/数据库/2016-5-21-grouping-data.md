---
layout: single
title: "分组数据"
subtitle: "《Sams Teach Yourself SQL in 10 Minutes》第十讲"
date: 2016-5-21
author: "Anyinlover"
category: 数据库
tags:
  - 编程语言
  - SQL
---

本讲主要学习数据分组，使用了`GROUP BY`和`HAVING`字段。

## 理解数据分组

前一讲提到了 SQL 的统计函数，但那都是对全局数据操作的。如果要统计某个子集的数据，那就需要用到数据分组了。

## 创建分组

使用`GROUP BY`关键字可以创建分组：

```sql
SELECT vend_id, COUNT(*) AS num_prods
FROM Products
GROUP BY vend_id;
```

对于创建分组有几条重要的规则：

- `GROUP BY`可以包含多列，嵌套分组。
- 嵌套分组时数据在最后一列统计。
- `GROUP BY`中的列可以是原始列或者有效表达式（不能是统计函数），不能是别名。
- 除了统计函数，`SELECT`中的每一列都必须在`GROUP BY`中。
- `NULL`会被单独分为一组返回。
- `GROUP BY`防止在`WHERE`后，`ORDER BY`前。

## 过滤分组

对于组还可以过滤，使用`HAVING`关键字。其与`WHERE`的区别是过滤行和过滤组的区别，此外，`WHERE`是在分组前的过滤，`HAVING`是在分组后的过滤。

```sql
SELECT cust_id, COUNT(*) AS orders
FROM Orders
GROUP BY cust_id
HAVING COUNT(*)>=2;
```

此外，`WHERE`是在分组前的过滤，`HAVING`是在分组后的过滤。有可能同时会使用两者：

```sql
SELECT vend_id, COUNT(*) AS num_prods
FROM Products
WHERE prod_price>=4
GROUP BY vend_id
HAVING COUNT(*)>=2;
```

## 分组和排序

注意分组和排序是两回事。下面的例子可以说明：

```sql
SELECT order_num, COUNT(*) AS items
FROM OrderItems
GROUP BY order_num
HAVING COUNT(*)>=3

SELECT order_num, COUNT(*) AS items
FROM OrderItems
GROUP BY order_num
HAVING COUNT(*)>=3
ORDER BY items, order_num;
```

## `SELECT`分句次序

是时候总结一下`SELECT`分句次序了。

| 分句       | 作用           | 必要性             |
| :--------- | :------------- | :----------------- |
| `SELECT`   | 返回列或表达式 | 必要               |
| `FROM`     | 指定表         | 从表中取数据时必要 |
| `WHERE`    | 行过滤         | 不必要             |
| `GROUP BY` | 分组           | 分组统计必要       |
| `HAVING`   | 组过滤         | 不必要             |
| `ORDER BY` | 结果过滤       | 不必要             |
