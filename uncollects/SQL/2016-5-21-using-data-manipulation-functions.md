---
layout: single
title: "使用数据处理函数"
subtitle: "《Sams Teach Yourself SQL in 10 Minutes》第八讲"
date: 2016-5-21
author: "Anyinlover"
category: 数据库
tags:
  - 编程语言
  - SQL
---

本讲中主要学习什么是函数，数据库支持哪些类型的函数，以及如何使用这些函数。

## 理解函数

和其他语言一样，SQL 也支持使用函数来处理数据。比如前面提到的`RTRIM`

但 SQL 函数有一个问题，就是和数据库管理系统强相关。不同的数据库支持的函数差别较大。因此，使用函数来处理数据时，一个好的习惯是加上注释。

## 使用函数

大部分 SQL 实现了下面四种类型的函数：

- 文本函数，处理文本字符串。
- 数字函数，数学运算。
- 时间日期函数。
- 系统函数。

函数除了使用在`SELECT`部分，也可以使用在 SQL 语句的其他部分，比如`WHERE`。

### 文本处理函数

```sql
SELECT vend_name, UPPER(vend_name) AS vend_name_upcase
FROM Vendors
ORDER BY vend_name;
```

下面是一些通用的文本处理函数：

| 函数        | 作用             |
| :---------- | :--------------- |
| `LEFT()`    | 从左取子字符串   |
| `LENGTH()`  | 字符串长度       |
| `LOWER()`   | 字符串小写       |
| `LTRIM()`   | 截除字符串首空格 |
| `RIGHT()`   | 从右取字符串     |
| `RTRIM()`   | 截除字符串尾空格 |
| `SOUNDEX()` | 返回字符串音值   |
| `UPPER()`   | 字符串大写       |

这里面`SOUNDEX()`比较特殊，用来查找发音相似的字符串：

```sql
SELECT cust_name, cust_contact
FROM Customers
WHERE SOUNDEX(cust_contact)=SOUNDEX('Michael Green');
```

## 日期时间处理函数

日期和时间在数据库中以数据库管理系统特有的格式存储，实际使用时需要通过日期时间处理函数来调用，这一部分函数非常重要，但同时也非常不具有普遍性。

```sql
SELECT order_num
FROM Orders
WHERE YEAR(order_date)=2012;
```

具体的日期时间处理函数需要参照不同 SQL 的产品文档。

## 数字函数

数字函数和上面两者相比最不常用，却是最普遍性的。下面列举一些常见的数字处理函数：

- `ABS()`
- `COS()`
- `EXP()`
- `PI()`
- `SIN()`
- `SQRT()`
- `TAN()`
