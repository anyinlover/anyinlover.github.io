---
title: FreshRSS配置
tags: RSS
category: 工具
---

## 安装数据库

mysql root zjsxsyx
CREATE USER ‘freshrss@localhost' IDENTIFIED BY ‘freshrss';
CREATE DATABASE `freshrss`;
GRANT ALL privileges ON `freshrss`.* TO ‘freshrss'@localhost;
FLUSH PRIVILEGES;
QUIT;

## 配置Freshrss

用户名 anyinlover

## 设置自动同步

[官方文档](https://freshrss.github.io/FreshRSS/en/admins/08_FeedUpdates.html)

在ubuntu上配置：

```crontab -e```

编辑增加：

```10 * * * * ubuntu php -f /usr/share/FreshRSS/app/actualize_script.php > /tmp/FreshRSS.log 2>&1```
