---
layout: single
title: "用python爬链家房价"
subtitle: "《Web Scraping With Python》实战演练"
date: 2016-5-8
author: "Anyinlover"
category: 实践
tags:
  - 编程语言
  - 爬虫
  - Python
---

之前曾经突发奇想的想做一个房价预测，于是花了点时间去学习python爬虫。结果数据爬下来之后发现房价预测是个时间序列相关的问题，单靠现在一点爬下来的数据显然是行不通的，最后房价预测的问题不了了之了，倒是掌握了简单的爬虫方法。

说是爬虫，其实只是学了BeautifulSoup这个库，就能够爬下来网页了。总结起来爬网页最重要的就是分析其结构，然后层层深入到最内层再做一个数据的提取。

详细的教程懒得总结了，比较简单，我就是看了《Web Scraping With Python》一二两章，同时参考BeautifulSoup的官方文档，基本就完成了链家上海的二手房数据爬取。下面贴一下我的代码，也没有做多线程，最后大概两个小时跑出来的。

```python
import csv
from urllib.request import urlopen
from bs4 import BeautifulSoup

file = "shlj.csv"
url = "http://sh.lianjia.com/ershoufang/d"
with open(file,'w',newline='') as csvfile:
    writer = csv.writer(csvfile)

    def house_scrap(house, writer):
        title = house.find("div",{"class":"info-panel"}).a.string
        place = house.find("div",{"class":"where"}).findAll("span")
        neighbourhood = place[0].string
        layout = place[1].string.strip()
        square = place[3].string.strip()

        info = [title,neighbourhood,layout,square]

        zones = house.find("div",{"class":"con"}).findAll("a")

        for zone in zones:
            info.append(zone.string)

        details = house.find("div",{"class":"con"}).findAll("span")

        for detail in details:
            info.append(detail.next_sibling.string.strip())

        price = house.find("div",{"class":"price"}).span.string
        per_price = house.find("div",{"class":"price-pre"}).string

        info.append(price)
        info.append(per_price)

        writer.writerow(info)

    def page_scrap(url,writer):

        html = urlopen(url)
        bsObj = BeautifulSoup(html)
        infos = bsObj.find("div",{"class":"list-wrap"})
        for house in infos.findAll("li"):
            house_scrap(house,writer)

    for i in range(4000):
        page_scrap(url+str(i+1),writer)
```
