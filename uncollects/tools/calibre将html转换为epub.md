---
tags:
  - calibre
  - html
---

# calibre将html转换为epub

## 原理

经过多次的教训，我明白了一个道理，所有技术问题，首先看Document。官方的文档永远是第一手资料，最及时也最本真。只有那些Document缺失或者语焉不详的，才去求助第三方资料。对于calibre来说，一份Document足矣。

一般的电子书转换分四步走，第一步将输入格式的电子书转为html，第二步将html转换为xhtml，同时会识别文档结构，第三步会应用css格式，第四步会输出指定格式的电子书。开启debug之后可以看到每个阶段的处理中文件。

## xpath

Calibre利用xpath来解析html，其实xpath也可以用来做爬虫。在电子书应用中xpath的应用比较简单，主要就是三种方式：

* 通过标签筛选
* 通过属性筛选
* 通过内容筛选

可以参考[calibre xpath tutorial](https://manual.calibre-ebook.com/xpath.html#xpath-tutorial)， 非常简短，有HTML基础一学就会。

为什么我们要说xpath呢，因为识别目录结构时常常需要手工指定。下面就能用得上。

## 命令转换

相比于前台界面，后台命令更加灵活。要注意的是在Mac下如果是手工安装包安装的，命令藏的很深，是默认没有环境变量的。推荐用brew cask安装。

后台使用的电子书转换命令是`ebook-convert`:

~~~shell
ebook-convert index.html ~/Downloads/TLDP.epub --breadth-first --chapter='//h:div[@class = "PART" or @class = "CHAPTER" or @class = "SECT1" or @class = "APPENDIX"]' --level1-toc='//h:div[@class = "PART"]' --level2-toc='//h:div[@class = "CHAPTER" or @class = "APPENDIX"]' --level3-toc='//h:div[@class = "SECT1"]'
~~~

主要是通过xpath识别文档结构。
