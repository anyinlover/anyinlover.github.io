---
layout: single
title: "技术博客搭建"
subtitle: "感谢gitpage，jekyll和hux模板"
date: 2016-4-21
author: "Anyinlover"
category: 工具
tags:
  - jekyll
  - github
---

搭建一个属于自己的博客是我一直都有的梦想。作为一个有技术的人，当然应该选择 gitpage。使用 git 来管理，用 markdown 来撰写，不担心内容丢失，也不用为繁杂的语法困扰，十分合我心意。下面就记录一下我的博客搭建过程，操作系统是 OS X。

## 申请 gitpage 账号，建立仓库

想要用 gitpage，首先得有个 github 账号。

### github 建立博客仓库

在 github 上建立一个仓库，仓库命名是有规范的。比如我的 github 用户名是 anyinlover，那么仓库必须命名为 anyinlover.github.io。

### 远程仓库同步到本地

找个合适的文件夹防止本地目录，比如 Documents 下：

    git clone anyinlover.github.io

假如还没有安装 git，需要先装上 git。

## 安装 jekyll 并绑定文件夹

### 安装 jekyll

    gem install jekyll

### 绑定文件夹

    cd anyinlover.github.io
    jekyll build

## 安装配置模板

网上有挺多 jekyll 模板，除了[官方模板网站](http://jekyllthemes.org)，我更喜欢[dr. jekyll themes](https://drjekyllthemes.github.io)。如果要足够简单，[poole](http://getpoole.com)是个不错的选择，本次我使用的是一位中国人做的模板[hux](https://github.com/Huxpro/huxpro.github.io)，功能强大，在各方面都很趁意。

### 安装模板

从模板网站上把文件夹下载下来，解压到本地的博客文件夹。可以在本地打开 jekyll 先看一下，在浏览器输入 0.0.0.0:4000：

    jekyll serve

### 配置模板

配置模板是件比较烦的事情。特别是 hux 这个模板，虽然好看，但也意味着可配置的东西太多。配置文件主要是\_config.xml 这个文件，其他地方就需要修改 html 代码了。

#### 配置\_config.xml

前面的 title 啥的不必提了，看几个特别的地方。

SNS 设置那里我添加了豆瓣和简书的支持。增加 douban_username 和 jianshu_username 两行。

anchorjs 那里我把默认的 true 改成了 false，因为正文中标题前莫名出现一个#看起来很难受，锚定的功能也用不上。

评论系统我选择了 disqus，注释了默认的 duoshuo。

kramdown 那里作者配置了输入 GFM，我也把它去掉了，我需要用 latex 语法输入数学公式，这是 GFM 不支持的。

#### 配置 about.html

作者把 about.html 直接写成了 html，我觉得更合适的还是用 md 文件来表示。修改其中的 description，删除秀恩爱照片。在正文中去掉了中英文版本（按钮略丑）。替换内容，只留下一段自己的介绍。

#### 配置 index.html 和 tags.html

作者把这两页的描述都写进了 html 文件里，打开修改成自己要说的 description,删除默认图片

#### 配置 page.html

在 page.html 中添加了豆瓣和简书的支持，在 143 行插入：

    {% if site.douban_username %}
    <li>
        <a target="_blank" href="https://www.douban.com/people/{{ site.douban_username }}">
           <span class="fa-stack fa-lg">
                <i class="fa fa-circle fa-stack-2x"></i>
                <i class="fa  fa-stack-1x fa-inverse">豆</i>
           </span>
         </a>
    </li>
    {% endif %}
    {% if site.jianshu_username %}
    <li>
        <a target="_blank" href="http://jianshu.com/users/{{ site.jianshu_username }}/timeline">
           <span class="fa-stack fa-lg">
                <i class="fa fa-circle fa-stack-2x"></i>
                <i class="fa fa-stack-1x fa-inverse">简</i>
           </span>
        </a>
    </li>
    {% endif %}

#### 配置 footer.html

在 footer.html 中也要添加 page.html 里添加的代码。在最后的 Theme by Hux，删去了 github 部分。

## 撰写博客并上传

### 撰写

用 macdown 写博客，放在\_posts 文件夹下。需要有文件头类似如下：

    ```yaml
    ---
    layout: single
    title: "技术博客搭建"
    subtitle: "感谢gitpage，jekyll和hux模板"
    date: 2016-4-21
    author: "Anyinlover"
    category: 工具
    tags:
      - 工具
    ---
    ```

### 上传

    git add *.md
    git commit -m "add somefile"
    git push origin master

## 结尾

好啦，博客搭建过程大概如此，现在让我们一块欣赏吧。
