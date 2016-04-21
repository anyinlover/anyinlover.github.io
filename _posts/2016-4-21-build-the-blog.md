---
layout: post
title: "技术博客搭建"
subtitle: "感谢gitpage，jekyll和hux模板"
date: 2016-4-21
author: "Anyinlover"
catalog: true
tags:
  - 工具
---

搭建一个属于自己的博客是我一直都有的梦想。作为一个有技术的人，当然应该选择gitpage。使用git来管理，用markdown来撰写，不担心内容丢失，也不用为繁杂的语法困扰，十分合我心意。下面就记录一下我的博客搭建过程，操作系统是OS X。

## 申请gitpage账号，建立仓库。
想要用gitpage，首先得有个github账号。

### github建立博客仓库
在github上建立一个仓库，仓库命名是有规范的。比如我的github用户名是anyinlover，那么仓库必须命名为anyinlover.github.io。

### 远程仓库同步到本地
找个合适的文件夹防止本地目录，比如Documents下：

	git clone anyinlover.github.io
	
假如还没有安装git，需要先装上git。

## 安装jekyll并绑定文件夹

### 安装jekyll

	gem install jekyll
	
### 绑定文件夹

	cd anyinlover.github.io
	jekyll build
	
## 安装配置模板

网上有挺多jekyll模板，除了[官方模板网站](http://jekyllthemes.org)，我更喜欢[dr. jekyll themes](https://drjekyllthemes.github.io)。如果要足够简单，[poole](http://getpoole.com)是个不错的选择，本次我使用的是一位中国人做的模板[hux](https://github.com/Huxpro/huxpro.github.io)，功能强大，在各方面都很趁意。

### 安装模板

从模板网站上把文件夹下载下来，解压到本地的博客文件夹。可以在本地打开jekyll先看一下，在浏览器输入0.0.0.0:4000：

	jekyll serve
	
### 配置模板

配置模板是件比较烦的事情。特别是hux这个模板，虽然好看，但也意味着可配置的东西太多。配置文件主要是_config.xml这个文件，其他地方就需要修改html代码了。

#### 配置_config.xml
前面的title啥的不必提了，看几个特别的地方。

SNS设置那里我添加了豆瓣和简书的支持。增加douban_username和jianshu_username两行。

anchorjs那里我把默认的true改成了false，因为正文中标题前莫名出现一个#看起来很难受，锚定的功能也用不上。

评论系统我选择了disqus，注释了默认的duoshuo。

kramdown那里作者配置了输入GFM，我也把它去掉了，我需要用latex语法输入数学公式，这是GFM不支持的。

#### 配置about.html
作者把about.html直接写成了html，我觉得更合适的还是用md文件来表示。修改其中的description，删除秀恩爱照片。在正文中去掉了中英文版本（按钮略丑）。替换内容，只留下一段自己的介绍。

#### 配置index.html和tags.html
作者把这两页的描述都写进了html文件里，打开修改成自己要说的description,删除默认图片

#### 配置page.html
在page.html中添加了豆瓣和简书的支持，在143行插入：

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
    
   I
#### 配置footer.html
在footer.html中也要添加page.html里添加的代码。在最后的Theme by Hux，删去了github部分。

## 撰写博客并上传

### 撰写
用macdown写博客，放在_posts文件夹下。需要有文件头类似如下：

~~~
---
layout: post
title: "技术博客搭建"
subtitle: "感谢gitpage，jekyll和hux模板"
date: 2016-4-21
author: "Anyinlover"
catalog: true
tags:
  - 工具
---
~~~

### 上传

	git add *.md
	git commit -m "add somefile"
	git push origin master
	
## 结尾
好啦，博客搭建过程大概如此，现在让我们一块欣赏吧。