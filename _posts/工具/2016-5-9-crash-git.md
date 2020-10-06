---
layout: single
title: "Git速成"
subtitle: "摘自《廖雪峰的Git教程》"
date: 2016-5-9
author: "Anyinlover"
category: 工具
tags:
  - git
  - github
---

## Git 简介

Git 是一种分布式版本控制系统，由开发出 Linux 的大神 Linus 撰写。SVN 是目前使用最多的集中式版本库控制系统。

## 安装 Git

在 Ubuntu 下可以用下面的命令安装：

`sudo apt-get install git`

在 Mac 下则使用 Homebrew：

`brew install git`

安装完成后设置一下姓名，邮箱和默认编辑器：

```shell
git config --global user.name "Miachel Gu"
git config --global user.email "anyinlover@gmail.com"
git config --global core.editor vim
```

`--global`是一个全局参数，也可以对每个 Git 仓库设置不同的配置。

## 创建版本库

### 创建

在某个目录下使用下面的命令，将其初始化成一个 Git 仓库：

`git init`

### 添加文件到暂存区

新添或者修改一个文件后，使用`git add`将其添加到暂存区：

`git add filename`

### 从暂存区添加到版本库

使用命令`git commit`将暂存区文件提交到仓库, `-m`后是提交的备注：

`git commit -m "add a file"`

`add`和`commit`分成了两步，可以多次`add`，一次`commit`。

## 版本管理

`git status`查看仓库当前的状态：

`git status`

`git diff`比较修改过的文件：

`git diff filename`

### 版本回退

`git log`可以用来查看版本记录，其中一大串数字显示的是 SHA1 计算出来的`commit id`。

用下面的命令可以回退到上一版本，`HEAD`表示当前版本，`HEAD^`表示上一版本。依次类推，`HEAD^^`表示上上版本, `HEAD~100`表示往上 100 个版本：

`git reset --hard HEAD^`

如果又反悔回退了，只要知道`commit_id`，仍然可以重新回退到新版本

`git reset --hard commit_id`

`commit_id`可以只写前面几位，唯一匹配即可。

如果关闭窗口后找不到新版本的`commit_id`了，那就需要使用下面的命令来查看命令历史：

`git reflog`

### 工作区和缓存区

Git 最重要的概念就是三分天下：工作区-缓存区-版本库。`add`操作是将工作区的文件加到缓存区中，`commit`操作是将缓存区文件加到版本库。

### 管理修改

Git 管理的是修改而非文件。下面的命令可以比较版本库中和工作区中文件的差异：

`git diff HEAD -- filename`

### 撤销修改

如果只是修改了工作区的文件，用下面的命令可以撤销到上一次`commit`或`add`提交后的状态：

`git checkout -- filename`

如果修改已经提交到暂存区了，那就需要使用下面的命令丢弃暂存区修改：

`git reset HEAD filename`

记得丢弃暂存区修改后还要丢弃工作区修改。

### 删除文件

使用`git rm`删除文件，后面还需要`git commit`提交修改：

`git rm filename`

假如误删等同于丢弃暂存区修改，想要还原需要两步走：

```shell
git reset HEAD filename
git checkout -- filename
```

## 远程仓库

使用 SSH 方式连接 github，需要配置 SSH Key。假如主目录下无隐藏文件夹.ssh，就创建一个`ssh-keygen -t rsa -C "anyinlover@gmail.com"`。在.ssh 文件下有 id_rsa 和 id_rsa.pub 两个文件，将 id_rsa.pub 内容复制到 GitHub SSH Keys，注意不要有多余的空格或换行符。

### 添加远程库

在 github 上 Create a new repo，命名为 learngit，在本地仓库已存在的情况下，将本地库与远程库相关联：

`git remote add origin git@github.com:anyinlover/learngit`

- 执行错误提醒： fatal: remote origin already exists.
  解决办法：`git remote rm origin`

远程库默认名字是 origin，不必修改。
推送本地库内容到远程库：

`git push -u origin master`

参数`-u`用以将本地 master 分支与远程 master 分支关联，以后推送只需要:

`git push origin master`

- ssh 警告
  第一次 push 或 clone 时出现，用以显示第一次验证 github 服务器上的 SSH key，输入 yes 即可。

### 克隆远程库

对于新项目而言，最好先在在 gihub 上创建新仓库，然后克隆到本地：

`git clone git@github.com:guguoshenqi/gitskill.git`

Git 其实还能支持其他协议，比如`https`，但`https`速度慢，且每次推送需要输入口令，只有在只开放 http 端口的公司内网环境下使用。

## 分支管理

分支提供了平行世界的功能，使多人协作更加的方便。

### 创建与合并分支

在版本回退中，每次提交 Git 都会串成时间线也就是分支。在 Git 中，默认的那个分支称为主分支，即`master`分支。在前面的单分支情况下，`HEAD`严格来说是指向`master`的，而`master`才是指向提交的，所以`HEAD`指向的就是当前分支。

![git1](\img\git_1.png)

当新建分支`dev`时，指向与`master`相同的提交，并把`HEAD`指向`dev`，表示当前分支在`dev`上。

![git2](\img\git_2.png)

现在对工作区的修改和提交发生在了`dev`分支上，提交一次后会变成下面这样：

![git3](\img\git_3.png)

如果`dev`分支的工作完成了，需要将其合并到`master`上，只需要让`master`指向`dev`对应的提交，合并就完成了：

![git4](\img\git_4.png)

这个时候删除`dev`分支也没有关系（就是删除一个指针），删完后就变成：

![git5](\img\git_5.png)

首先创建并切换到新分支`dev`上：

`git checkout -b dev`

上面的命令相当于下面两步：

```shell
git branch dev
git checkout dev
```

`git branch`命令可以查看所有分支，带星号的是当前分支。

如果想切换分支，使用下面的命令：

`git chekcout master`

如果想把`dev`分支的内容合并到`master`分支上,`Fast-forward`是一种快速合并的模式，也就是前面提到的移动指针。有时候可能会不适用这种快速合并的方式。

`git merge dev`

合并完成后就可以放心的删除`dev`分支了：

`git branch -d dev`

### 解决冲突

当两个分支都有新的提交时，会发生合并冲突，需要手工修改后重新提交。
可以用`git log --graph`查看分支合并图:

`git log --graph --pretty=oneline --abbrev-commit`

### 分支管理策略

Git 默认会使用`Fast forward`模式，这种模式在删除分支后，会丢掉分支信息。如果禁用 Fast forward 模式合并，会产生一个新的 commit，从而可以保存被删除分支信息，使用`--no-ff`参数：

`git merge --no-ff -m "merge with no-ff" dev`

实际开发的分支原则

1. master 分支非常稳定，只用来发布新版本
2. 在 dev 分支上干活，到版本发布时才合并到 master
3. 多人协作时每个人在自己分支上干活，时不时合并到 dev 分支。

类似于下图的团队合作：
![摘自廖雪峰git教程](\img\git_6.png)

### Bug 分支

如果要修复一个紧急 Bug，而现有分支上的工作区又不干净，可以先用下面的命令保存工作现场：

`git stash`

在完成创建 Bug 分支并合并到现有分支时，可以再恢复工作现场。
用下面的命令可以查看保存 stash 列表：

`git stash list`

有两种方法可以恢复工作现场
`git stash apply`
后不会删除 stash 列表，需要再用`git stash drop`来删除。

`git stach pop`
在恢复的同时也会删除 stash 列表。

在多次 stash 的情况下，可以先查看 stash 列表，然后再恢复指定的 stash：

`git stash apply stash@{0}`

### Feature 分支

开发一个新 feature，最好新建分支。

如果要强行删除未合并分支：

`git branch -D feature-vulcan`

### 多人协作

当从远程仓库克隆时，Git 自动会把本地`master`分支和远程`master`分支对应，远程仓库默认名称是`origin`。

要查看远程库的信息，使用`git remote`，`git remote -v` 显示更详细的信息。
如果要把本地`master`分支推送到远程：

`git push origin master`

如果要推送其他分支，如`dev`，那就用：

`git push origin dev`

注意不是全部分支需要远程推送：

- master 主分支，dev 开发分支需要推送
- bug 分支一般用于本地修复，不需推送
- feature 分支是否推送取决于是否合作开发

从远程库 clone 时，默认只有本地`master`分支，如果要创建远程 origin/dev 分支到本地：

`git checkout -b dev origin/dev`

完成开发后就可以`push`到远程库。但如果想用`git pull`抓取远程库，还要指定本地 dev 分支与远程 origin/dev 分支的连接：

`git branch --set-upstream dev origin/dev`

多人协作的工作模式：

1. 首先，尝试用`git push origin branch-name` 推送自己的修改；
2. 如果推送失败，是由于远程分支比你的本地更新， 需要先`git pull`试图合并。
3. 如果合并有冲突，则解决冲突，并在本地提交
4. 没有冲突后再次推送。

- 如果`git pull`提示”no tracking information"，说明本地分支和远程分支无关联，参考上面的指令。

## 标签管理

在发布版本时，打一个标签，可以唯一确定打标签时候的版本。标签是版本库的一个快照，实质是一个不能移动的指向 commit 的指针。

### 创建标签

首先切换到要打标签的分支上，创建新标签，默认打在最新提交的 commit 上：

`git tag v1.0`

`git tag`查看所有标签

如果是要为历史提交打标签，需要找到对应的`commit_id`：

`git tag v0.9 commit_id`

如果要查看标签，用`git tag`找到标签名然后使用下面的命令：

`git show <tagname>`

创建标签时还能带上说明文字：

`git tag -a v0.1 -m "version 0.1 released" commit_id`

用`-s`则能用私钥签名标签，签名采用 PGP 签名。需要安装 gpg。

`git tag -s v0.2 -m "signed version 0.2 released" commit_id`

### 操作标签

标签打错是可以删除的：
`git tag -d v0.1`

如果要把标签推送到远程：

`git push origin v1.0`

还可以推送全部标签到远程：

`git push origin --tags`

如果要删除远程标签，有些麻烦，需要先在本地删除，然后：

`git push origin :refs/tags/v0.9`

## 使用 github

- 在 Github 上， 可以任意 Fork 开源仓库
- 自己拥有 Fork 后的仓库的读写权限
- 可以推送 pull request 给官方仓库来贡献代码

## 自定义 Git

Git 除了最前面的姓名和邮箱，还有很多可以配置的。

### 忽略特殊文件

在 git 工作区根目录下建一个.gitignore 文件，可以指定不想提交的文件。
[Github 官网](https://github.com/github/gitignore)有不错的示例。

忽略文件原则：

- 忽略自动生成文件、如缩略图;
- 忽略编译生成的中间文件、可执行文件等。
- 忽略自己的带敏感信息的配置文件

### 配置别名

`git config --global alias.st status`
还有好些有意思的别名参考[原文](http://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000/001375234012342f90be1fc4d81446c967bbdc19e7c03d3000)。

配置文件存放位置：

- 每个仓库的配置在`.git/config`
- 当前用户(Global)配置在`~/.gitconfig`

### 搭建 git 服务器

[参考原文](http://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000/00137583770360579bc4b458f044ce7afed3df579123eca000)
