---
title: 'Vscode快捷键'
description: 'Vscode快捷键总览'
pubDate: 'May 05 2024'
heroImage: '/vscodevim.jpg'
---

以下的快捷键都是搭配vim插件使用的，在编辑效率上有极大的提升。注意mac和windows下的按键会有略微区别。

## General

1. `Ctrl + Shift + P`: 打开命令界面
2. `Ctrl + P`: 快速打开某个文件，`@`可以快速跳转到某个符号
3. `Ctrl + Shift + N`: 打开新的窗口
4. `Ctrl + Shift + W`: 关闭窗口
5. `Ctrl + ,`: 打开设置，注意在编辑器界面会失效，最好在侧栏中使用
6. `Ctrl + K Ctrl + S`: 打开快捷键设置

## Basic editing

1. `dd`: 删除行
2. `yy`: 复制行
3. `p`: 在光标后粘贴
4. `P`: 在光标前粘贴
5. `o`: 在下一行插入
6. `O`: 在上一行插入
7. `%`: 跳转到对应的括号
8. `>>`: 缩进行
9. `<<`: 反缩进行
10. `0`: 到达行首
11. `$`: 到达行尾
12. `gg`: 到达文件头
13. `G`: 到达文件尾
14. `Ctrl + b`: 往上翻页
15. `Ctrl + f`: 往下翻页
16. `za`: 折叠、反折叠
17. `Ctrl + /`: 注释反注释行
18. `Shift + Alt + A`: 注释反注释块

## Navigation

1. `Ctrl + T`: 查看符号，当前光标在符号上会查看当前符号来源
2. `Num + G`: 跳转到某一行
3. `Ctrl + Shift + O`: 跳转到文件内某个符号
4. `Ctrl + Shift + M`: 打开问题面板
5. `F8`: 跳到下一个问题提示
6. `Shift + F8`: 跳到上一个问题提示
7. `Ctrl + O`: 跳转回到上一个位置
8. `Ctrl + I`: 撤销跳转回到上一个位置
9. `Ctrl + Shift + Tab`: 编辑文件列表跳转

## Search and replace

1. `/`: 向下搜索
2. `?`: 向上搜索
3. `:s`: 替换
4. `n`: 匹配下一个
5. `N`: 匹配上一个
6. `*`: 向下搜索当前光标内容
7. `#`: 向上搜索当前光标内容

## Multi-cursor and selection

1. `V`: 行选择模式
2. `I`: 在行首插入光标
3. `A`: 在行尾插入光标

## Rich languages editing

1. `Shift + Alt + F`: 格式化文档
2. `gd`: 跳转到定义
3. `Alt + F12`: 浮现定义
4. `Ctrl + .`: 快速修复
5. `Shift + F12`: 浮现引用
6. `F2`: 重命名符号

## Editor management

1. `:q`: 退出编辑
2. `Ctrl + W V`: 左右分窗口
3. `Ctrl + W S`: 上下分窗口
4. `Ctrl + W H/J/K/L`: 移动
5. `Ctrl + W W`: 遍历

## File management

1. `:enew`: 新建空白文件
2. `:e`: 编辑或新建文件
3. `:w`: 保存
4. `:wa`: 保存所有
5. `:qa`: 退出所有
6. `:x`: 保存并退出

## Display

1. `F11`: 开关全屏
2. `Ctrl + =`: 放大编辑器
3. `Ctrl + -`: 缩小编辑器
4. `Ctrl + B`: 开关侧栏
5. `Ctrl + K Z`: 禅模式

## Debug

1. `F9`: 开关断点
2. `F5`: 开始继续
3. `Shift + F5`: 停止
4. `F11`: 下钻调试
5. `Shift + F11`: 下钻调试
6. `F10`: 单步调试

## Integrated terminal

1. ``Ctrl + ` ``: 显示集成终端
2. ``Ctrl + Shift + ` ``: 新建终端
