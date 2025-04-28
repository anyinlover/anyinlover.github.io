# Windows FAQ

## win10 文件资源管理器的快速访问取消自动添加

查看-选项-隐私 去掉两个勾

* 在“快速访问”中显示最近使用的文件
* 在“快速访问”中显示常用文件夹

## 资源管理器中双击文件夹在新窗口打开

在开始菜单-运行中输入

regsvr32 "%SystemRoot%\\System32\\actxprxy.dll"
regsvr32 "%ProgramFiles%\\Internet Explorer\\ieproxy.dll"
