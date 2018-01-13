---
title: "shadowsocksR"
category: 工具
tags:
  - shadowsocks
  - 翻墙
---

时代在发展，shadowsocks源作者已经被请去喝茶了，现在更流行的一个叫shadowsocksR加强版，感谢两位作者。这里也趁着换VPS的同时新装了一个shadowsocksR。

## 安装shadowsocksR

在VPS上安装了最新的Ubuntu 16.04。然后发现可恶的搬瓦工对这个系统没有一键安装shadowsocks的脚本，在网上找了一下，发现这个哥们做了[一键安装的脚本](https://github.com/teddysun/shadowsocks_install)很有用：

~~~ shell
wget --no-check-certificate https://raw.githubusercontent.com/teddysun/shadowsocks_install/master/shadowsocksR.sh
chmod +x shadowsocksR.sh
./shadowsocksR.sh 2>&1 | tee shadowsocksR.log
~~~

安装完后配置都可以和原来保持一致，这样客户端就不用修改了。

## shadowsocksR的客户端

客户端大家都是兼容的：

Mac上我用的是[ShadowsocksX-NG](https://github.com/shadowsocks/ShadowsocksX-NG)。
IOS上我用的是Wingy，之前国区被下架了。
