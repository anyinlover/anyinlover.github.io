---
tags:
    - shadowsocks
---

## What is shadowsocks

Shadowsocks is a secure sock5 proxy that provides an encrypted tunnel to bypass the firewalls.

## How shadowsocks works

## Shadowsocks implementations

## Deploying

## References

- [shadowsocks homepage](https://shadowsocks.org/)

## How to set up


### Set up the server

### Shadowsocks-libev 安装与配置

请注意，随着时间流逝，以下教程必然会过时，因此仅作为个人服务器搭建的参考。

安装图省心推荐秋水逸冰的[一键安装脚本](https://teddysun.com/357.html)

**请注意，秋水逸冰已经宣布放弃继续维护该脚本，因此该脚本可能随时会失效**

1. wget --no-check-certificate -O shadowsocks-all.sh [https://raw.githubusercontent.com/teddysun/shadowsocks_install/master/shadowsocks-all.sh](https://raw.githubusercontent.com/teddysun/shadowsocks_install/master/shadowsocks-all.sh)
    chmod \+x shadowsocks-all.sh
    ./shadowsocks-all.sh 2\>&1 | tee shadowsocks-all.log
    使用root用户ssh登录
2. 选择
3. 跟随提示选择相对应的模式即可
    1. 加密方式推荐
    2. 端口可以尽量设置高一点,避免443, 1080等常用端口,
    3. **安装 simple-obfs , 选择 http 模式**
安装成功后会有如下提示:
4. Congratulations, your_shadowsocks_version install completed\!
5. 使用root用户登录
    ./shadowsocks-all.sh uninstall
    卸载
6. 启动：/etc/init.d/shadowsocks-libev start
    停止：/etc/init.d/shadowsocks-libev stop
    重启：/etc/init.d/shadowsocks-libev restart
    查看状态：/etc/init.d/shadowsocks-libev status
    常用命令

### Set up the client

#### OSX

[shadowsocksx-ng](https://github.com/shadowsocks/ShadowsocksX-NG/releases/)

In fact, Shadowsocks for OSX has a auto proxy mode with a GFWList PAC, in the most time you just open the auto proxy mode and surf the Internet without blocks. However, there are some websites are blocked and haven’t been updated in the GFWList. In such situation, we need the Chrome and its add-on [SwitchyOmega](https://github.com/FelisCatus/SwitchyOmega). The project has a detailed introduction and installation and configuration.

目前的版本中已经支持obfs混淆，并且已经直接集成于客户端之中，无需额外下载。

Windows

[shadowsocks-windows](https://github.com/shadowsocks/shadowsocks-windows/releases)

#### IOS

 shadowrocket.

国区已经下架，老版本暂时够用

#### Android

[shadowsocks-android](https://github.com/shadowsocks/shadowsocks-android)

obfs插件

[simple-obfs](https://github.com/shadowsocks/simple-obfs-android/releases)

[shadowsocks wiki](https://github.com/Shadowsocks-Wiki/shadowsocks)

OK, let’s enjoy the Internet rather than the Chinese Intranet\!
