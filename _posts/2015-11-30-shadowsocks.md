---
layout: single
title: "Use Shadowsocks to over the GFW"
data: 2015-11-30
category: 工具
tags:
  - 翻墙
  - shadowsocks
---

Live in China, learn how to over the GFW is the nesessary skill to google. As VPNs are expensive and risky to be blocked, and goagent is slow and not stable, shadowsocks is a great choice for explorers.

## What is shadowsocks
Shadowsocks is a secure sock5 proxy that provides an encrypted tunnel to bypass the firewalls.

## How to set up
If you want to use shadowsocks, you need to have a VPS as your shadowsocks server. Here I use the DigitalOcean 512M/20G droplet. It has a moderate price and a good speed, which cost \$5 per month and has 1000G Transfer.(If you are a student, github has a student package including \$100 for DigitalOcean).
Remember choose a good datacenter region. Most people recommend San Francisco for it's in the America West.

I select Ubuntu 14.04 as my server os. How to install just refer DigitalOcean's help.

### Set up the server
In the Ubuntu os, install server is easy:

    apt-get install python-pip
    pip install shadowsocks

Although it can be configurated by the command line, I prefer by the config file.

Create a config file `/etc/shadowsocks.json` like the following:

    {
        "server": "192.241.219.20",
        "server_port": 8388,
        "local_address": "127.0.0.1",
        "local_port": 1080,
        "password": "overthegfw"
        "timeout": 300,
        "method": "aes-256-cfb"
    }

Then you can run the server in the backgroud:

    ssserver -c /etc/shadowsocks.json -d start

If you want to stop it:

    ssserver -c /etc/shadowsocks.json -d stop

if you want to check the log:

    less /var/log/shadowsocks.log

### Set up the client

#### OSX
Personally I use a MBP, and [Shadowsocks for OSX](https://github.com/shadowsocks/shadowsocks-iOS/wiki/Shadowsocks-for-OSX-Help) as the client.

In fact, Shadowsocks for OSX has a auto proxy mode with a GFWList PAC, in the most time you just open the auto proxy mode and surf the Internet without blocks. However, there are some websites are blocked and haven't been updated in the GFWList. In such situation, we need the Chrome and its add-on [SwitchyOmega](https://github.com/FelisCatus/SwitchyOmega). The project has a detailed introduction and installation and configuration.

#### IOS

I have some good news. Shadowsocks for IOS can't apply to IOS 9. But some alternative apps have shown up. I choose the cheapest one: shadowrocket. It just work like a VPN. For me, that's enough.

OK, let's enjoy the Internet rather than the Chinese Intranet!
