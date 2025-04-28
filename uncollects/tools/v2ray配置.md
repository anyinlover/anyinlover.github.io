---
tags: 
  - v2ray
  - 翻墙
---

# v2ray配置

v2ray 这一套访问速度过慢，当前已经被弃用。

## 安装

[官方指导](https://github.com/v2fly/fhs-install-v2ray/blob/master/README.zh-Hans-CN.md)

在国内VPS上安装时在线下载非常慢可以使用离线安装方式：

```shell
sudo bash install-release.sh -l v2ray-linux-64.zip
systemctl enable v2ray
systemctl start v2ray
```

## 服务端配置

[参考博文](https://toutyrater.github.io/basic/vmess.html)

```json
{
  "inbounds": [{
    "port": 10088,
    "protocol": "vmess",
    "settings": {
      "clients": [{ "id": "08b9f9e6-58ae-4132-babd-d8c61d107a59",
                    "alterId": 64
                 }]
    }
  }],
  "outbounds": [{
    "protocol": "freedom",
    "settings": {}
  }]
}
```

```shell
systemctl status firewalld
firewall-cmd --zone=public --add-port=10088/tcp --permanent
firewall-cmd --reload
firewall-cmd --list-ports
```

## 客户端配置

[参考博文](https://yuan.ga/v2ray-android-tutorial/)
