# windows远程唤醒

在windows上需要设置两个地方。

华硕BIOS上设置：

1. Onboard Devices Configuration - Intel LAN OPROM enabled
2. APM Configuration - Power on by PCI-E enabled

设备管理器上设置：

1. 针对网卡驱动设置运行此设备唤醒计算机
2. 只允许幻包唤醒

在mac上安装wakeonlan，运行：

watch -n 800 wakeonlan f0:2f:74:df:aa:62
