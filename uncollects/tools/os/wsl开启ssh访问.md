# wsl开启ssh访问

## wsl启动ssh

首先需要在wsl中打开配置`/etc/ssh/sshd_config`：

- Port 端口从22改为2222
- PasswordAuthentication 从no改为yes

`sudo service ssh restart`重启ssh服务。

此时从windows下`ssh -p 2222 localhost`应该可以访问

## windows下配置端口转发

使用管理员启动powershell

`netsh interface portproxy add v4tov4 listenport=22 listenaddress=0.0.0.0 connectport=2222 connectaddress=127.0.0.1`
监听22端口，转发到内部的2222端口，即wsl的ssh服务

此时从windows下`ssh 192.168.10.108`应该可以访问

## windows下配置防火墙

`netsh advfirewall firewall add rule name="forwarded_SSHport_22" protocol=TCP dir=in localport=22 action=allow`
允许防火墙访问22端口

此时从另一台电脑下`ssh 192.168.10.108`应该可以访问，成功。

` netsh advfirewall firewall show rule name=forwarded_SSHport_22`

可以查看配置的规则

## 设置windows开机自动启动wsl-ssh

打开windows下的task schedular, 新增task在用户登录时触发"wsl -u root service ssh start"，验证成功

## MAC与IP绑定

对于家里使用而言，需要把mac与ip绑定，这样才能ip不丢失。当前的路由器支持这种功能。
