# Docker faq

## docker 修改默认镜像存放位置

在DOCKER_OPTS中添加-g参数。

`sudo systemctl edit docker.service`

```
[Service]
ExecStart=
ExecStart=/usr/bin/dockerd -g /mnt
```

```
sudo systemctl daemon-reload
sudo systemctl restart docker.service
```

## docker 配置https认证证书

`/etc/docker/certs.d/<docker registry>/ca.crt`

## docker 代理配置

新增`/etc/systemd/system/docker.service.d/http-proxy.conf`

```
[Service]
Environment="HTTP_PROXY=http://10.243.70.187:3128"
Environment="HTTPS_PROXY=http://10.243.70.187:3128"
Environment="NO_PROXY=localhost,127.0.0.1,docker-registry.example.com,.corp"
```

`sudo systemctl daemon-reload`

`sudo systemctl restart docker`

## windows docker 修改存储位置

```powershell
wsl --list -v  //找到 docker-desktop、docker-desktop-data
wsl --shutdown // 确保上面两个处于stopped状态。
mkdir D:\AppData\Local\Docker\wsl\data\
wsl --export docker-desktop-data "D:\AppData\Local\Docker\wsl\data\docker-desktop-data.tar"
wsl --unregister docker-desktop-data
wsl --import docker-desktop-data "D:\AppData\Local\Docker\wsl\data\" "D:\AppData\Local\Docker\wsl\data\docker-desktop-data.tar" --version 2
```




