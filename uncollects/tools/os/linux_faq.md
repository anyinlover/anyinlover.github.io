# Linux FAQ

## CentOS开启swap

```shell
dd if=/dev/zero of=/swapfile count=2048 bs=1MiB
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
swapon -s
```

## 文本乱码转换

```shell
iconv -f GBK -t utf-8 fromfile > tofile
```

## zsh清除失效软链接

```shell
rm -- *(-@D)
```

## 修改systemctl默认编辑器

在`~/.bashrc`中加入`export SYSTEMD_EDITOR=vim`

`sudo visudo`中添加`Defaults  env_keep += "SYSTEMD_EDITOR"`

`source ~/.bashrc`后生效

## curl 修改代理

在.bashrc中添加环境变量

```
export http_proxy="http://10.243.70.187:3128"
export https_proxy="http://10.243.70.187:3128"
```

## wget 修改代理

在.wgetrc中添加环境变量

```
use_proxy=on
http_proxy=10.243.70.187:3128
https_proxy=10.243.70.187:3128
```

## Get openssl cert

`echo -n | openssl s_client -showcerts -connect www.baidu.com:443 2>/dev/null | sed -ne '/-BEGIN CERTIFICATE-/,/-END CERTIFICATE-/p'`
