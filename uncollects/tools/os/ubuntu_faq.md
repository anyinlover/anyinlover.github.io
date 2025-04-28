# ubuntu faq

## Ubuntu Install Python

Ubuntu官方源当前还不支持python3.10，为了测试新功能，只能从第三方源里面安装。

```shell
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.10
```

## Ubuntu加载私有证书

将crt证书放置到/usr/local/share/ca-certificates/

然后通过`sudo update-ca-certificates`更新证书。

## Python加载私有证书

通过`certifi.where()`找到证书所在路径，然后将base64的证人内容append到对应文件下。

## Ubuntu repository清理

/etc/apt/sources.list.d 里面删除repository对应文件

最好对应清理GPG key

`sudo apt-key list` 查看对应key列表

`sudo apt-key del "F23C 5A6C F475 9775 95C8  9F51 BA69 3236 6A75 5776"` 删除对应key

## Ubuntu install lastest llvm

```shell
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh all
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 100 --slave /usr/bin/clang++ clang++ /usr/bin/clang++-15 --slave /usr/bin/clang-cpp clang-cpp /usr/bin/clang-cpp-15 --slave /usr/bin/ld.lld ld.lld /usr/bin/ld.lld-15 /usr/bin/clang-format clang-format /usr/bin/clang-format-15 /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-15
export CC="clang"
export CXX="clang++"
export CPP="clang-cpp"
export LD="ld.lld"
```

## Ubuntu install cmake 3.25

官仓里的cmake版本过低，安装最新版本cmake需要三方库

```shell
sudo apt purge --auto-remove cmake
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
sudo apt update
sudo apt install cmake

```

## Ubuntu install gcc 12

需要升级ubuntu版本到22.04，然后直接就可以安装

```shell
sudo apt install gcc-12 g++-12
sudo update-alternateive --install /usr/bin/g++ g++ /usr/bin/g++-12 100
sudo update-alternateive --install /usr/bin/gcc gcc /usr/bin/gcc-12 100
```

