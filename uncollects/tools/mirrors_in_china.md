# Mirrors in China

## Apt

ubuntu: [aliyun](http://mirrors.aliyun.com/ubuntu)

linux mint: [aliyun](https://mirrors.aliyun.com/linuxmint-packages)

## Flathub

[SJTU FLATHUB](https://mirrors.sjtug.sjtu.edu.cn/docs/flathub)

```sudo flatpak remote-modify flathub --url=https://mirrors.sjtug.sjtu.edu.cn/docs/flathub```

## Homebrew

[TUNA Homebrew](https://mirrors.tuna.tsinghua.edu.cn/help/homebrew/)

```shell
echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"' >> /home/anyinlover/.zshrc
echo 'export HOMEBREW_BREW_GIT_REMOTE="https://mirrors.tuna.tsinghua.edu.cn/git/homebrew/brew.git"' >> /home/anyinlover/.zshrc
echo 'export HOMEBREW_CORE_GIT_REMOTE="https://mirrors.tuna.tsinghua.edu.cn/git/homebrew/homebrew-core.git"' >> /home/anyinlover/.zshrc
```

## NPM

[Taobao NPM](https://npmmirror.com/)

```shell
npm config set registry https://registry.npmmirror.com
```

## RUST

[TUNA Crates](https://mirrors.tuna.tsinghua.edu.cn/help/crates.io-index/)

```shell
mkdir -vp ${CARGO_HOME:-$HOME/.cargo}

cat << EOF | tee -a ${CARGO_HOME:-$HOME/.cargo}/config.toml
[source.crates-io]
replace-with = 'mirror'

[source.mirror]
registry = "sparse+https://mirrors.tuna.tsinghua.edu.cn/crates.io-index/"
EOF
```
