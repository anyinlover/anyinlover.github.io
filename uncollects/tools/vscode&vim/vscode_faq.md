# vscode FAQ

## Chinese fixed-width font

Ubuntu Mono、inconsolata、Sarasa Mono SC、M+ 1m

## How to change default shell to zsh in flatpak env

[config](https://superuser.com/questions/1714960/how-to-use-zsh-in-flatpak-vscode) in `settings.json`:

```json
"terminal.integrated.profiles.linux": {
    "zsh": {
        "path": "/usr/bin/flatpak-spawn",
        "args": ["--host", "--env=TERM=xterm-256color", "zsh"],
        "overrideName": true,
    }
},
"terminal.integrated.defaultProfile.linux": "zsh",
```
