# Github

- On GitHub, you can freely fork open-source repositories.
- You have read and write access to the forked repository.
- You can push pull requests to the official repository to contribute code.

When connecting to GitHub using SSH, you need to configure an SSH Key. If there is no hidden folder `.ssh` in the home directory, create one using `ssh-keygen -t rsa -C "anyinlover@gmail.com"`. In the `.ssh` folder, there are two files: id_rsa and id_rsa.pub. Copy the content of id_rsa.pub to GitHub SSH Keys, ensuring there are no extra spaces or line breaks.

Sometimes The Chinese Firewall refuse to allow SSH connections to github. We can [use ssh over the https port](https://docs.github.com/en/authentication/troubleshooting-ssh/using-ssh-over-the-https-port).

```shell
# ~/.ssh/config
Host github.com
    Hostname ssh.github.com
    Port 443
    User git
```
