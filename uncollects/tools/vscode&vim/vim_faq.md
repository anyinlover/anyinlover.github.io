# Vim FAQ

## Count the pattern shot number

`:%s/pattern//gn`

## Delete the pattern shot line

`:g/pattern/d`

## Filter the pattern shot line

`:v/pattern/d`

## How to switch bewteen files in vim

* `bf`: Go to first file
* `bl`: Go to last file
* `bn`: Go to next file
* `bp`: Go to previous file

## vim行尾空格替换为tab

```vim
\( \+\)\(\d\+$\)
:%s//\t\2/
```
