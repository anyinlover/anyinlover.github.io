---
layout: single
title: "Chapter 07 - Seeing The World As The Shell Sees it"
subtitle: "Linux命令行第七讲"
date: 2016-5-10
author: "Anyinlover"
category: 操作系统
tags:
  - Linux
---

- echo - Display a line of text

## Expansion

    echo this is a test
    echo *

### Pathname Expansion

    echo D*
    echo /usr/*/share

### Tilde Expansion

    echo ~
    echo ~foo

### Arithmetic Expansion

    echo $((2+2))

Arithmetic Operators

| Operator |           Description           |
| :------: | :-----------------------------: |
|    +     |            Addition             |
|    -     |           Subtraction           |
|    \*    |         Multiplication          |
|    /     | Division, only supports integer |
|    %     |             Modulo              |
|   \*\*   |         Exponentiation          |

### Brace Expansion

    echo Front-{A,B,C}-Back
    echo Number_{1..5}

Brace expansions may be nested

    echo a{A{1,2},B{3,4}}b

### Parameter Expansion

    echo $USER

To see a list of available variables

    printenv | less

### Command Substitution

    echo $(ls)
    file $(ls -d /usr/bin/* | grep zip)

## Quoting

### Double Quotes

Parameter expansion, arithmetic expansion and command substitution still take place within double quotes

    echo "$USER $((2+2)) $(cal)"

### Single Quotes

Suppress all expansions

### Escaping Characters

    echo "The balance for user $USER is: \$5.00"
    mv bad\&filename good_filename

Backslash Escape Sequences

| Escape Sequence |     Meaning     |
| :-------------: | :-------------: |
|       \a        |      Bell       |
|       \b        |    Backspace    |
|       \n        |     Newline     |
|       \r        | Carriage return |
|       \t        |       Tab       |

Add the "-e" option to echo will enable interpretation of escape sequences

    sleep 10; echo -e "Time's up\a"

Also can place them inside \$' '

    sleep 10; echo "Time's up" $'\a'
