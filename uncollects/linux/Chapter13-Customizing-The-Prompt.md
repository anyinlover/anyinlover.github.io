---
tags:
  - Linux
---

# Chapter13 Customizing the Prompt

## Anatomy Of A Prompt

The prompt is defined by an environment variable named PS1

View the contents of PS1:

`echo $PS1`

Escape Codes Used in Shell Prompts

| Sequence |             Value Displayed              |
| :------: | :--------------------------------------: |
|    \a    | ASCII bell. This makes the computer beep when it is encountered |
|    \d    | Current date in day, mouth, date format  |
|    \h    | Hostname of the local machine minus the trailing domain name |
|    \H    |              Full hostname               |
|    \j    | Number of jobs running in the current shell session |
|    \l    |   Name of the current terminal device    |
|    \n    |           A newline character            |
|    \r    |            A carriage return             |
|    \s    |        Name of the shell program         |
|    \t    | Current time in 24 hour hours:minutes:seconds format |
|    \T    |      Current time in 12 hour format      |
|    \@    |   Current time in 12 hour AM/PM format   |
|    \A    | Current time in 24 hour hours:minites format |
|    \u    |       username of the current user       |
|    \v    |       Version number of the shell        |
|    \V    | Version and release number of the shell  |
|    \w    |  Name of the current working directory   |
|    \W    | Last part of the current working directory name |
|    \!    |  History number of the current command   |
|    \#    | Number of commands entered during this shell session |
|    \$    |     Display a "$" or "#"(superuser)      |
|    \[    | Signals the start of a series of non-printing characters |
|    \]    | Signals the end of a non-printing character sequence |

## Trying Some Alternative Prompt Designs

`ps1_old="$PS1"`
`PS1="<\u@\h \W>\$ "`

## Adding Color

Normal: `\033[0m`

Escape Sequences Used To Set text Colors

|  Sequence  | Text Color |  Sequence  |  Text Color  |
| :--------: | :--------: | :--------: | :----------: |
| \033[0;30m |   Black    | \033[1;30m |  Dark Gray   |
| \033[0;31m |    Red     | \033[1;31m |  Light Red   |
| \033[0;32m |   Green    | \033[1;32m | Light Green  |
| \033[0;33m |   Brown    | \033[1;33m |    Yellow    |
| \033[0;34m |    Blue    | \033[1;34m |  Light Blue  |
| \033[0;35m |   Purple   | \033[1;35m | Light Purple |
| \033[0;36m |    Cyan    | \033[1;36m |  Light Cyan  |
| \033[0;37m | Light Grey | \033[1;37m |    White

`PS1="\[\033[0;31m\]<\u@\h \W>\$\[\033[\033[0m\]"`

Escape Sequences Used To Set Background Color
| :--------: | :--------------: | :--------: | :--------------: |
| \033[0;40m |      Black       | \033[0;44m |       Blue       |
| \033[0;41m |       Red        | \033[0;45m |      Purple      |
| \033[0;42m |      Green       | \033[0;46m |       Cyan       |
| \033[0;43m |      Brown       | \033[0;47m |    Light Grey    |

## Moving The Cursor

Cursor Movement Escape Sequences

| Escape Code |                  Action                  |
| :---------: | :--------------------------------------: |
|  \033[l;cH  |  Move the cursor to line l and column c  |
|   \033[nA   |        Move the cursor up n lines        |
|   \033[nB   |       Move the cursor down n lines       |
|   \033[nC   |   Move the cursor forward n characters   |
|   \033[nD   |  Move the cursor backward n characters   |
|   \033[2J   | Clear the screen and move the cursor to the upper left corner (line 0, column 0) |
|   \033[K    | Clear from the cursor position to the end of the current line |
|   \033[s    |    Store the current cursor position     |
|   \033[u    |     Recall the stored cursor postion     |

`PS1="\[\033[s\033[0;0H\033[0;41m\033[K\033[1;33m\t\033[0m\033[u\]<\u@\h \W>\$ "`

## Saving The prompt
