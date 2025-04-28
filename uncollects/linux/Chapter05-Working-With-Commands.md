---
tags:
  - Linux
---

# Chapter 05 - Working with Commands

- type - Display a command's type
- which - Display an executable's location
- help - Get help for shell builtins
- --help - Display usage information for an executable
- man - Display a program's manual page
- apropos - Display appropriate commands
- whatis - Display a brief description of a command
- info - Display a CNU program's info entry
- alias - Create a alias for commands

## What Exactly Are Commands

1. An executable program
2. A command built into the shell itself
3. A shell function
4. An alias

## Identifying Commands

### type - Display A Command's Type

    type command

### which - Display An Executable's Location

    which ls

## Getting A Command's Documentation

### help - Get Help For Shell Builtins

    help cd

### --help - Display Usage Information

    mkdir --help

### man - Display A Program's Manual Page

    man ls

Man Page Organization

| Section | Contents                                       |
| :------ | :--------------------------------------------- |
| 1       | User commands                                  |
| 2       | Programming interfaces kernel system calls     |
| 3       | Programming interfaces to the C library        |
| 4       | Special files such as device nodes and drivers |
| 5       | File formats                                   |
| 6       | Games and amusements such as screen savers     |
| 7       | Miscellaneous                                  |
| 8       | System administration commands                 |

    man section search_term

    man 5 passwd

### apropos - Display Appropriate Commands

    apropos floppy

### whatis - Display A Very Brief Description Of A Command

    whatis ls

### info - Display A GNU Program's Info Entry

    info coreutils

info Commands

| Command           | Action                                                                       |
| :---------------- | :--------------------------------------------------------------------------- |
| ?                 | Display command help                                                         |
| PgUp or Backspace | Display previous page                                                        |
| PgDn or Space     | Display next page                                                            |
| n                 | Next - Display the next node                                                 |
| p                 | Previous - Display the previous node                                         |
| u                 | Up - Display the parent node of the currently displayed node, usually a menu |
| Enter             | Follow the hyperlink at the cursor lacation                                  |
| q                 | Quit                                                                         |

### README And Other Program Documentation Files

Files end with ".gz" can be opened by zless.

## Creating Your Own Commands With alias

put more than one command on a line by ;

    command1; command2; command3...

    cd /usr; ls; cd -

Create an alias

    alias name='string'

    alias foo='cd /usr; ls; cd -'

Remove an alias

    unalias foo

See all the aliases defined in the environment

    alias
