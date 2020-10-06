---
layout: single
title: "Chapter 11 - The Environment"
subtitle: "Linux命令行第十一讲"
date: 2016-5-10
author: "Anyinlover"
category: 操作系统
tags:
  - Linux
---

- printenv - Print part or all of the environment
- set - Set shell options
- export - Export environment to subsequently executed programs
- alias - Create an alias for a command
- source - re-read the configuration

## What is Stored In The Environment

Two basic types of data in the environment: environment variables and shell variables.

Some programmatic data: aliases and shell functions

### Examining The Environment

Display environment variables:

    printenv | less

List the value of a specific variable:

    printenv USER

Display both the shell and environment variables:

    set | less

View the contents of a variable:

    echo $HOME

View the aliases:

    alias

### Some interesting Variables

Environment Variables

| Variable |                                                Contents                                                 |
| :------: | :-----------------------------------------------------------------------------------------------------: |
| DISPLAY  |                   The name of your display if you are running a graphical environment                   |
|  EDITOR  |                           The name of the program to be used for text editing                           |
|  SHELL   |                                     The name of your shell program                                      |
|   HOME   |                                   The pathname of your home directory                                   |
|   LANG   |                     Defines the character set and collation order of your language                      |
|  PAGER   |                          The name of the program to be used for paging output                           |
|   PATH   | A colon-separated list of directories that are searched when you enter the name of a executable program |
|   PS1    |                     Prompt String 1. This difines the contents of your shell prompt                     |
|   PWD    |                                      The current working directory                                      |
|   TERM   |                                     The name of your terminal type                                      |
|    TZ    |                                         Specifies your timezone                                         |
|   USER   |                                              Your username                                              |

## How Is The Environment Established

Startup Files For Login Shell Sections

|      File       |                                                       Contents                                                        |
| :-------------: | :-------------------------------------------------------------------------------------------------------------------: |
|  /etc/profile   |                                A global configuration script that applies to all users                                |
| ~/.bash_profile |                                            A user's personal startup file                                             |
|  ~/.bash_login  |                          If ~/.bash_profile is not found, bash attempts to read this script                           |
|   ~/.profile    | If neither ~/.bash_profile nor ~/.bash_login is found, bash attempts to read this file. This is the default in Ubuntu |

Startup Files For Non-Login Shell Sessions

|       File       |                        Contents                         |
| :--------------: | :-----------------------------------------------------: |
| /etc/bash.bashrc | A golbal configuration script that applies to all users |
|    ~/.bashrc     |             A user's personal startup file              |

### What's In A Startup File

Tell the shell to make the contents of PATH available.

    export PATH

## Modifying The Environment

### Which Files Should We Modify

.profile: to add directories to your PATH, or define additional environment variables

.bashrc: to everthing else

### Text Editors

nano, vi, emacs

### Using A Text Editor

Create a backup copy before edit an important configuration file.

    cp .bashrc .bashrc.bak
    nano .bashrc

### Activating Our Changes

make bash re-read the modified .bashrc file:

    source .bashrc
