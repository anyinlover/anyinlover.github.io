---
layout: post
title: "Chapter 02 - Navigation"
date: 2015-12-13
categories: system
---

>* pwd - Display the current working directory
>* ls - List the files and directories in the current working directory
>* cd - Change the current working directory


# Understanding The File System Tree
* Hierarchical directory structure
* Linux always have a single file system tree

# The Current Working Directory
Display the current working directory

  pwd

Home directory: the only place a regular user is allowed to write files

# Listing The Contents Of A Directory
List the files and directories in the current working directory

  ls

# Changing The Current Working Directory

## Absolute Pathnames

  cd /usr/bin

## Relative Pathnames
`..` refers to the parent directory

  cd ..

`.` refers to the current directory and always can be omit

`cd ./bin` or `cd bin`

# Some Helpful Shortcuts
Changes to home directory

  cd

Changes to previous directory

  cd -

Changes to home directory of user_name

  cd ~user_name

*Important Facts About Filenames*

* Filenames that begin with a period character are hidden. Display them by `ls -a`
* Filenames and commands in Linux are case sensitive
* Linux has no concept of file extension
* Limit the punctuation characters in the names of files you create to `.` period, `-` dash and `_` underscore.
