---
layout: post
title: "Chapter 03 - Explore the System"
subtitle: "Linux命令行第三讲"
date: 2015-12-13
author: "Anyinlover"
catalog: true
tags:
  - Linux
  - Linux命令行
---

* ls - List the files and directories in current working directory
* file - Determine a file's type with file
* less - View file contents

## More Fun With ls

specify multiple directories

    ls ~ /usr

### Options And Arguments

    command -options arguments

short options and long options

    ls -lt --reverse

*Common ls Options*

|Option   |    Long Option   |Description|
| :------| :-----------| :------ |
| -a| --all     |  List all files  |
| -A| --almost-all | Like -a except not list . and ..|
|-d|--directory    |Use with -l to see detail about the directory|
|-F|--classify|Append an indicator character to the end of each listed name|
|-h|--human-readable|Display file sizes in human readable format rather than in bytes|
|-i| --inode| Reveal the inode information|
|-l||Display results in long format|
|-r|--reverse|Display results in reverse order|
|-s||Sort results by file size|
|-t||Sort by modification time|

### A Longer Look At Long Format

    -rw-r--r-- 1 root root 32059 2007-04-03 11:05 oo-cd-cover.odf

*Is Long Listing Fields*

|Field|Meaning|
|:---|:-----|
|`-rw-r--r--`|Access rights to the file|
|`1`|File's number of hard links|
|`root`|The username of the file's owner|
|`root`|The name of the group which owns the file|
|`32059`|Size of the file in bytes|
|`2007-04-03 11:05`|Date and time of the file's last modification|
|`oo-cd-cover.odf`|Name of the file|

## Determining A File's Type With file

    file filename

    file picture.jpg

## Viewing File Contents With less

    less filename

    less /etc/passwd

*less Commands*

|Command|Action|
|:------|:----|
|Page Up or b|Scroll back one page|
|Page Down or space|Scroll forward one page|
|Up Arrow|Scroll up one line|
|Down Arrow|Scroll down one line|
|G|Move to the end of the text file|
|1G or g|Move to the beginning of the text file|
|/characters|Search forward to the next occurrence of characters|
|n|Search for the next occurrence of the previous search|
|h|Display help screen|
|q|Quit less|


* Less Is More

## A Guided Tour

*Directories Found On Linux Systems*

|Directory|Comments|
|:-----|:------|
|/|The root directory|
|/bin|Contains binaries that must be present to boot and run|
|/boot|Contains the Linux kernel, initial RAM and the boot loader|
|/dev|Contains device nodes|
|/etc|Contains all the system-wide configuration files|
|/home|Contains users' Directory|
|/lib|Contains shared library files used by the core system programs|
|/lost+found|Used in the case of a partial recovery from a file system corruption event|
|/media|Contain the mount points for removable media such as USB|
|/mnt|Contains mount points for removable devices mounted manuallly|
|/opt| used to install "optional" software mainly for commercial software|
|/proc|A virtual file system, Contains feepholes into the kernel|
|/root|The home directory for the root account|
|/sbin| Contains "system"binaries, perform vital system tasks|
|/tmp|Contains temporary files created by various programs|
|/usr|Contains all the programs and support files used by regular users|
|/usr/bin|Contains the executable programs installed by Linux distribution|
|/usr/lib| The shared libraries for the programs in /usr/bin|
|/usr/local| Contain programs complied from source code|
|/usr/sbin| Contains more system administration programs|
|/usr/share|Contains all the shared data used by programs in /usr/bin|
|/usr/share/doc|Contains package documentations|
|/var|Where data that is likely to change is stored|
|/var/log| contains log files records of various system activity|

## Symbolic Links

    lrwxrwxrwx 1 root root 11 2007-08-11 07:34 libc. so. 6 -> libc-2.6.so

## Hard Links
A second type of link
