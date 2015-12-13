---
layout: post
title: "Chapter 04 - Manipulating Files and Directories"
date: 2015-12-13
categories: system
---

>* mkdir - Create directories
* cp - Copy files and directories
* mv - Move and rename files
* rm - Remove files and directories
* ln - Create links

# Wildcards

*Wildcard*

|Wildcard|Meaning|
|:-----:|:-----|
|*|Matches any characters|
|?|Matches any single character|
|[characters]|Matches any character that is a member of the set characters|
|!characters|Matches any character that is not a member of the set characters|
|[[:class:]]|Matches any character that is a member of the specified class|

*Commonly Used Character Classes*

|Character Class|Meaning|
|:---:|:---:|
|[:alnum:]| Matches any alphanumeric character|
|[:alpha:]|Matches any alphabetic character|
|[:digit:]|Matches any numeral|
|[:lower:]|Matches any lowercase letter|
|[:upper:]|Matches any uppercase letter|

*Wildcard Examples*

|Pattern|Matches|
|:----:|:-----|
|*|All files|
|g*|Any file beginning with "g"|
|b*.txt|Any file beginning with "b" followed by any characters and ending with ".txt"|
|Data???|Any file beginning with "Data" followed by exactly three characters|
|[abc]*|Any file beginning with either an "a", a "b", or a "c"|
|BACKUP.[0-9][0-9][0-9]|Any file beginning with "BACKUP." followed by exactly three numerals|
|[[:upper:]]*| Any file beginning with an uppercase letter|
|[![:digit:]]*| Any file not beginning with a numeral|
|*[[:lower:]123]| Any file ending with a lowercase letter or the numerals "1", "2", "3"|

# mkdir - Create Directories

  mkdir directory ...

* When three periods follow an argument, it means the argument can be repeated

  mkdir dir1

  mkdir dir1 dir2 dir3

# cp - Copy Files And Directories

copy the single file or directory to file or directory

  cp item1 item2

copy multiple items(either files or directories) into a directory

## Useful Options And Examples

*cp Options*

|Option|Meaning|
|:---:|:---|
|-a, --archive|Copy the files and directories and all of their attributes|
|-i, --interactive|Before overwriting an existing file, prompt the user for confirmation|
|-r, --recursive|Recursively copy directories and their contents|
|-u, --update| Only copy files that either don't exist, or are newer|
|-v, --verbose| Display informative messages as the copy is performed|

*cp Examples*

|Command|Results|
|:----:|:----|
|cp file1 file2| If file2 exists, overwrite. Not exist, create|
|cp -i file1 file2|Same as above, except prompt before overwrite|
|cp file1 file2 dir1| Copy file1 and file2 into dir1|
|cp dir1/* dir2| Copy all the files in dir1 into dir2|
|cp -r dir1 dir2| If dir2 exist, Copy dir1(and its contents)into dir2. If not, create a dir2 contain same contents as dir1|

# mv - Move And Rename Files
move or rename file or directory "item1" to "item2"

  mv item1 item2
move one or more items from one directory to another

  mv item... directory

## Useful Options And Examples

*mc Options*

|Option|Meaning|
|:----:|:----|
|-i, --interactive| Before overwriting an existing file, prompt the user for confirmation|
|-u, --update|Only move files that either don't exist, or are newer|
|-v, --verbose| Display informative messages as the move is performed|

*mv Examples*

|Command|Results|
|:-----:|:-----|
|mv file1 file2|If file2 exists, overwrite. Not exist, create|
|mv -i file1 file2|Same as above, except prompt before overwrite|
|mv file1 file2 dir1| Move file1 and file2 into dir1|
|mv -r dir1 dir2| If dir2 exist, Move dir1(and its contents)into dir2. If not, create a dir2 contain same contents as dir1|

# rm - Remove Files And Directories

romove files and directories
  rm item...

## Useful Options And Examples

*rm Options*

|Option|Meaning|
|:-----:|:-----|
|-i, --interactive| Before deleting an existing file, prompt the user for confirmation|
|-r, --recursive| Recursively delete directories|
|-f, --force|Ignore nonexisten files and do not prompt, overrides the -i option|
|-v, --verbose| Display informative messages as the deletion is performed|

*rm Examples*

|Command| Results|
|:----:|:----|
|rm file1|Delete file1 silently|
|rm -i file1| Same as above, except prompt before overwrite|
|rm -r file1 dir1| Delete file1 and dir1 and its contents|
|rm -rf file1 dir1| Same as above, except if either file1 or dir1 not exist, continue silently|

* Be Careful With rm!
* A useful tip: use ls before rm

## In - Create Links

Create a hard link

  ln file link

Create a symbolic link

  ln -s item link

# Hard Links

Two limitations:
1. Cannot reference a file outside its own file system
2. Cannot reference a directory

Hard Link file has the same inode as the file, `-i`option can reveal inode information

  ls -li

## Symbolic Links

Broken link: The file is deleted before the symbolic link

# Let's Build A Playground

See the << The Linux Command Line >>

* Creating Symlinks With The GUI: 
Hold Ctrl+Shift when drag the file
