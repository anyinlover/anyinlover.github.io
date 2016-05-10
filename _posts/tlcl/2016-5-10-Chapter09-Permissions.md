---
layout: post
title: "Chapter 09 - Permissions"
subtitle: "Linux命令行第九讲"
date: 2016-5-10
author: "Anyinlover"
catalog: true
tags:
  - Linux
  - Linux命令行
---

* id - Display user identity
* chmod - Change a file's mode
* umask - Set the default file permissions
* su - Run a shell as another user
* sudo - Execute a command as another user
* chown - Change a file's owner
* chgrp - CHange a file's group ownership
* passwd - Change a user's password

Linux: not only multitasking system, but also multi-user system

## Owners, Group Members, And Everybody Else

Find out information about your identity

	id
	
User accounts are defined in the `/etc/passwd`

Groups are defined in the `/etc/group`

## Reading, Writing, And Executing

`-rw-rw-r-- me	me 0 2008-03-06 14:52 foo.txt`

*File Types*

|Attribute|File Type|
|:---:|:---|
|-|A regular file|
|d|A directory|
|l|A symbolic link. The remaining file attributes are always "rwxrwxrwx" and are dummy values|
|c|A character special file, refers to a device that handles data as a stream of bytes|
|b|A block special file, refers to a device that handles data in blocks|

*Permission Attributes*

|Attribute|Files|Directories|
|:----:|:---|:---|
|r|Allows a file to be opened and read|Allows a directory's contents to be listed if the execute attribute is also set|
|w|Allows a file to be written to or truncated, not allow to be renamed or deleted, which is determined by directory attributes|Allows files within a directory to be created, deleted, and renamed if the execute attribute is also set|
|x|Allows a file to be treated as a program and executed| Allows a directory to be entered|

*Permission Attribute Examples*

|File Attributes|Meaning|
|:----:|:----|
|`-rwx------`|A regular file that is readable, writable, and executable by the file's owner. No one else has any access|
|`-rw-------`|A regular file that is readable and writable by the file's owner. No one else has any access|
|`-rw-r--r--`|A regular file that is readable and writable by the file's owner. Members of the file's owner group may read the file. The file is world-readable|
|`-rwxr-xr-x`|A regular file that is readable, writable, and executable by the file's owner. The file may be read and executed by everybody else|
|`rw-rw----`|A regular file that is readable and writable by the file's owner and members of the file's group owner only.
|`lrwxrwxrwx`|A symbolic link. All symbolic links have "dummy" permissions.|
|`drwxrwx---`|A directory. The owner and the members of the owner group may enter the directory and, create, rename and remove files within the directory.
|`drwxr-x---`|A directory. The owner may enter the directory and create, rename and delete files within the directory. Members of the owner group may enter the directory but cannot create, delete or rename files.| 

## chmod - Change File Mode

*File Modes In binary And Octal*

|Octal|Binary|File Mode|
|:---:|:---:|:---:|
|0|000|`---`|
|1|001|`--x`|
|2|010|`-w-`|
|3|011|`-wx`|
|4|100|`r--`|
|5|101|`r-x`|
|6|110|`rw-`|
|7|111|`rwx`|

	chmod 600 foo.txt

A few common ones: 7(`rwx`), 6(`rw-`), 5(`r-x`), 4(`r--`), and 0(`---`)

*chmod Symbolic Notation*

|Symbol|Meaning|
|:---:|:----|
|u|Short for "user" but means the file or directory owner|
|g|Group owner|
|o|Short for "others", but means world|
|a|Short for "all". The combination of "u", "g", and "o"

*chmod Symbolic Notation Examples*

|Notation|Meaning|
|:----:|:---|
|u+x|Add execute permission for the owner|
|u-x|Remove execute permission from the owner|
|+x|Add execute permission for the owner, group, and world|
|o-rw|Remove the read and write permission from anyone besides the owner and group owner|
|go=rw|Set the group owner and anyone besides the owner to have read and write permission. If eithet the group owner or world previously had execute permissions, they are removed|
|u+x, go=rx|Add execute permission for the owner and set the permissions for the group and others to read and execute|


## umask - Set Default Permissions

See the current value

	umask

Set another value

	umask 0022
	
* Some Special Permissions

setuid bit (4000): set the effective user ID from that of the real user (the user actually running the program) to that of the program's owner.

setgid bit(2000): set the effective group ID from the real group ID of the real user to that of the file owner.

sticky bit(1000): Prevent users from deleting or renaming files unless the user is either the owner of the directory, the owner of the file, or the superuser.

	chmod u+s program`->`-rwsr-xr-x
	chmod g+s dir`->`drwxrwsr-x
	chmod +t dir`->`drwxrwxrwt
	
## Changing Identities

### su - Run A Shell With Substitute User And Group IDs

	su [-[l]] [user]
	
If `-l` option is included, the resulting shell session is a login shell for the specified user.

To start a shell for the superuser

	su -
	
Return to the previous shell

	exit
	
Execute a single command, it's important to enclose the command in quotes

	su -c 'commad'
	
### sudo - Execute A Command As Another User

	sudo command
	
To see the privileges granted by sudo

	sudo -l
	
### chown - Change File Owner And Group

	chown [owner][:[group]] file...
	
*chown Argument Examples*

|Argument|Results|
|:---:|:----|
|bob|Changes the ownership of the file from its current owner to user bob|
|bob:users|Changes the ownership of the file from its current owner to user bob and changes the file group owner to group users.
|:admins|Changes the group owner to the group admins. The file owner is unchanged|
|bob:|Change the file owner from the current owner to user bob and changes the group owner to the login group of user bob|

### chgrp - Change Group Ownership

Used in older versions of Unix to change group ownership.

## Changing Your Password

	passwd [user]
