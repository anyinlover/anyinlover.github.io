---
title: Search For Files
category: 操作系统
tags:
  - Linux
---

- locate - Find files by name
- find - Search for files in a directory hierarchy
- xargs - Build and execute command lines from standard input
- touch - Change file times
- stat - Display file or file system status

## locate - Find Files The Easy Way

`locate bin/zip`
`locate zip | grep bin`

- Where Does The locate Database Come From?
- The locate database is created by another program named updatedb. It run once a day. Manually run it:

  `updatedb`

## find - Find Files The Hard Way

Produce a list of our home directory

`find ~`

Count the number of files:

`find ~ | wc -l`

### Tests

`find ~ -type d | wc -l`

`find ~ -type f | wc -l`

find File Types

| File Type | Description                   |
| :-------: | :---------------------------- |
|     b     | Block special device file     |
|     c     | Character special device file |
|     d     | Directory                     |
|     f     | Regular file                  |
|     l     | symbolic link                 |

`find ~ -type f -name "*.JPG" -size +1M | wc -l`

The "+", "-" notation means larger or smaller.

find Size Units

| Character | Unit                                                         |
| :-------: | :----------------------------------------------------------- |
|     b     | 512-byte blocks. This is the default if no unit is specified |
|     c     | Bytes                                                        |
|     w     | 2-byte words                                                 |
|     k     | Kilobytes (units of 1024 bytes)                              |
|     M     | Megabytes (units of 1048576 bytes)                           |
|     G     | Gigabytes (units of 1073741824 bytes)                        |

find Tests

|      Test      | Description                                                                                                                             |
| :------------: | :-------------------------------------------------------------------------------------------------------------------------------------- |
|    -cmin n     | Match files or directories whose content or attributes were last modified exactly n minutes ago                                         |
|  -cnewer file  | Match files or directories whose contents or attributes were last modified more recently than those of file                             |
|    -ctime n    | Match files or directories whose contents or attributes were last modified n\*24 hours ago                                              |
|     -empty     | Match empty files and directories                                                                                                       |
|  -group name   | Match file or directories belonging to group                                                                                            |
| -iname pattern | Like the -name test but case insensitive                                                                                                |
|    -inum n     | Match files with inode number n. This is helpful for finding all the hard links to a particular inode                                   |
|    -mmin n     | Match files or directories whose contents were last modified n minutes ago                                                              |
|    -mtime n    | Match files or directories whose contents were last modified n\*24 hours ago                                                            |
| -name pattern  | Match files and directories with the specified wildcard pattern                                                                         |
|  -newer file   | Match files and directories whose contents were modified more recently than the specified file                                          |
|    -nouser     | Match file and directories that do not belong to a valid user                                                                           |
|    -nogroup    | Match files and directories that do not belong to a valid group                                                                         |
|   -perm mode   | Match files or directories that have permissions sent to the specified mode. mode may be expressed by either octal or symbolic notation |
| -samefile name | Simoliar to hte -inum test. Mathches files that share the same inode number as file name                                                |
|    -size n     | Match files of size n                                                                                                                   |
|    -type c     | Match files of type c                                                                                                                   |
|   -user name   | Match files or directories belonging to user name                                                                                       |

### Operators

`find ~ \( -type f -not -perm 0600 \) -or \( -type d -not -perm 0700 \)`

find Logical Operators

| Operator | Description                                                                         |
| :------: | :---------------------------------------------------------------------------------- |
| -and, -a | Match if the tests on both sides of the operator are true, it is implied by default |
| -or, -o  | Match if a test on either side of the operator is true                              |
| -not, !  | Match if the test following the operator is false                                   |
|    ()    | Groups tests and operators together to form larger expressions                      |

find AND/OR Logic

| Results of expr1 | Operator | expr2 is...      |
| :--------------: | :------: | :--------------- |
|       True       |   -and   | Always performed |
|      False       |   -amd   | Never performed  |
|       True       |   -or    | Never performed  |
|      False       |   -or    | Always performed |

### Predefined Actions

Predefined find Actions

| Action  | Description                                                                      |
| :-----: | :------------------------------------------------------------------------------- |
| -delete | Delete the currently matching file                                               |
|   -ls   | Perform the equivalent of ls -dils on the matching file                          |
| -print  | Output the full pathname of the matching file to standard output. Default action |
|  -quit  | Quit once a match has been made                                                  |

### User-Defined Actions

`-exec rm '{}' ';'`

Since the brace and semicolon characters have special meaning to the shell, they must be quoted or escaped.

Execute a user-defined action interactively by using the `-ok` in place of `-exec`

`find ~ -type f -name 'foo*' -ok ls -l '{}' ';'`

### Improving Efficiency

Using `+` in place of `;`

`find ~ -type f -name 'foo*' -exec ls -l '{}' +`

### xargs

It accepts input from standard input and converts it into an argument list for a specified command.

`find ~ -type f -name 'foo*' -print | xargs ls -l`

- Dealing With Funny Filenames

  To handle files those containing embedded spaces in their names:
  `find ~ -iname '*.jpg' -print0 | xargs --null ls -l`

### A Return To The Playground

`mkdir -p playground/dir-{001..100}`

The `-p` option cause mkdir to create the parent directories of the specified paths.

`touch playground/dir-{001..100}/file-{A..Z}`

`touch` command is usuallt used to set or update the access, change, and modify times of files. Here can create empty file if the filename argument is a nonexistent file.

### Options

The options are used to control the scope of a find search.

Find Options

|      Option      | Description                                                                                                 |
| :--------------: | :---------------------------------------------------------------------------------------------------------- |
|      -depth      | Direct find to process a directory's files before the directory itself                                      |
| -maxdepth levels | Set the maximum number of levels that find will descend into a directory tree                               |
| -mindepth levels | Set the minimum number of levels that find will descend into a directory tree                               |
|      -mount      | Direct find not to traverse directories that mounted on other file systems                                  |
|     -noleaf      | Direct find not to optimize its search based on the assumption that it is searching a Unix-like file system |
