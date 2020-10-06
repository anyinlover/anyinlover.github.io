---
title: Archiving And Backup
category: 操作系统
tags:
  - Linux
---

- gzip - Compress or expand files
- bzip2 - A Block sorting file compressor
- tar - Tape archiving utility
- zip - Package and compress files
- rsync - Remote file and directory synchronization

## Compressing Files

Compress algorithms fall into two general categories: lossless and lossy.

### gzip

Compress file:

`gzip foo.txt`

Uncompress file:

`gunzip foo.txt`

gzip Options

| Option  | Description                                                                                                                                            |
| :-----: | :----------------------------------------------------------------------------------------------------------------------------------------------------- |
|   -c    | Write output to standard output and keep original files                                                                                                |
|   -d    | Decompress. Act like gunzip                                                                                                                            |
|   -f    | Force compression even if a compressed version of the original file already exists                                                                     |
|   -h    | Display usage information                                                                                                                              |
|   -l    | List compress statistics for each file compressed                                                                                                      |
|   -r    | If one or more arguments on the command line are directories, recursively compress files contained within them                                         |
|   -t    | Display verbose messages while compressing                                                                                                             |
| -number | Set amount of compression. number is an integer in the range of 1(fastest, least compression) to 9 (slowest, most compression). The default value is 6 |

`gzip -tv foo.txt.gz`

`gzip -d foo.txt.gz`

`ls -l /etc | gzip > foo.txt.gz`

To view the contents of a compressed text file, following three ways are same:

`gunzip -c foo.txt.gz | less`

`zcat foo.txt.gz | less`

`zless foo.txt.gz`

### bzip2

Similar to gzip, achieves higher levels of compression at the cost of compression speed. The extension is `.bz2`

`bzip2 foo.txt`

`bunzip2 foo.txt.bz2`

All the options (except for -r) that for gzip are supported in bzip2.

`bzcat foo.txt.bz2 | less`

`bzless foo.txt.bz2`

- Don't be Compressive Compulsive

  Don't do it:

  `gzip picture.jpg`

## Archiving Files

### tar

The classic tool for archiving files

`tar mode[options] pathname...`

tar Modes

| Mode | Description                                               |
| :--: | :-------------------------------------------------------- |
|  c   | Create an archive from a list of files and/or directories |
|  x   | Extract an archive                                        |
|  r   | Append specified pathnames to the end of an archive       |
|  t   | List the contents of an archive                           |

Create a tar archive
`tar cf playground.tar playground`

`f`option is used to specify the name of the tar archive

List the contents of the archive

`tar tf playground.tar`

For a more detailed listing

`tar tvf playground.tar`

Extract the tar archive

`tar xf ../playground.tar`

The default for pathnames is relative, rather than absolute

Extract a single file from an archive

`tar xf archive.tar pathname`

`tar xf ../playground2.tar --wildcards 'home/me/playground/dir-*/file-A'`

tar is often used in conjunction with find

`find playground -name 'file-A' -exec tar rf playground.tar '{}' '+'`

tar can also make use of both standard input and output

`find playground -name 'file-A' | tar cf - --files-from=- | gzip > playground.tgz`

The filename '-' is specified to mean standard input or output.

The `--files-from` or `-T` option causes tar to read its list of pathnames from a file rather than the command line.

GNU tar support both gzip and bzip2 compression directly

`find playground -name 'file-A' | tar czf playground.tgz -T -`

`find playground -name 'file-A' | tar cjf playground.tbz -T -`

Transfer a directory from a remote system to our local system

`ssh remote-sys 'tar cf - Documents' | tar xf -`

### zip

The zip program is both a compression tool and an archiver

`zip options zipfile file...`

`zip -r playground.zip playground`

Extract the contents of a zip file:

`unzip ../playground.zip`

If an existing archive is specified, it is updated rather than replaced.

List and extract selectively from a zip archive

`unzip -l playground.zip playground/dir-087/file-Z`

`-l` option causes unzip to merely list the contents of the archive without extracting the file.

zip ccan make use of standard input and output via `-@` option.

`find playground -name "file-A" | zip -@ file-A.zip`

zip accept standard input

`ls -l /etc/ | zip ls-etc.zip -`

`-` is interprets as "use standard input for the input file"

unzip allows its output to be sent to standard output when the `-p` option is specified

`unzip -p ls-etc.zip | less`

## Synchronizing Files And Directories

`rsync options source destination`

`rsync -av playground foo`

Perform a useful system backup

`sudo rsync -av --delete /etc /home /usr/local /media/BigDisk/backup`

### Using rsync Over A Network

The first way is with another system that has rsync installed, along with a remote shell program such as ssh.

`sudo rsync -av --delete --rsh=ssh /etc/ home /usr/local remote-sys:/backup`

`--rsh-ssh` instruct rsync to use the ssh program as its remote shell

The second way that rsync can be used to synchronize files over a network is by using an rysnc server.

`rsync -av --delete rsync://rsync.gtlib.gatech.edu/fedora-linux-core/development/i386/os fedora-devel`
