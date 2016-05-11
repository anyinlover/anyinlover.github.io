---
layout: post
title: "Chapter 06 - Redireaction"
subtitle: "Linux命令行第六讲"
date: 2016-5-10
author: "Anyinlover"
catalog: true
tags:
  - Linux
  - Linux命令行
---

* > - Redirect standard output to a file
* >> - Append redirected output to a file
* 2> - Redirect standard error to a file
* &> - Redirect standard output and error to a same file
* cat - Concatenate files
* \| - Pipelines
* sort - Sort the lines
* uniq - Report or omit repeated lines
* wc - Print line, word, and byte counts
* grep - Print lines matching a pattern
* head/tail - Print first/last part of files
* tee - Read from stdin and output to stdout and files

## Standard Input, Output, And Error
Everything is a file

## Redirecting Standard Output

	ls -l /usr/bin > ls-output.txt

Create a new, empty file

	> ls-output.txt

Append redirected output to a file

	ls -l /usr/bin >> ls-output.txt
	
## Redirecting Standard Error

	ls -l /bin/usr 2> ls-error.txt
	
### Redirecting Standard Output And Standard Error To One File

	ls -l /bin/usr > ls-output.txt 2>&1

Second modern way

	ls -l /bin/usr &> ls-output.txt
	
Append the standard output and standard error streams to a single file

	ls -l /bin/usr &>> ls-output.txt
	
### Disposing Of Unwanted Output

	ls -l /bin/usr 2> /dev/null
	
## Redirecting Standard Input

### cat - Concatenate Files

	cat [file...]
	
Display short text files

	cat ls-output.txt
	
Join files together

	cat movie.mpeg.0* > movie.mpeg
	
Read from standard input

	cat

* Type Ctrl+d to reach end of file(EOF)

Create short text files

	cat > lazy_dog.txt

Redirect standard input

	cat < lazy_dog.txt


## Pipelines

	command1 | command2
	ls -l /usr/bin | less

### Filters

	ls /bin /usr/bin | sort | less
	
### uniq - Report Or Omit Repeated Lines

Omit repeated lines

	ls /bin /usr/bin | sort | uniq | less
	
report repeated lines

	ls /bin /usr/bin | sort | uniq -d | less
	
### wc - Print Line, Word, And Byte Counts

	wc ls-output.txt
	wc /bin /usr/bin | sort | uniq | wc -l
	
### grep - Print Lines Matching A Pattern

	grep pattern [file...]
	ls /bin /usr/bin | sort |uniq |grep zip
	
`-i` ignore case
`-v` print lines that do not match the pattern

### head/tail - Print First/Last Part Of Files

	head -n 5 ls-output.txt
	tail -n 5 ls-output.txt
	ls /usr/bin | tail -n 5
	
view files in real-time

	tail -f /var/log/syslog
	
### tee - Read From Stdin And Output To Stdout And Files

	ls /usr/bin | tee ls.txt | grep zip