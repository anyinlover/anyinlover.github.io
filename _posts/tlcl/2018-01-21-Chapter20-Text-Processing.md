---
category: TLCL
tags:
  - Linux
---

* cat - Concatenate files and print on the standard ouput
* sort - Sort lines of text lines
* uniq - Report or omit repeated lines
* cut - Remove sections from each line of files
* paste - Merge lines of files
* join - Join lines of two files on a common field
* comm - Compare two sorted files line by line
* diff - Compare files line by line
* patch - Apply a diff file to an original
* tr - Translate or delete characters
* sed - Stream editor for filtering and transforming text
* aspell - Interactive spell checker

## Applications Of Text
### Documents
### Web pages
### Email
### Printer Output
### Program Source Code

## Revisiting Some Old Friends
### cat

`cat > foo.txt`

To display all characters:

`cat -A foo.txt`

* MS-DOS Text Vs. Unix Text
  hidden carriage returns
  dos2unix and unix2dos

-n numbers lines and -s suppress the output of multiple blank lines.

`cat -ns foo.txt`

### sort

`sort > foo.txt`

`sort file1.txt file2.txt file3.txt > final_sorted_list.txt`

*Common sort Options*

| Option | Description                              |
| :----: | :--------------------------------------- |
|   -b   | Ignore leading spaces in lines           |
|   -f   | Makes sorting case-insensitive           |
|   -n   | Performs sorting based on the numeric evaluation of a string |
|   -r   | Sorting in reverse order                 |
|   -k   | Sort based on a key field loccated from field1 to field2 rather than the entire line |
|   -m   | Treat each argument as the name of a presorted file |
|   -o   | Send sorted output to file rather than standard output |
|   -t   | Define the field-separator character     |

`du -s /usr/share/* | head`

The du command lists the results of a summary in pathname order

`du -s /usr/share/* | sort -nr | head`

Sort on a specified field

`ls -l /usr/bin | sort -nr -k 5 | head`

Sort on multiple keys

`sort -k 1, 1 -k 2n distros.txt`

The key option allows specification of offsets within fields

`sort -k 3.7nbr -k 3.1nbr -k 3.4nbr distros.txt`

Define the field separator character

`sort -t ':' -k 7 /etc/passwd | head`

### uniq

Remove any duplicate lines in a sorted file. Often used in conjunction with sort.

`sort foo.txt | uniq`

*Common uniq Options*

| Options | Description                              |
| :-----: | :--------------------------------------- |
|   -c    | Output a list of duplicate lines preceded by the number of times the line occurs |
|   -d    | Only output repeated lines, rather than unique lines |
|  -f n   | Ignore n leading fields in each line     |
|   -i    | Ignore case during the line comparisions |
|  -s n   | Skip (ignore) the leading n characters of each line |
|   -u    | Only output unique lines                 |

`sort foo.txt | uniq -c`

## Slicing And Dicing

### cut

*cut Selection Options*

|    Option     | Description                              |
| :-----------: | :--------------------------------------- |
| -c char_list  | Extract the portion of the line defined by char_list |
| -f field_list | Extract one or more fields from the line as defined by field_list |
| -d delim_char | When -f is specified, use delim_char as the field delimiting character. Default it's a single tab character |
| --complement  | Extract the entire line of text, except for those portions specified by -c and/or -f |

* cut is best used to extract text from files that are produced by other programs, rather than text directly typed by humans

  `cut -f 3 distros.txt`

  `cut -f 3 distros.txt | cut -c 7-10`

* Expanding Tabs

  Program expand can replace the tab characters within the file with the corresponding number of spaces

  `expand distros.txt | cut -c 23-`

  Specify a different field delimiter

  `cut -d ':' -f 1 /etc/passwd | head`

### paste

`paste distros-dates.txt distros-versions.txt`

### join

A join is an operation usually associated with relational databases where data from multiple tables with a shared key field is combined to form a desired result.

`join distros-key-names.txt distros-key-vernums.txt | head`

## Comparing Text

### comm

`comm file1.txt file2.txt`

Support options -n to suppress the columns

`comm -12 file1.txt file2.txt`

### diff

`diff file1.txt file2.txt`

*diff Change Commands*

| Change | Description                              |
| :----: | :--------------------------------------- |
| r1ar2  | Append the lines at the position r2 in the second file to the position r1 in the first line |
| r1cr2  | Change (replace) the lines at position r1 with the lines at the position r2 in the second line |
| r1dr2  | Delete the lines in the first file at position r1, which would have appeared at range r2 in the second file |

View using the context format

`diff -c file1.txt file2.txt`

*diff Context Format Change Indicators*

| Indicator | Meaning                  |
| :-------: | :----------------------- |
|   blank   | A line shown for context |
|     -     | A line deleted           |
|     +     | A line added             |
|     !     | A line changed           |

View using the unified format

`diff -u file1.txt file2.txt`

*diff Unified Format Change Indicators*

| Character | Meaning                                  |
| :-------: | :--------------------------------------- |
|   blank   | This line is shared by both files        |
|     -     | This line was removed from the first file |
|     +     | This line was added to the first file    |

### patch

The patch program is used to apply changes to text files.

`diff -Naur file1.txt file2.txt > patchfile.txt`

`patch < pathfile.txt`

## Editing On the Fly

### tr

The tr program is used to transliterate characters.

`echo "lowercase letters" | tr a-z A-Z`

`echo "lowercase letters" | tr [:lower:] A`

* To convert MS-DOS text files to Unix-style text:
*
  `tr -d '\r' < dos_file> unix_file`

  Squeeze repeated instances of a character (adjoining is needed)

  `echo "aaabbbccc" | tr -s ab`

### sed

The name sed is short for stream editor.

`echo "front" | sed 's/front/back/'`

The choice of the delimiter character is arbitrary

`echo "front" | sed 's_front_back_'`

Most commands in sed may be preceded by an address, which specifies which line(s) will be edited

`echo "front" | sed '1s/front/back/'`

*sed Address Notation*

|   Address    | Description                              |
| :----------: | :--------------------------------------- |
|      n       | A line number where n is a positive integer |
|      $       | The last line                            |
|   /regexp/   | Lines matching a POSIX basic regular expression |
| addr1, addr2 | A range of lines addr1 to addr2, inclusive |
|  first~step  | Match the line represented by the number first, then each subsequent line at step intervals |
|  addr1, +n   | Match addr1 and the following n lines    |
|    addr!     | Match all lines except addr              |

`sed -n '1, 5p' distros.txt`

The opton -n to cause sed not to print every line by default

`sed -n '/SUSE/p' distros.txt`

`sed -n '/SUSE/!p' distros.txt`

The option -i to rewrite the file

`sed -i 's/lazy/laxy/; s/jumped/jimped/' foo.txt`

*sed Basic Editing Commands*

|        Command        | Description                              |
| :-------------------: | :--------------------------------------- |
|           =           | Output current line number               |
|           a           | Append text after the current line       |
|           d           | Delete the current line                  |
|           i           | Insert text in front of the current line |
|           p           | Print the current line                   |
|           q           | Exit sed without processing any more lines, output the current line is -n si not specfied |
|           Q           | Exit sed without processing any more lines |
| s/regexp/replacement/ | Substitute the contents of replacement wherever regexp is found |
|      y/set1/set2      | Perform transliteration by converting characters from set1 to the corresponding characters in set2 |

`sed 's/\([0-9]\{2\}\)\/\([0-9]\{2\}\)\/\([0-9]\{4\}\)$/\3-\1-\2/' distros.txt`

Back reference: `\3-\1\-2`

Change all the instances by adding the g flags:

`echo "aaabbbccc" | sed 's/b/B/g'`

Construct more complex commands in a script file using the -f option

```
# sed script to produce Linux distributions report

1 i\
\
Linux Distribution Report\

s/\([0-9]\{2\}\)\/\([0-9]\{2\}\)\/\([0-9]\{4\}\)$/\3-\1-\2/
y/abcdefghijklmnopqrstuvwxyz/ABCDEFGHIJKLMNOPQRSTUVWXYZ/
```

Save it as distros.sed

`sed -f distros.sed distros.txt`

* People Who Like sed Also Like...
  awk and perl is used for larger tasks.

### aspell

`aspell check textfile`

`aspell check foo.txt`

Include the checking-mode option

`aspell -H check foo.txt`

It is used to check HTML files.
