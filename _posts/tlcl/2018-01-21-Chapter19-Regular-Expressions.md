---
category: TLCL
tags:
  - Linux
---

### What Are Regular Expressions?

Limited in the POSIX standard

## grep

`ls /usr/bin | grep zip`

`grep [options] regex [file...]`

*grep Options*

| Option | Description                              |
| :----: | :--------------------------------------- |
|   -i   | Ignore case                              |
|   -v   | Invert match                             |
|   -c   | Print the number of matches instead of the lines themselves |
|   -l   | Print the name of each file that contais a match instead of the lines themselves |
|   -L   | Print only the names of files that do not contain matches |
|   -n   | Prefix each matching line with the number of the line within the file |
|   -h   | For multi-file searches, suppress the output of filenames |

`grep bzip dirlist*.txt`

`grep bzip -l dirlist*.txt`

`grep bzip -L dirlist*.txt`

## Metacharacters And Literals

Regular expression metacharacters:

^ $ . [ ] { } - ? * + ( ) | \

* Many of these have meaning to the shell, so it's vital to be enclosed

## The Any Character

`grep -h '.zip' dirlist*.zip`

## Anchors

`grep -h '^zip' dirlist*.txt`

`grep -h 'zip$' dirlist*.txt`

`grep -h '^zip$' dirlist*.txt`

* A Crossword Puzzle Helper
  Linux system contains a dictionary in the /usr/share/dict
  `grep -i '^..j.r$' /usr/share/dict/words`

## Bracket Expressions And Character Classes

`grep -h '[bg]zip' dirlist*.txt`

### Negation

`grep -h '[^bg]zip' dirlist*.txt`

### Traditional Character Ranges

`grep -h '^[ABCDEFGHIJKLMNOPQRSTUVWXYZ]' dirlist*.txt`

`grep -h '^[A-Z]' dirlist*.txt`

`grep -h '^[A-Za-z0-9]' dirlist*.txt`

make the dash character in the first as literal:

`grep -h '[-AZ]' dirlist*.list`

### POSIX Character Classes

*POSIX Character Classes*

| Character Class | Description                              |
| :-------------: | :--------------------------------------- |
|    [:alnum:]    | The alphanumeric characters              |
|    [:word:]     | The same as [:alnum:], with the addition of the underscore (_) character |
|    [:alpha:]    | The alphabetic characters                |
|    [:blank:]    | Includes the space and tab characters    |
|    [:cntrl:]    | The ASCII control codes: 0-31 and 127    |
|    [:digit:]    | The numerals zero through nine           |
|    [:graph:]    | The visible characters: 33-126           |
|    [:lower:]    | The lowercase letters                    |
|    [:punct:]    | The punctuation characters.              |
|    [:print:]    | The printable characters                 |
|    [:space:]    | [\t\r\n\v\f]                             |
|    [:upper:]    | The uppercase characters                 |
|   [:xdigit:]    | Characters used to express hexadecimal numbers |

## POSIX Basic Vs. Extended Regular Expressions
Two kinds:

basic regular expressions (BRE) and extended regular expressions (ERE)

BRE: `^ $ . [ ] *`; Escaped with a backslash: `( ) { } ? + |`

ERE: `^ $ . [ ] * ( ) { } ? + |`

## Alternation

`echo "AAA" | grep -E 'AAA|BBB'`

`grep -Eh '^(bz|gz|zip)' dirlist*.txt`

## Quantifiers

### ? - Match An Element Zero Or One Time
`echo "(555 123-4567)" | grep -E '^\(?[0-9][0-9][0-9]\)? [0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]$'`

### * - Match An Element Zero Or More Times
`echo "This works." | grep -E '[[:upper:]][[:upper][:lower] ]*\.'`

### + - Match An Element One Or More Times
`echo "This that" | grep -E '^([[:alpha:]]+ ?)+$'`

###{} - Match An Element A Specific Number Of Times

*Specifying The Number Of Matches*

| Specifier | Meaning                                  |
| :-------: | :--------------------------------------- |
|    {n}    | Match the preceding element if it occurs exactly n times |
|   {n,m}   | Match the preceding element if it occurs at least n times, but no more than m times |
|   {n,}    | Match the preceding element if it occurs n or more times |
|   {,m}    | Match the preceding element if it occurs no more than m times |

`echo "(555) 123-4567" | grep -E '^\(?[0-9]{3}\)? [0-9]{3}-[0-9]{4}$'1`

## Putting Regular Expressions To Work

### Validating A Phone List With grep

`for i in {1..10}; do echo "(${RANDOM:0:3}) ${RANDOM:0:3}-${RANDOM:0:4}" >> phonelist.txt; done`

Scan the file for invalid numbers

`grep -Ev '^\([0-9]{3}\) [0-9]{3}-[0-9]{4}$' phonelist.txt`

### Finding Ugly Filenames With find

`find . -regex '.*[^-_./0-9a-zA-Z]'`

### Searching For Files With locate

basic (`--regexp` option) and extended (`--regex`option)

`locate --regex 'bin/(bz|gz|zip)'`

### Searching For Text With less And vim

less support extended regex

`/^\([0-9]{3}\) [0-9]{3}-[0-9]{4}$`

vim support basic regex

`/^([0-9]\{3\}) [0-9]\{3\}-[0-9]\{4\}$`
