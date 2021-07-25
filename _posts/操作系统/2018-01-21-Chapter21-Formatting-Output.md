---
title: Formatting Output
category: 操作系统
tags:
  - Linux
---

- nl - Number lines
- fold - Wrap each line to a specified length
- fmt - A simple text formatter
- pr - Prepare text for printing
- printf - Format and print data
- groff - A document formatting system

## Simple Formatting Tools

### nl - Number Lines

`nl distros.txt | head`

nl Markup

| Markup    | Meaning                      |
| :-------- | :--------------------------- |
| \\:\\:\\: | Start of logical page header |
| \\:\\:    | Start of logical page body   |
| \\:       | Start of logical page footer |

Each of the above markup elements must appear alone on its own line.

Common nl Options

| Option    | Meaning                                                                                                                                                                                       |
| :-------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| -b style  | Set body numbering to style                                                                                                                                                                   |
| -f style  | Set footer numbering to style                                                                                                                                                                 |
| -h style  | Set header numbering to style                                                                                                                                                                 |
| -i number | Set page numbering increment to number                                                                                                                                                        |
| -n format | Sets numbering format, where format is: ln = left justified, without leading zeros; rn = right justified, without leading zeros (default); rz = right justified, with leading zeros |
| -p        | Do not reset page numbering at the beginning of each logical page                                                                                                                             |
| -s string | Add string to the end of each line number to create a separator                                                                                                                               |
| -v number | Set first line number of each logical page to number                                                                                                                                          |
| -w width  | Set width of the line number field to width                                                                                                                                                   |

Notice: need to write the distros-nl.sed first

`sort -k 1,1 -k 2n distros.txt | sed -f distros-nl.sed | nl`

`sort -k 1,1 -k 2n distros.txt | sed -f distros-nl.sed | nl -n rz`

`sort -k 1,1 -k 2n distros.txt | sed -f distros-nl.sed | nl -w 3 -s ' '`

### fold - Wrap Each Line To A Specified Length

`echo "The quick brown fox jumped over the lazy dog." | fold -w 12`

-s option will cause fold to break the line at the last availble space before the line width is reached:

`echo "The quick brown fox jumped over the lazy dog." | fold -w 12 -s`

### fmt - A Simple Text Formatter

`fmt -w 50 fmt-info.txt | head`

-c option can solve the indentation of the first line.

`fmt -cw 50 fmt-info.txt | head`

fmt Options

| Option    | Description                                                                                        |
| :-------- | :------------------------------------------------------------------------------------------------- |
| -c        | Operate in crown margin mode. This preserves the indentation of the first two lines of a paragraph |
| -p string | Only format those lines beginning with the prefix string                                           |
| -s        | Split-only mode. Short lines will not be joined to fill lines                                      |
| -u        | Perform uniform spacing. This means a single space between words and two spaces between sentences  |
| -w width  | Format text to fit within a column width characters wide (default is 75)                           |

The -p option are used to format all begin with the same sequence of characters.

`fmt -w 50 -p '# ' fmt-code.txt`

### pr - Format Text For Printing

`pr -l 15 -w 65 distros.txt`

-l option for page length and -w option for page width

### printf - Format And Print Data

`printf "I formatted the string: %s\n" foo`

`printf "I formatted '%s' as a string.\n" foo`

Common printf Data Type Specifiers_
| Specifier | Description                                                                 |
| :-------- | :-------------------------------------------------------------------------- |
| d         | Format a number as a signed decimal integer                                 |
| f         | Format and output a floating point number                                   |
| o         | Format an integer as an octal number                                        |
| s         | Format a string                                                             |
| x         | Format an integer as a hexadecimal number unsing lowercase a-f where needed |
| X         | Same as X but use uppercase letters                                         |
| %         | Print a literal % symbol (i.e., specify "%%")                               |

`printf "%d, %f, %o, %s, %x, %X\n" 380 380 380 380 380 380`

`%[flags][width][.precision]conversion_specification`

printf Conversion Specification Compenents

| Component  |Description |
| :--------- | :------------------------------------------- |
| flags      | There are five different flags|
| width      | A number specifying the minimum field width|
| .precision | For floating point numbers, specify the number of digits of precision to be output after the decimal point. For string conversion, precision specifies the number of characters to output |

Flags

| Flags | Description                                                                                        |
| :---- | :------------------------------------------------------------------------------------------------- |
| #     | Use the "alternate format" for output. For O, prefixed with 0. For x and X, prefixed with 0x or 0X |
| 0     | Pad the output with zeros                                                                          |
| -     | Left-align the output                                                                              |
| ' '   | Produce a leading space for positive numbers                                                       |
| +     | Sign positive numbers                                                                              |

print Conversion Specification Examples

| Argument    | Format     | Result      |
| :---------- | :--------- | :---------- |
| 380         | "%d"       | 380         |
| 380         | "%#x"      | 0x17c       |
| 380         | "%05d"     | 00380       |
| 380         | "%05.5f"   | 380.00000   |
| 380         | "\$010.5f" | 0380.00000  |
| 380         | "%+d"      | +380        |
| 380         | "%-d"      | 380         |
| abcdefghijk | "%5s"      | abcdefghijk |
| abcdefghijk | "%.5s"     | abcde       |

`printf "%s\t%s\t%s\n" str1 str2 str3`

`printf "Line: %05d %15.3f Result: %+15d\n" 1071 3.14156295 32589`

`prinf "<html>\n\t<head>\n\t\t<title>%s</title>\n\t</head>\n\t<body>\n\t\t<p>%s</p>\n\t</body>\n</html>\n" "Page Title" "Page Content"`

## Document Formatting Systems

Two main families of document formatters:

roff and Tex

### groff

`zcat /usr/share/man/man1/ls.1.gz | head`

`zcat /usr/share/man/man1/ls.1.gz | groff -mandoc -T ascii | head`

If no format is specified, PostScrpt is output by default

`zcat /usr/share/man/man1/ls.1.gz | groff -mandoc | head`

`zcat /usr/share/man/man1/ls.1.gz | groff -mandoc > foo.ps`

It's possible to convert the PostScript file into a PDF

`ps2pdf foo.ps foo.pdf`

Use tbl in distros.sed

Using -t option to groff instructs it to pre-process the text stream with tbl

`sort -k 1,1 -k 2n distros.txt | sed -f distros-tbl.sed | groff -t -T ascii 2>/dev/null`

`sort -k 1,1 -k 2n distros.txt | sed -f distros-tbl.sed | groff -t > foo.ps`
