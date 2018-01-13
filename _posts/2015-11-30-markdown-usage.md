---
layout: single
title: "Markdown's Usage"
category: 笔记
tags:
  - 工具
  - markdown
  - 语法
---

> Markdown is a text-to-HTML conversion tool for web writers. Markdown allows you to write using an easy-to-read, easy-to-write plain text format, then convert it to structurally valid XHTML (or HTML).

I have known Markdown for a year. First use the Marxico as a supplement of evernote. But today I want to find a freedom way to write technical articles. Because the jekyll's default markdown is Kramdown, and I research all the markdown syntaxs and find Kramdown is the most active and functional. So I just learn and review the basic syntaxs with the experience using Marxico.

I use the [Kramdown Quick Reference] as the learning material.

[kramdown Quick Reference]: http://kramdown.gettalong.org/quickref.html


## Block-level Elements

### Paragraphs
You have to add a blank line to

separate different block level elements.

Explicit line breaks in a paragraph can be
made by using two spaces at the end of a line.

### Headers

I prefer the "#" style headers.

Different numbers of "#" show the different level of headers just like this article.

A header must be preceded by a blank line.

### Blockquotes


> A blockquote is started using the ">" marker
>
> #### Other block-level elements can be used in a blockquote
>
> > Use the Nested blockquotes
>
> if there is no blank line
even no ">" is OK.

After a blank line it's a separate paragraph.

### Code Blocks

    Lines indented with one tab is the code block.

~~~ ruby
use tilde characters to
    specify a code block with a language.
~~~

### Horizontal Rules

***

Use three or more asterisks to insert a horizontal rule.

### Lists

1. Start with a number, a period, a space and the content.
2. > have a blockquote here
3. a paragraph
here
4. Nested list
    1. hello
    2. hey
    3. hi
5. several elements

## Why not

> Look so ugly

    * no number
    * equal treatment

### Tables

| Name   | Address   | Tel   |
|:-----|:-------:|----:|
| mike | Xi'an   |110  |
| Alice| Shanghai|120  |
| John | Beijing |119  |

## Span-Level Elements

### Emphasis

Use asterisks to *emphasized words*.

Use double asterisks to **strong emphasized words**.

### Links and Images

Here is my [homepage]

[homepage]: http://anyinlover.github.io

Here is a picture:

![](\img\flower.jpg)

### Inline Code

Use `backticks to surrouding the inline code`

Use `` `two backticks to express literal backticks in the code` ``

### Footnotes

Set a footnote marker[^1].

[^1]: Consists of square brackets with a caret and the footnote name.

Need a better one[^2]?

[^2]: > Surprise!
