---
tags: 
    - vim
    - Linux
---

# Vim Usage

## Concepts

### Modes

#### three basic mode in vim

- Normal mode: the default mode where we can run commands
- Insert mode: where we can write text
- Visual mode: where we can visually select a bunch of text to operate on

#### mode switch

- Normal to Insert: i I a A o O r R s S
- Insert to Normal: \<Esc\>
- Normal to Visual: v V
- Visual to Normal: \<Esc\>

| Command | Action                               |
| :-----: | :----------------------------------- |
|    i    | insert text just before the cursor   |
|    I    | insert text at the start of the line |
|    a    | append text just after the cursor    |
|    A    | append text at the end of the line   |
|    o    | open a new line below                |
|    O    | open a new line above                |

Replace

| Command | Action                           |
| :-----: | :------------------------------- |
|    s    | substitute the current character |
|    S    | substitute the current line      |
|    r    | replace the current character    |
|    R    | replace continuous characters    |

v Works on a character basis
V Works on a line basis
c Change the text

## Use help

The reference manual:
`:help usr_toc`
To search for a particular topic in
`:help index`

### :helpgrep

`:helpgrep beginning of a word`
`:cnext`
`:cprev`
`:clist`

### Online forum

- Vim Group search page
- Vim IRC forum

## Jump

To the previous location: C-o
To the next location: C-i
To the link location: C-]

## Editing Basics

### Some concepts

- Buffers
- Swap

### Directory and file

`:w`
`:e`
`:cd ../tmp`
`:pwd`

## Undo

`:earlier 4m`
`:later 45s`
`:undo 5`
`:undolist`

## Search

`/patten`
