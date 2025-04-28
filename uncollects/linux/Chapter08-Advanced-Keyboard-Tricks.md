---
tags:
  - Linux
---

# Chapter 08 - Advanced Keyboard Tricks

- clear - Clear the screen
- history - Display the contents of the history list

## Command Line Editing

### Cursor Movement

Cursor Movement Commands

|  Key   |                  Action                  |
| :----: | :--------------------------------------: |
| Ctrl-a | Move cursor to the beginning of the line |
| Ctrl-e |    Move cursor to the end of the line    |
| Ctrl-f |    Move cursor forward one character     |
| Ctrl-b |    Move cursor backward one character    |
| Alt-f  |       Move cursor forward one word       |
| Alt-b  |      Move cursor backward one word       |
| Ctrl-l |             Clear the screen             |

### Modifying Text

Text Editing Commands

|  Key   |                                       Action                                        |
| :----: | :---------------------------------------------------------------------------------: |
| Ctrl-d |                     Delete the character at the cursor location                     |
| Ctrl-t |  Transpose(exchange)the character at the cursor location with the one preceding it  |
| Alt-t  |         Transpose the word at the cursor location with the one preceding it         |
| Alt-l  | Convert the characters from the cursor location to the end of the word to lowercase |
| Alt-u  | Convert the characters from the cursor location to the end fo the word to uppercase |

### Cutting And Pasting (Killing And Yanking) Text

Cut And Paste Commands

|      Key      |                                 Action                                  |
| :-----------: | :---------------------------------------------------------------------: |
|    Ctrl-k     |          Kill text from the cursor location to the end of line          |
|    Ctrl-u     |    Kill text from the cursor location the the beginning of the line     |
|     Alt-d     |    Kill text from the cursor location to the end of the current word    |
| Alt-Backspace | Kill text from the cursor location to the beginning of the current word |
|    Ctrl-y     |    Yank text from the kill-ring and insert it at the cursor location    |

- The Meta key -> Alt

## Completion

Tab

Completion Commands

|  Key   |                                  Action                                  |
| :----: | :----------------------------------------------------------------------: |
| Alt-?  | Display list of possible completions, or press the tab key a second time |
| Alt-\* |                     Insert all possible completions                      |

- Programmable Completion

## Using History

The history of commands is kept in home directory in a file called .bash_history.

### Searching History

View the contents of the history list

    history | less

Find a command

    history | grep /usr/bin

Use history line

    !88

Incremental search

- Ctrl-r to start and find next one
- Ctrl-g or Ctrl-c to quit searching
- Enter to execute the command
- Ctrl-j to copy the command for editing

History Commands

|  Key   |                                  Action                                  |
| :----: | :----------------------------------------------------------------------: |
| Ctrl-p |                    Move to the previous history entry                    |
| Ctrl-n |                      Move to the next history entry                      |
| Alt-<  |              Move to the beginning(top) of the history list              |
| Alt->  |               Move to the end(bottom) of the history list                |
| Ctrl-r |                        Reverse incremental search                        |
| Alt-p  |                              Reverse search                              |
| Alt-n  |                              Forward search                              |
| Ctrl-o | Execute the current item in the history list and advance to the next one |

### History Expansion

History Expansion Commands

| Sequence |                       Action                       |
| :------: | :------------------------------------------------: |
|    !!    |              Repeat the last command               |
| !number  |          Repeat history list item number           |
| !string  | Repeat last history list item starting with string |
| !?string |  Repeat last history list item containing string   |

- Script: to recond an entire shell session and store it in a file
