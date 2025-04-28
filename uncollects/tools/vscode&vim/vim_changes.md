# Vim Changes

## Inserting text

| Number | Command | Description                                                   |
| ------ | ------- | ------------------------------------------------------------- |
| N      | a       | append text after the cursor (N times)                        |
| N      | A       | append text at the end of the line (N times)                  |
| N      | i       | insert text before the cursor (N times)                       |
| N      | I       | insert text before the first non-blank in the line (N times)  |
| N      | gI      | insert text in column 1 (N times)                             |
|        | gi      | insert at the end of the last change                          |
| N      | o       | open a new line below the current line, append text (N times) |
| N      | O       | open a new line above the current line, append text (N times) |

in Visual block mode:

| Number | Command | Description                                             |
| ------ | ------- | ------------------------------------------------------- |
|        | I       | insert the same text in front of all the selected lines |
|        | A       | append the same text after all the selected lines       |

## Deleting text

| Number | Command       | Description                                        |
| ------ | ------------- | -------------------------------------------------- |
| N      | x             | delete N characters under and after the cursor     |
| N      | X             | delete N characters before the cursor              |
| N      | d{motion}     | delete the text that is moved over with {motion}   |
|        | {visual}d     | delete the highlighted text                        |
| N      | dd            | delete N lines                                     |
| N      | D             | delete to the end of the line (and N-1 more lines) |
| N      | J             | join N-1 lines (delete EOLs)                       |
|        | {visual}J     | join the highlighted lines                         |
| N      | gJ            | like "J", but without inserting spaces             |
|        | {visual}gJ    | like "{visual}J", but without inserting spaces     |
|        | :[range]d [x] | delete [range] lines [into register x]             |

## Copying and moving text

| Number | Command    | Description                                            |
| ------ | ---------- | ------------------------------------------------------ |
|        | "{char}    | use register {char} for the next delete, yank, or put  |
|        | "*         | use register `*` to access system clipboard            |
|        | :reg       | show the contents of all registers                     |
|        | :reg {arg} | show the contents of registers mentioned in {arg}      |
| N      | y{motion}  | yank the text moved over with {motion} into a register |
|        | {visual}y  | yank the highlighted text into a register              |
| N      | yy         | yank N lines into a register                           |
| N      | Y          | yank N lines into a register                           |
| N      | p          | put a register after the cursor position (N times)     |
| N      | P          | put a register before the cursor position (N times)    |
| N      | ]p         | like p, but adjust indent to current line              |
| N      | [p         | like P, but adjust indent to current line              |
| N      | gp         | like p, but leave cursor after the new text            |
| N      | gP         | like P, but leave cursor after the new text            |

## Changing text

| Number | Command         | Description                                                                                       |
| ------ | --------------- | ------------------------------------------------------------------------------------------------- |
| N      | r{char}         | replace N characters with {char}                                                                  |
| N      | gr{char}        | replace N characters without affecting layout                                                     |
| N      | R               | enter Replace mode (repeat the entered text N times)                                              |
| N      | gR              | enter virtual Replace mode: Like Replace mode but without affecting layout                        |
|        | {visual}r{char} | in Visual block, visual, or visual line modes: Replace each char of the selected text with {char} |
| N      | c{motion}       | change the text that is moved over with {motion}                                                  |
|        | {visual}c       | change the highlighted text                                                                       |
| N      | cc              | change N lines                                                                                    |
| N      | S               | change N lines                                                                                    |
| N      | C               | change to the end of the line (and N-1 more lines)                                                |
| N      | s               | change N characters                                                                               |
|        | {visual}c       | in Visual block mode: Change each of the selected lines with the entered text                     |
|        | {visual}C       | in Visual block mode: Change each of the selected lines until end-of-line with the entered text   |
|        | {visual}~       | switch case for highlighted text                                                                  |
|        | {visual}u       | make highlighted text lowercase                                                                   |
|        | {visual}U       | make highlighted text uppercase                                                                   |
|        | g~{motion}      | switch case for the text that is moved over with {motion}                                         |
|        | gu{motion}      | make the text that is moved over with {motion} lowercase                                          |
|        | gU{motion}      | make the text that is moved over with {motion} uppercase                                          |
| N      | CTRL-A          | add N to the number at or after the cursor                                                        |
| N      | CTRL-X          | subtract N from the number at or after the cursor                                                 |
| N      | <{motion}       | move the lines that are moved over with {motion} one shiftwidth left                              |
| N      | <<              | move N lines one shiftwidth left                                                                  |
| N      | >{motion}       | move the lines that are moved over with {motion} one shiftwidth right                             |
| N      | >>              | move N lines one shiftwidth right                                                                 |
| N      | gq{motion}      | format the lines that are moved over with {motion} to 'textwidth' length                          |

## Complex changes

| Number | Command                                     | Description                                                                                       |
| ------ | ------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| N      | `!{motion}{command}<CR>`                    | filter the lines that are moved over through {command}                                            |
| N      | `!!{command}<CR>`                           | filter N lines through {command}                                                                  |
|        | `{visual}!{command}<CR>`                    | filter the highlighted lines through {command}                                                    |
|        | `:[range]! {command}<CR>`                   | filter [range] lines through {command}                                                            |
|        | :[range]s[ubstitute]/{pattern}/{string}/[g] | substitute {pattern} by {string} in [range] lines; with [g], replace all occurrences of {pattern} |
|        | &                                           | Repeat previous ":s" on current line without options                                              |

## Text objects (only in Visual mode or after an operator)

| Number | Command    | Description                                         |
| ------ | ---------- | --------------------------------------------------- |
| N      | aw         | Select "a word"                                     |
| N      | iw         | Select "inner word"                                 |
| N      | aW         | Select "a WORD"                                     |
| N      | iW         | Select "inner WORD"                                 |
| N      | as         | Select "a sentence"                                 |
| N      | is         | Select "inner sentence"                             |
| N      | ap         | Select "a paragraph"                                |
| N      | ip         | Select "inner paragraph"                            |
| N      | a], a[     | select '[' ']' blocks                               |
| N      | i], i[     | select inner '[' ']' blocks                         |
| N      | ab, a(, a) | Select "a block" (from "[(" to "])")                |
| N      | ib, i), i( | Select "inner block" (from "[(" to "])")            |
| N      | a>, a<     | Select "a &lt;&gt; block"                           |
| N      | i>, i<     | Select "inner <> block"                             |
| N      | aB, a{, a} | Select "a Block" (from "[{" to "]}")                |
| N      | iB, i{, i} | Select "inner Block" (from "[{" to "]}")            |
| N      | at         | Select "a tag block" (from `<aaa>` to `</aaa>`)     |
| N      | it         | Select "inner tag block" (from `<aaa>` to `</aaa>`) |
| N      | a'         | Select "a single quoted string"                     |
| N      | i'         | Select "inner single quoted string"                 |
| N      | a"         | Select "a double quoted string"                     |
| N      | i"         | Select "inner double quoted string"                 |
| N      | a`         | Select "a backward quoted string"                   |
| N      | i`         | Select "inner backward quoted string"               |

## Repeating commands

| Number | Command                           | Description                                                                                        |
| ------ | --------------------------------- | -------------------------------------------------------------------------------------------------- |
| N      | .                                 | repeat last change (with count replaced with N)                                                    |
|        | q{a-z}                            | record typed characters into register {a-z}                                                        |
|        | q{A-Z}                            | record typed characters, appended to register {a-z}                                                |
|        | q                                 | stop recording                                                                                     |
| N      | @{a-z}                            | execute the contents of register {a-z} (N times)                                                   |
| N      | @@                                | repeat previous @{a-z} (N times)                                                                   |
|        | :@{a-z}                           | execute the contents of register {a-z} as an Ex command                                            |
|        | :@@                               | repeat previous :@{a-z}                                                                            |
|        | :[range]g[lobal]/{pattern}/[cmd]  | execute Ex command [cmd](default: ':p') on the lines within [range] where {pattern} matches        |
|        | :[range]g[lobal]!/{pattern}/[cmd] | execute Ex command [cmd](default: ':p') on the lines within [range] where {pattern} does NOT match |

## Undo/Redo commands

| Number | Command | Description                | Note                                                       |
| ------ | ------- | -------------------------- | ---------------------------------------------------------- |
| N      | u       | undo last N changes        | Current implementation may not cover every case perfectly. |
| N      | CTRL-R  | redo last N undone changes | As above.                                                  |
|        | U       | restore last changed line  |                                                            |
