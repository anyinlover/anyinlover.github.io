# Vim motions

## Left-right motions

| Number | Command | Description                                                                    |
| ------ | ------- | ------------------------------------------------------------------------------ |
| N      | h       | left                                                                           |
| N      | l       | right                                                                          |
|        | 0       | to first character in the line                                                 |
|        | ^       | to first non-blank character in the line                                       |
| N      | $       | to the last character in the line (N-1 lines lower)                            |
|        | g0      | to first character in screen line (differs from "0" when lines wrap)           |
|        | g^      | to first non-blank character in screen line (differs from "^" when lines wrap) |
| N      | g$      | to last character in screen line (differs from "$" when lines wrap)            |
|        | gm      | to middle of the screen line                                                   |
| N      | \|      | to column N (default: 1)                                                       |
| N      | f{char} | to the Nth occurrence of {char} to the right                                   |
| N      | F{char} | to the Nth occurrence of {char} to the left                                    |
| N      | t{char} | till before the Nth occurrence of {char} to the right                          |
| N      | T{char} | till before the Nth occurrence of {char} to the left                           |
| N      | ;       | repeat the last "f", "F", "t", or "T" N times                                  |
| N      | ,       | repeat the last "f", "F", "t", or "T" N times in opposite direction            |

## Up-down motions

| Number | Command | Description                                                                               |
| ------ | ------- | ----------------------------------------------------------------------------------------- |
| N      | k       | up N lines                                                                                |
| N      | j       | down N lines                                                                              |
| N      | -       | up N lines, on the first non-blank character                                              |
| N      | +       | down N lines, on the first non-blank character                                            |
| N      | _       | down N-1 lines, on the first non-blank character                                          |
| N      | G       | goto line N (default: last line), on the first non-blank character                        |
| N      | gg      | goto line N (default: first line), on the first non-blank character                       |
| N      | %       | goto line N percentage down in the file; N must be given, otherwise it is the `%` command |
| N      | gk      | up N screen lines (differs from "k" when line wraps)                                      |
| N      | gj      | down N screen lines (differs from "j" when line wraps)                                    |

## Text object motions

| Number | Command | Description                                                 |
| ------ | ------- | ----------------------------------------------------------- |
| N      | w       | N words forward                                             |
| N      | W       | N blank-separated WORDs forward                             |
| N      | e       | N words forward to the end of the Nth word                  |
| N      | E       | N words forward to the end of the Nth blank-separated WORD  |
| N      | b       | N words backward                                            |
| N      | B       | N blank-separated WORDs backward                            |
| N      | ge      | N words backward to the end of the Nth word                 |
| N      | gE      | N words backward to the end of the Nth blank-separated WORD |
| N      | )       | N sentences forward                                         |
| N      | (       | N sentences backward                                        |
| N      | }       | N paragraphs forward                                        |
| N      | {       | N paragraphs backward                                       |
| N      | ]]      | N sections forward, at start of section                     |
| N      | [[      | N sections backward, at start of section                    |
| N      | ][      | N sections forward, at end of section                       |
| N      | []      | N sections backward, at end of section                      |
| N      | [(      | N times back to unclosed '('                                |
| N      | [{      | N times back to unclosed '{'                                |
| N      | ])      | N times forward to unclosed ')'                             |
| N      | ]}      | N times forward to unclosed '}'                             |

## Pattern searches

| Number | Command                     | Description                                           |
| ------ | --------------------------- | ----------------------------------------------------- |
| N      | `/{pattern}[/[offset]]<CR>` | search forward for the Nth occurrence of {pattern}    |
| N      | `?{pattern}[?[offset]]<CR>` | search backward for the Nth occurrence of {pattern}   |
|        | `/<CR>`                     | repeat last search, in the forward direction          |
|        | `?<CR>`                     | repeat last search, in the backward direction         |
| N      | n                           | repeat last search                                    |
| N      | N                           | repeat last search, in opposite direction             |
| N      | *                           | search forward for the identifier under the cursor    |
| N      | #                           | search backward for the identifier under the cursor   |
| N      | g*                          | like "*", but also find partial matches               |
| N      | g#                          | like "#", but also find partial matches               |
|        | gd                          | goto local declaration of identifier under the cursor |

## Marks

| Number | Command   | Description                                                                     |
| ------ | --------- | ------------------------------------------------------------------------------- |
|        | m{a-zA-Z} | mark current position with mark {a-zA-Z}                                        |
|        | `{a-z}    | go to mark {a-z} within current file                                            |
|        | `{A-Z}    | go to mark {A-Z} in any file                                                    |
|        | `{0-9}    | go to the position where Vim was previously exited                              |
|        | ``        | go to the position before the last jump                                         |
|        | `[        | go to the start of the previously operated or put text                          |
|        | '[        | go to the start non-blank character of the previously operated or put text line |
|        | `]        | go to the end of the previously operated or put text                            |
|        | `.        | go to the position of the last change in this file                              |
|        | '.        | go to the start non-blank character of the last change line in this file        |
|        | :marks    | print the active marks                                                          |
| N      | CTRL-O    | go to Nth older position in jump list                                           |
| N      | CTRL-I    | go to Nth newer position in jump list                                           |
|        | :ju[mps]  | print the jump list                                                             |

## Scrolling

| Number | Command | Description                                    |
| ------ | ------- | ---------------------------------------------- |
| N      | CTRL-E  | window N lines downwards (default: 1)          |
| N      | CTRL-D  | window N lines Downwards (default: 1/2 window) |
| N      | CTRL-F  | window N pages Forwards (downwards)            |
| N      | CTRL-Y  | window N lines upwards (default: 1)            |
| N      | CTRL-U  | window N lines Upwards (default: 1/2 window)   |
| N      | CTRL-B  | window N pages Backwards (upwards)             |
|        | zt      | redraw, current line at top of window          |
|        | zz      | redraw, current line at center of window       |
|        | zb      | redraw, current line at bottom of window       |

## Various

| Number | Command | Description                                                                                        |
| ------ | ------- | -------------------------------------------------------------------------------------------------- |
|        | %       | find the next brace, bracket, comment, or "#if"/ "#else"/"#endif" in this line and go to its match |
| N      | H       | go to the Nth line in the window, on the first non-blank                                           |
|        | M       | go to the middle line in the window, on the first non-blank                                        |
| N      | L       | go to the Nth line from the bottom, on the first non-blank                                         |
