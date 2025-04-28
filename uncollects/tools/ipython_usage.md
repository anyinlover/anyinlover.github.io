---
tags:
  - IPython
---

# IPython Usage

## IPython Basic

launch IPython

~~~ shell
ipython
~~~

basic usage

~~~ python
a = 1
a
from numpy.random import randn
data = {i:randn() for i in range(7)}
data
~~~

## Tab Completion

~~~ python
an_apple = 27
an_example = 42
an<Tab>
b = [1, 2, 3]
b.<Tab>
import datatime
datatime.<Tab>
book_scripts/<Tab>
path = 'book_scripts/<Tab>'
~~~

## Introspection

~~~python
b?

def add_numbers(a,b):
    """
    Add two numbers together

    Returns
    --------
    the sum : type of arguments
    """
    return a + b

add_numbers?
add_numbers?? # show the function's source code if possible

np.*load*? # search the IPython namespace
~~~

## The %run Command

~~~python
%run ipython_script_test.py
%run -i ipython_script_test.py # give the script access to variables defined in the IPython namespace.
~~~

`Ctrl+C` to interrupting running code

## Executing Code from the Clipboard

~~~python
%paste

%cpaste # enter '--' alone on the line to stop or use `Ctrl+D`
~~~

## IPython interaction with editors and IDEs

* vim
* Emacs
* PyDev for Eclipse
* Python Tools for VS

## Keyboard Shortcuts

|Command|Description|
|:------|:----------|
|`Ctrl+P` or `uparrow`|Search backward in command history for commands starting with currently-entered text|
|`Ctrl+N` or `down-arrow`|Search forward in command history for commands starting with currently-entered text|
|`Ctrl+R`|Readline-style reverse history search (partial matching)
|`Ctrl+Shift-V`| Paste text from Clipboard|
|`Ctrl+C`|Interrupt currently-executing code|
|`Ctrl+A`|Move cursor to beginning of line|
|`Ctrl+E`|Move cursor to end of line|
|`Ctrl+K`|Delete text from cursor until end of line|
|`Ctrl+U`|Discard all text on current line|
|`Ctrl+F`|Move cursor forward one character|
|`Ctrl+B`|Move cursor back one character|
|`Ctrl+L`|Clear screen|

## Exceptions and Tracebacks

IPython will by default print a full call stack trace if an exception is raised.

The amount of context shown can be controlled using the %xmode magic command.

## Magic commands

|Command|Description|
|:------|:-----|
|`%quickref`|Display the IPython Quick Reference Card|
|`%magic`| Display detailed documentation for all of the available magic commands|
|`%debug`|Enter the interaction debugger at the bottom of the last exception traceback|
|`%hist`|Print command input (and optionally output) history|
|`%pdb`|Automatically enter debugger after any exception|
|`%paste`|Execute pre-formatted Python code from clipboard|
|`%cpaste`|Open a special prompt for manually pasting Python code to be executed|
|`%reset`|Delete all variables/ names defined in interactive namespace|
|`%page OBJECT`| Pretty print the object and display it through a paper|
|`%run script.py`| Run a Python script inside IPython|
|`%prun statement`|Execute statement with cProfile and report the profiler output|
|`%time statment`|Report the execution time of single statement|
|`%timeit statement`|Run a statement multiple times to compute an emsemble average execution time. Useful for timing code with very short execution time|
|`%who, %who_ls, %whos`|Display variables defined in interactive namespace, with varying levels of information/verbosity|
|`%xdel variable`|Delete a variable and attempt to clear any references to the object in the IPython internals|

## Qt-based Rich GUI Console

~~~ shell
ipython qtconsle --pylab=inline
~~~
