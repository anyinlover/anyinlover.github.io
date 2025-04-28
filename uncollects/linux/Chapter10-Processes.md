---
tags:
  - Linux
---

# Chapter 10 - Processes

- ps - Report a snapshot of current processes
- top - Display tasks
- jobs - List active jobs
- bg - Place a job in the background
- fg - Place a job in the foreground
- kill - Send a signal to a process
- killall - Kill processes by name
- shutdown - shutdown or reboot the system

## How A Process Works

init: the kernel launch the program when a system starts up
PID: process ID

## Viewing Processes

Show the processes associated with the current terminal session

    ps

Show all of the processes regardless of what terminal (if any) they are controlled by.

    ps x

Process States

| State |                                                        Meaning                                                        |
| :---: | :-------------------------------------------------------------------------------------------------------------------: |
|   R   |                            Running. This means that the process is running or ready to run                            |
|   S   |     Sleeping. The process is not running; rather, it is waiting for an event, such as keystroke or network packet     |
|   D   |                        Uninterruptible Sleep. Process is waiting for I/O such as a disk drive                         |
|   T   |                                     Stopped. Process has been instructed to stop                                      |
|   Z   | A defunct or "zombie" process. This is a child process that has terminated, but has not been cleaned up by its parent |
|   <   |                                                A high priority process                                                |
|   N   |                                              A process with low priority                                              |

Show processes belonging to every user

    ps aux

BSD style ps Column Headers

| Header |                                         Meaning                                          |
| :----: | :--------------------------------------------------------------------------------------: |
|  USER  |                        User ID. This is the owner of the process                         |
|  %CPU  |                                   CPU usage in percent                                   |
|  %MEM  |                                 Memory usage in percent                                  |
|  VSZ   |                                   virtual memory size                                    |
|  RSS   | Resident Set Size. The amount of physical memory (RAM) the process is using in kilobytes |
| START  |                              Time when the process started                               |

### Viewing Processes Dynamically With top

To see a more dynamic view of the machine's activity

    top

A system summary at the top of the display, followed by a table of processes sorted by CPU activity

- `h` display help
- `q` quit top

## Controlling Processes

A sample program

    xlogo

### Interrupting A Process

press `Ctrl - c`

### Putting A Process In The Background

    xlogo &

List the jobs that have been launched from our terminal

    jobs

### Returning A Process To The Foreground

    fg %1

If we only have one background job, the jobspec is optional

### Stopping (Pausing) A Process

press `Ctrl - z`

### Moving a process from the foreground to the background

    bg %1

## Signals

    kill 28401

### Sending Signals To Processes With kill

    kill [-signal] PID

TERM is sent by default

Common Signals

| Number | Name |                                                Meaning                                                |
| :----: | :--: | :---------------------------------------------------------------------------------------------------: |
|   1    | HUP  |         Hangup. This is a vestige of the good old days when terminals were attached to remote         |
|   2    | INT  |                                       Interrupt. Same as Ctrl-c                                       |
|   9    | KILL | Kill. The kernel immediately terminates the process. A last resort when other termination signas fail |
|   15   | TERM |                        Terminate. The default signal sent by the kill command                         |
|   18   | CONT |                       Continue. This will restore a process after a STOP signal                       |
|   19   | STOP |              Stop. This signal causes a process to pause without terminating. Like KILL               |

    kill -1 13456
    kill -INT 13601

Other Common Signals

| Number | Name  |            Meaning            |
| :----: | :---: | :---------------------------: |
|   3    | QUIT  |             Quit              |
|   11   | SEGV  |    Segmentation Violation     |
|   20   | TSTP  | Terminal Stop. Same as Ctrl-z |
|   28   | WINCH |         Window Change         |

see a complete list of signals

    kill -l

### Sending Signals To Multiple Processes With killall

    killall [-u usr] [-signal] name...

## More Process Related Commands

Other Process Related Commands

| Command |                                                  Description                                                   |
| :-----: | :------------------------------------------------------------------------------------------------------------: |
| pstree  | Outpus a process list arranged in a tree-like pattern showing the parent/child relationships between processes |
| vmstat  |                Outputs a snapshot of system resource usage including, memory, swap and disk I/O                |
|  xload  |                      A graphical program that draws a graph showing system load over time                      |
|  tload  |                            Similar to the xload program, but draws in the terminal                             |
