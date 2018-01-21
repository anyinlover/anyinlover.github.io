---
category: TLCL
tags:
  - Linux
---

* ping - Send an ICMP ECHO_REQUEST to network hosts
* traceroute - Print the route packets trace to a network host
* netstat - Print network connections, routing tables, interface statistics, masquerade connections, and multicast memberships
* ftp - Internet file transfer program
* wget - Non-interactive network downloader
* ssh - OpenSSH SSH client (remote login program)

## Examining And Monitoring A Network

### ping
`ping linuxcommand.org`

### traceroute
Note: Use tracepath in Ubuntu

`traceroute slashdot.org`

### netstat
Using the "-ie" option to examine the network interfaces in our system

`netstat -ie`

Using the "-r" option will display the kernel's network rounting table

`netstat -r`

## Transporting Files Over A Network

### ftp

### lftp - A Better ftp

### wget
`wget http://linuxcommand.org/index.php`

## Secure Communication With remote Hosts

### ssh

`ssh remote-sys`
`ssh bob@remote-sys`

Execute a single command on a remote system

`ssh remote-sys free`

`ssh remote-sys 'ls *' >dirlist.txt`

* Tunneling With SSH
  `ssh -X remote-sys`

### scp And sftp

`scp remote-sys:document.txt .`

`scp bob@remote-sys:document.txt .`

`sftp remote-sys`

* sftp is supported by many of the graphical file managers like Nautilus.
