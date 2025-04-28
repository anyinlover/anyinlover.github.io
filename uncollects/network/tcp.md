# TCP

## What

Transmission Control Protocol is a [[transport_layer]] protocol which provides a reliable, connection-oriented service to the invoking application.

TCP is defined in [RFC 793](https://datatracker.ietf.org/doc/html/rfc793), [RFC 1122](https://datatracker.ietf.org/doc/html/rfc1122), [RFC 2018](https://datatracker.ietf.org/doc/html/rfc2018), [RFC 5681](https://datatracker.ietf.org/doc/html/rfc5681) and [RFC 7323](https://datatracker.ietf.org/doc/html/rfc7323)

Besides [[udp]] provides, it also provides:

1. Reliable data transfer
2. Congestion control

![TCP Segment Structure](../../assets/image/tcp_segment.png)

## Why

### Connection-Oriented multiplexing and demultiplexing

A TCP socket is identified by a four-tuple: source IP address, source port number, destination IP address, destination port number.

The listen socket four-tuple:

```shell
(base) ➜  ~ netstat -ntl | grep 12000
tcp        0      0 0.0.0.0:12000           0.0.0.0:*               LISTEN
```

The connection socket and client socket four-tuple:

```shell
(base) ➜  ~ netstat -nt | grep 12000
tcp        0      0 127.0.0.1:12000         127.0.0.1:45618         ESTABLISHED
tcp        0      0 127.0.0.1:45618         127.0.0.1:12000         ESTABLISHED
```

### TCP Connection Management

![TCP three-way handshake](../../assets/image/tcp_handshake.png)

## How

### How to show all tcp sockets in linux

```netstat -nt```

### How to show all listening tcp sockets in linux

```netstat -ntl```
