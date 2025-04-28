# Delay

## What

As a packet travels, the packet suffers from several types of delays at each node along the path:

1. nodal processing delay
2. queuing delay
3. transmission delay
4. propagation delay

These delays accumulate to give a total nodal delay

![Nodal Delay](../../assets/image/nodal_delay.png)

## Why

### Nodal Processing Delay

The time required to examine the packet’s header and determine where to direct the packet.

The processing delay can also include other factors, such as the time needed to check for bit-level errors in the packet.

Processing delays in high-speed routers are typically on the order of microseconds or less.

### Queuing Delay

The length of the queuing delay of a specific packet will depend on the number of earlier-arriving packets that are queued and waiting for transmission onto the link.

The average queuing delay depends on traffic intensity $La/R$.

- $L$: the packet length
- $a$: the average rate at which packets arrive at the queue
- $R$: the transmission rate

![Queuing Delay](../../assets/image/queuing_delay.png)

Queuing delays can be on the order of microseconds to milliseconds in practice.

### Transmission Delay

This is the amount of time required to push (that is, transmit) all of the packet’s bits into the link.

The transmission delay is $L/R$.

- $L$: the packet length
- $R$: the transmission rate

Transmission delays are typically on the order of microseconds to milliseconds in practice.

### Propagation Delay

This is the time required to propagate from the beginning of the link to the end.

The propagation speed depends on the physical medium of the link (that is, fiber optics, twisted-pair copper wire, and so on) and is equal to, or a little less than the speed of light.

In wide-area networks, propagation delays are on the order of milliseconds.

## How

### How to use traceroute to determine the nodal delay

Some routers block UDP packets, can change to another type of probe. Here `-I` is for ICMP.

```shell
sudo traceroute -I www.baidu.com
```
