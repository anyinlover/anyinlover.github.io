# UDP

## What

[User Datagram Protocol](https://datatracker.ietf.org/doc/html/rfc768) is a [[transport_layer]] protocol that provides an unreliable connectionless service to the invoking application.

It provides:

1. process-to-process data delivery
2. error checking

Typical Application based on UDP:

- HTTP3
- NFS
- SNMP
- DNS

Some applications are better suited for UDP:

1. Finer application-level control over what data is sent, and when
2. No connection establishment
3. No connection state
4. Small packet header overhead

![UDP segment structure](../../assets/image/UDP%20segment%20structure.png)

![UDP wireshark](../../assets/image/UDP%20wireshark.png)

## Why

### Connectionless multiplexing and demultiplexing

a UDP socket is fully identified by a two-tuple consisting of a destination IP address and a destination port number.

### Error checking

The UDP checksum provides for error detection.

UDP at the sender side performs the 1s complement of the sum of all the 16-bit words in the segment, with any overflow encountered during the sum being wrapped around.

If no errors are introduced into the packet, then clearly the sum at the receiver will be 1111111111111111. If one of the bits is a 0, then we know that errors have been introduced into the packet.

In truth, the checksum is also calculated over a few of the fields in the IP header in addition to the UDP segment.

More details are in [RFC1071](https://datatracker.ietf.org/doc/html/rfc1071)

## How
