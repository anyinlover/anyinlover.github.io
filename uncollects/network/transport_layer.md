# Transport Layer

## What

A transport-layer protocol provides for logical communication between application processes.

It extending the network layer’s delivery service between two end systems to a delivery service between two application-layer processes running on the end systems.

Two problem the transport protocols may consider:

1. how two entities can communicate reliably over a medium that may lose and corrupt data.
2. how to control the transmission rate of transport-layer entities in order to avoid, or recover from, congestion within the network.

Two of Internet transport protocols are UDP and TCP.

## Why

### transport-layer multiplexing and demultiplexing

![transport-layer multiplexing and demultiplexing](../../assets/image/transport-layer%20multiplexing%20and%20demultiplexing.png)

1. Sockets have unique identifiers
2. Each segment have special fields that indicate the socket to which the segment is to be delivered.

Each port number is a 16-bit number, ranging from 0 to 65535.

The port numbers ranging from 0 to 1023 are called [well-known port numbers](http://www.iana.org).

Each socket in the host could be assigned a port number, and when a segment arrives at the host, the transport layer examines the destination port number in the segment and directs the segment to the corresponding socket. The segment’s data then passes through the socket into the attached process.

### Principles of Reliable Data Transfer

![GBN sender](../../assets/image/gbn%20sender.png)

![GBN receiver](../../assets/image/gbn%20receiver.png)

| Mechanism               | Use, Comments                                                                                                                                        |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Checksum                | Used to detect bit errors in a tramsmitted packet                                                                                                    |
| Timer                   | Used to timeout/retransmit a packet, possibly because the packet (or its ACK) was lost within the channel.                                           |
| Sequence number         | Used for sequential numbering of packets of data flowing from sender to receiver.                                                                    |
| Acknowledgment          | Used by the receiver to tell the sender that a packet or set of packets has been received correctly.                                                 |
| Negative acknowledgment | Used by the receiver to tell the sender that a packet has not been received correctly.                                                               |
| Window, pipelining      | By allowing multiple packets to be transmitted but not yet acknowledged, sender utilization can be increased over a stop-and-wait mode of operation. |

## How
