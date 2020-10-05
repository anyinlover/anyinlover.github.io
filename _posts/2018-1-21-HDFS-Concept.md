---
category: 笔记
tags:
  - Hadoop
  - Hive
---

distributed filesystems: filesystems that manage the storage across a network of machines.

HDFS: Hadoop Distributed filesystem

*Hadoop can integrate with other storage systems such as the local filesystem and Amazon S3*

## The Desion of HDFS

A filesystem designed for storing **very large files** with **Streaming data access** patterns, running on clusters of **commodity hardware**.

HDFS is not good for **Low-latency data access**, **Lots of small files** and **Multiple writers, arbitrary file modifications**.

## HDFS Concepts

### Blocks

A disk has a block size: the minimum amount of data that it can read or write.

Filesystem blocks are typically a few kilobytes in size, disk blocks are normally 512 bytes.

Tools to perform filesystem maintenance such as `df` and `fsck` operate on the filesystem block level.

HDFS block: **128MB* by default.

*Why so large? To minimize the cost of seeks*

Three benefits for having a block abstraction:

* A file can be larger than any single disk in the network.
* Simplify storage management and eliminate metadata concerns.
* Provide fault tolerence and availability.

HDFS also have a `fsck` command just like disk filesystem.

### Namenodes and Datanodes

An HDFS cluster has two types of nodes operating in a master-worker pattern: a namenode and a number of datanodes.

The namenode manages the filesystem namespace. **It maintains the filesystem tree and the metadata for all the files and directories in the tree**. It's stored persistently on the local disk in the form of two files: **the namespace image and the edit log**. It also knows the block locations, but not persistently.

A client access the filesystem on behalf of the user by communicating with the namenode and datanodes.

Datanodes are the workhorses of the filesystem. They **store and retrieve blocks** when they are told to and **report back to the namenode periodically** with lists of blocks that they are storing.

Without the namenode, the filesystem cannot be used. So it's important to make the namenode resilient to failure.

Two mechanisms:

* Back up the files that make up the persistent state of the filesystem metadata. Synchronous and atomic. Local disk as well as a remote NFS mount.
* Run a secondary namenode. The main role is to periodically merge the namespace image with the edit log to prevent the edit log from becoming too large. But the state of the secondary namenode lags that of the primary, so it can have data loss.

### Block Caching

Normally a datanode reads blocks from disk, but for frequently accessed files the blocks may be explicitly cached in the datanode's memory: block cache.

Users or applications instruct the namenode which files to cache by adding a cache directive to a cache pool.

### HDFS Federation

The namenode keeps a reference to every file and block in the filesystem in memory. So memory becomes the limiting factor for scaling to a large cluster. HDFS federation allow several namenodes, each of which manages a portion of the filesystem namespace.

### HDFS High Availability

The combination of replicating namenode metadata on multiple filesystems and using the secondary namenode to create checkpoints protects against data loss, but the namenode is still a single point of failure (SPOF).

To Recover from a failed namenode, an administrator should starts a new primary namenode with one of the filesystem metadata replicas and configures datanodes and clients to use this new namenode. It need three steps to serve requests:

1. Loaded its namespace image into memory
2. Replayed its edit log
3. Received enough block reports from the datanodes to leave safe mode.

Hadoop 2 add support for HDFS high availability (HA). There are a pair of namenodes in an active-standby configuration. It need a few architectural changes:

* the namenodes must use highly available shared storage to share the edit log.
* Datanodes must send block reports to both namenodes.
* Clients must be configured to handle namenode failover.
* The secondary namenode's role is subsumed by the standby, which takes periodic checkpoints of the active namenode's namespace.

Two choices for the highly available shared storage: an NFS filer, or a quorum journal manager (QJM). QJM is designed for the sole purpose of providing a highly available edit log. The QJM runs as a group of journal nodes, and each edit must be written to a majority of the journal nodes.

#### Failover and fencing

failover controller: the entity manage the transition from the active namenode to the standby.

The default implementation uses ZooKeeper to ensure that only one namenode is active. Each namenode runs a lightweight failover controller process to monitor its namenode for failures (using a simple heartbeating mechanism) and trigger a failover.

graceful failover: initiate failover manually

fencing: to make sure the previously active namenode is prevented from doing any damage and causing corruption.

* Revoke the namenode's access to the shared storage directory.
* Disable its network port
* STONITH: shoot the other node in the head

## The Command-Line Interface

~~~Shell
hadoop fs -copyFromLocal input/docs/quangle.txt hdfs://localhost/user/tom/quangle.txt
hadoop fs -copyFromLocal input/docs/quangle.txt quangle.txt
hadoop fs -ls
~~~

HDFS has a permissions model that is much like the POSIX model. But the execute permission is ignored for a file.
