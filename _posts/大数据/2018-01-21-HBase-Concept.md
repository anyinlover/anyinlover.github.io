---
title: HBase学习笔记
category: 大数据
tags:
  - Hadoop
  - HBase
---

HBase is a distributed column-oriented database built on top of HDFS.

HBase is not relational and does not support SQL.

## Concepts

### Whirlwind Tour of the Data Model

Table cells are versioned. The version is a timestamp auto-assigned by HBase at the time of cell insertion by default.

A cell's content is an uninterpreted array of bytes.

The HBase data model:

![hbase-model](../../img/hbase-model.png)

Table rows are sorted by row key, aka the table's primary key.

Row columns are grouped into column families. All column family members have a common prefix. The prefix must be composed of printable characters. The column family and the qualifier are separated by a colon character.

A table's column families must be specified up front as part of the table schema definition, but new column family members can be added on demand.

Physically, all column family members are stored together on the filesystem.

#### Regions

Tables are automatically portioned horizontally by HBase into regions. Regions are the units that get distributed over an HBase cluster.

#### Locking

Row updates are atomic.

### Implementation

Base is made up of an HBase master node orchestrating a cluster of one or more region server workers.

![hbase-cluster](../../img/hbase-cluster.png)

The HBase master:

- Bootstrap a virgin install
- Assign regions to registered regionservers
- Recover regionserver failures

The regionserver:

- Carry regions and manage region splits
- field client read/write requests
- Inform the HBase master about the new daughter regions.

HBase depends on ZooKeeper. By default it manages a ZooKeeper instance as the authority on cluster state.The ZooKeeper ensemble bosts vitals such as the location of the `hbase:meta` catalog table and the address of the current cluster master.

Regionserver worker nodes are listed in the HBase `cons/regionservers` file.

#### HBase in operation

HBase keeps a special catalog table named `hbase:meta`, within which it maintains the current list, state, and locations of all user-space regions afloat on the cluster. Entries in it are keyed by region name, which is made up of the name of the table, the region's start row, its time of creation, and an MD5 hash of all of these.

Fresh clients connect to the ZooKeeper cluster first to learn the location of `hbase:meta`, then to figure out the hosting user-space region and its location. Then it interacts directly with the hosting regionserver.

The clients caches all they learn while doing lookups for `hbase:meta`.

Writes arriving at a regionserver are first appened to a commit log and then added to an in-memory memstore. When a memstore fills, its content is flushed to the filesystem.

The commit log is hosted on HDFS. When the master notices that a regionserver is no longer reachable, it splits the dead regionserver's commit log by region.

When reading, the region's memstore is consulted first. Otherwise, flush files are consulted in order, from newest to oldest.

A backgroud process compacts flush files once the number has exceeded a threshold. The process cleans out versions beyond the schema-configured maximum and removes deleted and expired cells.

A separe process monitors flush file sizes, splitting the region when it excess the maximum.

## Clients

There are a number of client options for interacting with an HBase cluster:

- Java
- MapReduce
- REST and Thrift

## HBase Versus RDBMS

HBase has the following characteristics:

- No real indexes
- Automatic partioning
- Scale linearly and automatically with new nodes
- Commodity hardware
- Fault tolerance
- Batch processing

## Praxis

### HDFS

HBase's use of HDFS is very different from how it's used by MapReduce. In MapReduce, HDFS files are opened with their content streamed through a map task and then closed. In HBase, datafiles are opened on cluster startup and kept open.

HBase tends to see the following issues:

- Running out of file descriptors
- Running out of datanode threads

### UI

HBase runs a web server on the master to present a view on the state of your running cluster.

### Metrics

Hadoop has a metrics system that can be used to emit vitals over a period to a context.
