---
title: Hive学习笔记
category: 大数据
tags:
  - Hadoop
  - Hive
---

Hive is used as a general-purpose, scalable data processing platform.

## Installing Hive

Hive converts your SQL query into a series of jobs for execution on a Hadoop cluster. Hive organises data into tables, which provide a means for attaching structure to data stored in HDFS. Metadata is stored in a database called the metastore.

### The Hive Shell

The shell is the primary way we interact with Hive, by issuing commands in HiveQL. HiveQL is Hive's query language and is heavily influenced by MySQL.

It is the ability to execute SQL queries against our raw data that gives Hive its power.

## Running Hive

### Configuring Hive

Hive is configured using an XML configuration file `hive-site.xml`.

You can specify the filesystem and resource manager and metastore configuration in the configuration file.

Hive also permits you to set properties on a per-session basis, by passing the `-hive-conf` option to the `hive` command.

Settings can be changed within a seesion, using the `SET` command.

The precedence hierarchy to setting properties:

1. The Hive `SET` command
2. The command-line `-hiveconf` option
3. `hive-site.xml` and the Hadoop site files
4. The Hive defaults and Hadoop defaults files

#### Execution engines

Hive can use MapReduce, Tez or Spark as execution engine.

It's controlled by the `hive.execution.engine` property.

#### Logging

Hive's error log are on the local filesystem. MapReduce task logs are also useful.

The logging configuration is in `conf/hive-log4j.properties`, the file can be edited to change log levels and other logging-related settings. It's more convenient to set logging configuration for the session:

```shell
hive -hiveconf hive.root.logger=DEBUG, console
```

### Hive Services

You can specify the service to run using the `—service`option. Following are the most useful ones:

- cli
- hiveserver2
- beeline
- hwi
- jar
- metastore

#### Hive clients

If you run Hive as a server (`hive —service hiveserver2`), there are a number of different mechanisms for connecting to it from applications:

- Thrift Client
- JDBC driver
- ODBC driver

Hive architecture:

![hive-architecture](../../img/hive-architecture.png)

### The Metastore

The metastore is the central repository of Hive metadata. The metastore is divided into two pieces: a service and the backing store for the data.

Eembedd metastore: By default, the megastore service runs in the same JVM as the Hive service and contains an embedded Derby database instance.

Local metastore: Use a standalone database to support multiple sessions.

Remote metastore: Metastore servers run in separate processes to the Hive service.

![metastore-configurations](../../img/metastore-configurations.png)

## Comparison with Traditional Databases

### Schema on Reading Versus Schema on Write

Schema on write: A traditional database checks data against the schema when it written into the database.

Schema on read: Hive doesn't verify the data when it is loaded, but rather when a query is issued.

Schema on read makes for a very fast initial load, and schema on write makes query time performance faster.

### Updates, Transactions, and Indexes

HDFS does not provide in-place file updates, so changes resulting from inserts, updates, and deletes are stored in small delta files.

Hive has support for table- and partition-level locking.

Hive indexes can speed up queries in certain cases. There are currently two index types:

- Compact index: store the HDFS block numbers of each value
- Bitmap index: use compressed bitsets

### SQL-on-Hadoop Alternatives

- Cloudera Impala
- Presto from Facebook
- Apache Drill
- Spark SQL
- Apache Phoenix

## HiveQL

HiveQL is a mixture of SQL-92, MySQL and Oracle's SQL dialect.

## Tables

A Hive table is logically made up of the data being stored and the associated metadata describing the layout of the data in the table.

Hive supports the facility for multiple namespaces.
