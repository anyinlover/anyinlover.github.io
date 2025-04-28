---
title: Case Studies of Multicore Architectures
description: Case studies of multicore architectures
sidebar_position: 11
---

## Case Studies of Multicore Architectures I

The objectives of this module are to discuss about the need for multicore processors and discuss the salient features of the Intel multicore architectures as a case study.

In many of the previous modules, we have looked et different ways of exploiting ILP. ILP is parallelism at the machine -instruction level, through which the processor can re-order, pipeline instructions, use predication, do aggressive branch prediction, etc. ILP had enabled rapid increases in processor speeds for two decades or so. During this period, the transistor densities doubled every 18 months (Moore’s Law) and the frequencies increased making the pipelines deeper and deeper. Figure 37.1 shows the trend that happened. But deeply pipelined circuits had to do deal with the heat generated, due to the rise in temperatures. The design of such complicated processors was difficult and there was also diminishing returns with increasing frequencies. All these made it difficult to make single-core clock frequencies even higher. The power densities increased so rapidly that they had to be somehow contained within limits. Figure 37.2 shows the increase in power densities.

![](img/Case%20Studies%20of%20Multicore%20Architectures%20I%20%E2%80%93%20Computer%20Architecture2-191.png)

![](img/Case%20Studies%20of%20Multicore%20Architectures%20I%20%E2%80%93%20Computer%20Architecture2-192.png)

Additionally, from the application point of view also, many new applications are multithreaded. Single-core superscalar processors cannot fully exploit TLP. So, the general trend in computer architecture is to shift towards more parallelism. There has been a paradigm shift from complicated single core architectures to simpler multicore architectures. This is parallelism on a coarser scale – explicitly exploiting TLP.

Multi-core processors take advantage of a fundamental relationship between power and frequency. By incorporating multiple cores, each core is able to run at a lower frequency, dividing among them the power normally given to a single core. The result is a big performance increase over a single-core processor. The following illustration in Figures 37.3 and 37.4 shows this key advantage.

![](img/Case%20Studies%20of%20Multicore%20Architectures%20I%20%E2%80%93%20Computer%20Architecture2-193.png)

As shown in Figure 37.3, i ncreasing the clock frequency by 20 percent to a single core delivers a 13 percent performance gain, but requires 73 percent greater power. Conversely, decreasing clock frequency by 20 percent reduces power usage by 49 percent, but results in just a 13 percent performance loss.

![](img/Case%20Studies%20of%20Multicore%20Architectures%20I%20%E2%80%93%20Computer%20Architecture2-194.png)

A second core is added to the under-clocked example above. This results in a dual-core processor that at 20 percent reduced clock frequency, effectively delivers 73 percent more performance while using approximately the same power as a single-core processor at maximum frequency. This is shown in Figure 37.4.

A multicore processor is a special kind of a multiprocessor, where all processors are on the same chip. Multicore processors fall under the category of MIMD – different cores execute different threads (Multiple Instructions), operating on different parts of memory (Multiple Data). Multicore is a shared memory multiprocessor, since all cores share the same memory. Therefore, all the issues that we have discussed with MIMD style of shared memory architectures hold good for these processors too.

Multicore architectures can be *homogeneous or heterogeneous*. Homogeneous processors are those that have identical cores on the chip. Heterogeneous processors have different types of cores on the chip. The software technologists will need to understand the architecture to get appropriate performance gains. With dual core & multicore processors ruling the entire computer industry, some of the architectural issues would be different than the traditional issues . For example, the new performance mantra would be performance per watt.

We shall discuss different multicore architectures as case studies. The following architectures will be discussed:

•      Intel

•      Sun’s Niagara

•      IBM’s Cell BE

Of these, this module will deal with the salient features of the Intel’s multicore architectures.

**Intel’s Multicore Architectures:** Like all other partners in the computer industry, Intel also acknowledged that it had hit a ”thermal wall‘’ in 2004 and disbanded one of its most advanced design groups and said it would abandon two advanced chip development projects. Now, Intel is embarked on a course already adopted by some of its major rivals: obtaining more computing power by stamping multiple processors on a single chip rather than straining to increase the speed of a single processor. The evolution of the Intel processors from 1978 to the introduction of multicore processors is shown in Figure 37.5.

![](img/Case%20Studies%20of%20Multicore%20Architectures%20I%20%E2%80%93%20Computer%20Architecture2-195.png)

The following are the salient features of the Intel’s multicore architectures:

•      Hyper-threading

•      Turbo boost technology

•      Improved cache latency with smart L3 cache

•      New Platform Architecture

•      Quick Path Interconnect (QPI)

•      Intelligent Power Technology

•      Higher Data-Throughput via PCI Express 2.0 and DDR3 Memory Interface

•      Improved Virtualization Performance

•      Remote Management of Networked Systems with Intel Active Management Technology

•      Other improvements – Increase in window size, better branch prediction, more instructions and accelerators, etc.

We shall discuss each one of them in detail.

**Optimized Multithreaded Performance through Hyper-Threading:** Intel introduced Hyper-Threading Technology on its processors in 2002. Hyper -threading exposes a single physical processing core as two logical cores to allow them to share resources between execution threads and therefore increase the system efficiency, as shown in Figure 37.6. Because of the lack of OSs that could clearly differentiate between logical and physical processing cores, Intel removed this feature when it introduced multicore CPUs. With the release of OSs such as Windows Vista and Windows 7, which are fully aware of the differences between logical and physical core, Intel brought back the hyper-threading feature in the Core i7 family of processors. Hyper-threading allows simultaneous execution of two execution threads on the same physical CPU core. With four cores, there are eight threads of execution. Hyper-threading technology benefits from larger caches and increased memory bandwidth of the Core i7 processors, delivering greater throughput and responsiveness for multithreaded applications.

![](img/Case%20Studies%20of%20Multicore%20Architectures%20I%20%E2%80%93%20Computer%20Architecture2-196.png)

**CPU Performance Boost via Intel Turbo Boost Technology:** The applications that are run on multicore processors can be single threaded or multi -threaded. In order to provide a performance boost for lightly threaded applications and to also optimize the processor power consumption, Intel introduced a new feature called Intel Turbo Boost. Intel Turbo Boost features offer processing performance gains for all applications regardless of the number of execution threads created. It automatically allows active processor cores to run faster than the base operating frequency when certain conditions are met. This mode is activated when the OS requests the highest processor performance state. The maximum frequency of the specific processing core on the Core i7 processor is dependent on the number of active cores, and the amount of time the processor spends in the Turbo Boost state depends on the workload and operating environment.

![](img/Case%20Studies%20of%20Multicore%20Architectures%20I%20%E2%80%93%20Computer%20Architecture2-197.png)

Figure 37.7 illustrates how the operating frequencies of the processing cores in the quad-core Core i7 processor change to offer the best performance for a specific workload type. In an idle state, all four cores operate at their base clock frequency. If an application that creates four discrete execution threads is initiated, then all four processing cores start operating at the quad-core turbo frequency. If the application creates only two execution threads, then two idle cores are put in a low-power state and their power is diverted to the two active cores to allow them to run at an even higher clock frequency. Similar behavior would apply in the case where the applications generate only a single execution thread.

For example, the Intel Core i7-820QM quad-core processor has a base clock frequency of 1.73 GHz. If the application is using only one CPU core, Turbo Boost technology automatically increases the clock frequency of the active CPU core on the Intel Core i7-820QM processor from 1.73 GHz to up to 3.06 GHz and places the other three cores in an idle state, thereby providing optimal performance for all application types. This is shown in the table below.

![](img/Case%20Studies%20of%20Multicore%20Architectures%20I%20%E2%80%93%20Computer%20Architecture2-198.png)

The duration of time that the processor spends in a specific Turbo Boost state depends on how soon it reaches thermal, power, and current thresholds. With adequate power supply and heat dissipation solutions, a Core i7 processor can be made to operate in the Turbo Boost state for an extended duration of time. In some cases, the users can manually control the number of active processor cores through the controller’s BIOS to fine tune the operation of the Turbo Boost feature for optimizing performance for specific application types.

**Improved Cache Latency with Smart L3 Cache:** All of us are aware of the fact that the cache is a block of high-speed memory for temporary data storage located on the same silicon die as the CPU. If a single processing core, in a multicore CPU, requires specific data while executing an instruction set, it first searches for the data in its local caches (L1 and L2). If the data is not available, also known as a cache-miss, it then accesses the larger L3 cache. In an exclusive L3 cache, if that attempt is unsuccessful, then the core performs cache snooping – searches the local caches of other cores – to check whether they have data that it needs. If this attempt also results in a cache-miss, it then accesses the slower system RAM for that information. The latency of reading and writing from the cache is much lower than that from the system RAM, therefore a smarter and larger cache greatly helps in improving processor performance.

The Core i7 family of processors features an inclusive s hared L3 cache that can be up to 12 MB in size. Figure 37.8 shows the different types of caches and their layout for the Core i7-820QM quad-core processor. It features four cores, where each core has 32 kilobytes for instructions and 32 kilobytes for data of L1 cache, 256 kilobytes per core of L2 cache, along with 8 megabytes of shared L3 cache. The L3 cache is shared across all cores and its inclusive nature helps increase performance and reduces latency by reducing cache snooping traffic to the processor cores. An inclusive shared L3 cache guarantees that if there is a cache-miss, then the data is outside the processor and not available in the local caches of other cores, which eliminates unnecessary cache snooping. This feature provides improvement for the overall performance of the processor and is beneficial for a variety of applications including test, measurement, and control.

![](img/Case%20Studies%20of%20Multicore%20Architectures%20I%20%E2%80%93%20Computer%20Architecture2-199.png)

**New Platform Architecture:** As shown in Figure 37.9, the previous Intel micro-architectures for a single processor system included three discrete components – a CPU, a Graphics and Memory Controller Hub (GMCH), also known as the Northbridge and an I/O Controller Hub (ICH), also known as the Southbridge. The GMCH and ICH combined together are referred to as the chipset. In the older Penryn architecture, the front-side bus (FSB) was the interface for exchanging data between the CPU and the northbridge. If the CPU had to read or write data into system memory or over the PCI Express bus, then the data had to traverse over the external FSB. In the new Nehalem micro-architecture, Intel moved the memory controller and PCI Express controller from the northbridge onto the CPU die, reducing the number of external data buses that the data had to traverse. These changes help increase data-throughput and reduce the latency for memory and PCI Express data transactions. These improvements make the Core i7 family of processors ideal for test and measurement applications such as high-speed design validation and high-speed data record and playback.

![](img/Case%20Studies%20of%20Multicore%20Architectures%20I%20%E2%80%93%20Computer%20Architecture2-200.png)

**Higher – Performance Multiprocessor Systems with Quick Path Interconnect (QPI):** Not only was the memory controller moved to the CPU for Nehalem processors, Intel also introduced a distributed shared memory architecture using Intel QuickPath Interconnect (QPI). QPI is the new point-to-point interconnect for connecting a CPU to either a chipset or another CPU. It provides up to 25.6 GB/s of total bidirectional data throughput per link. Intel’s decision to move the memory controller in the CPU and introduce the new QPI data bus has had an impact for single-processor systems. However, this impact is much more significant for multiprocessor systems. Figure 37.10 illustrates the typical block diagrams of multiprocessor systems based on the previous generation and the Nehalem microarchitecture.

![](img/Case%20Studies%20of%20Multicore%20Architectures%20I%20%E2%80%93%20Computer%20Architecture2-201.png)

Each CPU has access to local memory but they also can access memory that is local to other CPUs via QPI transactions. For example, one Core i7 processor can access the memory region local to another processor through QPI either with one direct hop or through multiple hops. QPI is a high-speed, point-to-point interconnect. Figure 37.11 illustrates this. It provides high bandwidth and low latency, which delivers the interconnect performance needed to unleash the new microarchitecture and deliver the Reliability, Availability, and Serviceability (RAS) features expected in enterprise applications. RAS requirements are met through advanced features which include: CRC error detection, link-level retry for error recovery, hot-plug support, clock fail-over, and link self-healing. Detection of clock failure & automatic readjustment of the width & clock being transmitted on a predetermined dataline increases availability.

It supports simultaneous movement of data between various components. Support for several features such as lane/polarity reversal, data recovery and deskew circuits, and waveform equalization that ease the design of the high-speed link, are provided. The Intel® QuickPath Interconnect includes a cache coherency protocol to keep the distributed memory and caching structures coherent during system operation. It supports both low-latency source snooping and a scalable home snoop behavior. The coherency protocol provides for direct cache-to-cache transfers for optimal latency.

![](img/Case%20Studies%20of%20Multicore%20Architectures%20I%20%E2%80%93%20Computer%20Architecture2-202.png)

The QPI supports a layered architecture as shown in Figure 37.12. The Physical layer consists of the actual wires, as well as circuitry and logic to support ancillary features required in the transmission and receipt of the 1s and 0s – unit of transfer is 20-bits, called a Phit (for Physical unit). • The Link layer is responsible for reliable transmission and flow control – unit of transfer is an 80-bit Flit (for Flow control unit). The Routing layer provides the framework for directing packets through the fabric. The Transport layer is an architecturally defined layer (not implemented in the initial products) providing advanced routing capability for reliable end-to-end transmission. The Protocol layer is the high-level set of rules for exchanging packets of data between devices. A packet is comprised of an integral number of Flits.

![](img/Case%20Studies%20of%20Multicore%20Architectures%20I%20%E2%80%93%20Computer%20Architecture2-203.png)

**Intelligent Power Technology:** Intel’s multicore processors provide intelligent power gates that allow the idling processor cores to near zero power, thus reducing the power consumption. Power consumption is reduced to 10 watts compared to 16 to 50 watts earlier. There is support provided for automated low power states to put the processor and memory in the lowest power state allowable in a workload.

**Higher Data-Throughput via PCI Express 2.0 and DDR3 Memory Interface:** To support the need of modern applications to move data at a faster rate, the Core i7 processors offer increased throughput for the external databus and its memory channels. The new processors feature the PCI Express 2.0 databus, which doubles the data throughput from PCI Express 1.0 while maintaining full hardware and software compatibility with PCI Express 1.0. A x16 PCI Express 2.0 link has a maximum throughput of 8 GB/s/direction. To allow data from the PCI Express 2.0 databus to be stored in system RAM, the Core i7 processors feature multiple DDR3 1333 MHz memory channels. A system with two channels of DDR3 1333 MHz RAM had a theoretical memory bandwidth of 21.3 GB/s. This throughput matches well with the theoretical maximum throughput of a x16 PCI Express 2.0 link.

**Improved Virtualization Performance:** Virtualization is a technology that enables running multiple OSs side-by-side on the same processing hardware. In the test, measurement, and control space, engineers and scientists have used this technology to consolidate discrete computing nodes into a single system. With the Nehalem mircoarchitecture, Intel has added new features such as hardware-assisted page-table management and directed I/O in the Core i7 processors and its chipsets that allow software to further improve their performance in virtualized environments.

**Remote Management of Networked Systems with Intel Active Management Technology (AMT):** AMT provides system administrators the ability to remotely monitor, maintain, and update systems. Intel AMT is part of the Intel Management Engine, which is built into the chipset of a Nehalem-based system. This feature allows administrators to boot systems from a remote media, track hardware and software assets, and perform remote troubleshooting and recovery. Engineers can use this feature for managing deployed automated test or control systems that need high uptime. Test, measurement, and control applications are able to use AMT to perform remote data collection and monitor application status. When an application or system failure occurs, AMT enables the user to remotely diagnose the problem and access debug screens. This allows for the problem to be resolved sooner and no longer requires interaction with the actual system. When software updates are required, AMT allows for these to be done remotely, ensuring that the system is updated as quickly as possible since downtime can be very costly. AMT is able to provide many remote management benefits for PXI systems.

**Other Improvements:** Apart from the features discussed above, there are also other improvements brought in. They are listed below:

• Greater Parallelism

1.   Increase amount of code that can be executed out-of-order by increasing the size of the window and scheduler

• Efficient algorithms

1.  Improved performance of    synchronization primitives
2.  Faster handling of branch mis-prediction
3.  Improved hardware prefetch & better Load-Store scheduling

• Enhanced Branch Prediction

1.  New Second-Level Branch Target Buffer(BTB)
2.  New Renamed Return Stack Buffer (RSB)

• New Application Targeted Accelerators and Intel SSE4 (Streaming SIMD extensions-4)

1.  Enable XML parsing/accelerators

To summarize, we have dealt in detail the salient features of intel’s multicore architectures in this module. The following features were discussed:

-   Hyper-threading
-   Turbo boost technology
-   Improved cache latency with smart L3 cache
-   New Platform Architecture
-   Quick Path Interconnect (QPI)
-   Intelligent Power Technology
-   Higher Data-Throughput via PCI Express 2.0 and DDR3 Memory Interface
-   Improved Virtualization Performance
-   Remote Management of Networked Systems with Intel Active Management Technology
-   Other improvements

**Web Links / Supporting Materials**

-   Computer Architecture – A Quantitative Approach , John L. Hennessy and David A. Patterson, 5th  Edition, Morgan Kaufmann, Elsevier, 2011.
-   Intel White Papers:
-   [https://www.pogolinux.com/learn/files/quad-core-06.pdf](https://www.pogolinux.com/learn/files/quad-core-06.pdf)
-   Intel Tech/Research: [http://www.intel.com/technology/index.htm](http://www.intel.com/technology/index.htm)
-   Energy Efficient Performance: [http://www.intel.com/technology/eep/index.htm](http://www.intel.com/technology/eep/index.htm)
-   Intel Core Microarchitecture: [http://www.intel.com/technology/architecture/coremicro/](http://www.intel.com/technology/architecture/coremicro/)
-   Dual-core processor: [http://www.intel.com/technology/computing/dual-core/index.htm](http://www.intel.com/technology/computing/dual-core/index.htm)
-   Multi/Many Core: [http://www.intel.com/multi-core/index.htm](http://www.intel.com/multi-core/index.htm)
-   Intel Platforms: [http://www.intel.com/platforms/index.htm](http://www.intel.com/platforms/index.htm)
-   Threading: [http://www3.intel.com/cd/ids/developer/asmo-na/eng/dc/threading/index.htm](http://www3.intel.com/cd/ids/developer/asmo-na/eng/dc/threading/index.htm)
-   Top Eight Features of the Intel Core i7 Processors for Test, Measurement, and Control:
-   http://www.ni.com/white-paper/11266/en/#toc1

## Case Studies of Multicore Architectures II

The objectives of this module are to discuss about the salient features of two different styles of Multicore architectures viz. Sun’s Niagara and IBM’s Cell Broadband Engine.

The previous module discussed the need for multicore processors and discussed the salient features of the Intel multicore architectures as a case study. This module will discuss two more multicore architectures as case studies. We shall first discuss Sun’s multicore architecture.

Computer designers initially focused on improving the performance of a single thread in a desktop processing environment by increasing the frequencies and exploiting instruction level parallelism using the different techniques that we have discussed earlier. However, the emphasis on single-thread performance has shown diminishing returns because of the limitations in terms of latency to main memory and the inherently low ILP of applications. This has led to an explosion in microprocessor design complexity and made power dissipation a major concern. For these reasons, Sun Microsystems’ processors take a radically different approach to microprocessor design. Instead of focusing on the performance of single or dual threads, Sun optimized the architecture for multithreaded performance in a commercial server environment like web services and data bases. With multiple threads of execution, it is possible to find something to execute every cycle. Significant throughput speedups are thus possible and processor utilization is much higher.

There are basically two processors that were introduced by Sun Microsystems in the UltraSPARC T line – The UltraSPARC T1 called Niagara introduced in 2005 using the 90nm technology and its successor, the UltraSPARC T2 codenamed Niagara2, introduced in 2007 with the 65nm technology. We shall discuss the salient features of these architectures in detail.

Niagara supports 32 hardware threads by combining ideas from chip multiprocessors and fine-grained multi-threading. The parallel execution of many threads effectively hides memory latency. However, having 32 threads places a heavy demand on the memory system to support high bandwidth. To provide this bandwidth, a crossbar interconnects scheme routes memory references to a banked on-chip level-2 cache that all threads share. Four independent on-chip memory controllers provide in excess of 20

Gbytes/s of bandwidth to memory. Figure 38.1 provides the salient features of the Niagara processor.

![](img/Case%20Studies%20of%20Multicore%20Architectures%20II%20%E2%80%93%20Computer%20Architecture2-204.png)

Niagara supports 32 threads of execution in hardware. The architecture organizes four threads into a thread group. The group shares a processing pipeline, referred to as the SPARC pipe. Niagara uses eight such thread groups, resulting in 32 threads on the CPU. Each SPARC pipe contains level-1 caches for instructions and data. The hardware hides memory and pipeline stalls on a given thread by scheduling the other threads in the group onto the SPARC pipe with a zero cycle switch penalty. The 32 threads share a 3-Mbyte level-2 cache. This cache is 4-way banked and pipelined. It is 12-way set-associative to minimize conflict misses from the many threads. The crossbar interconnect provides the communication link between SPARC pipes, L2 cache banks, and other shared resources on the CPU. It provides more than 200 Gbytes/s of bandwidth. A two-entry queue is available for each source-destination pair, and it can queue up to 96 transactions each way in the crossbar. The crossbar also provides a port for communication with the I/O subsystem. The crossbar is also the point of memory ordering for the machine. The memory interface is four channels of dual-data rate 2 (DDR2) DRAM, supporting a maximum bandwidth in excess of 20 Gbytes/s, and a capacity of up to 128 Gbytes. Figure 38.2 shows a block diagram of the Niagara processor.

![](img/Case%20Studies%20of%20Multicore%20Architectures%20II%20%E2%80%93%20Computer%20Architecture2-205.png)

**SPARC Pipeline:** The SPARC pipeline is a simple and straight forward pipeline consisting of six stages. Figure 38.3 shows the SPARC pipeline. Each SPARC pipeline supports four threads. Each thread has a unique set of registers and instruction and store buffers. The thread group shares the L1 caches, translation look -aside buffers (TLBs), execution units, and most pipeline registers. The pipeline consists of the six stages – fetch, thread select, decode, execute, memory, and write back. In the fetch stage, the instruction cache and instruction TLB (ITLB) are accessed. The following stage completes the cache access by selecting the way. The critical path is set by the 64-entry, fully associative ITLB access. A thread-select multiplexer determines which of the four thread program counters (PC) should perform the access. The pipeline fetches two instructions each cycle. A predecode bit in the cache indicates long -latency instructions.

![](img/Case%20Studies%20of%20Multicore%20Architectures%20II%20%E2%80%93%20Computer%20Architecture2-206.png)

In the thread-select stage, the thread-select multiplexer chooses a thread from the available pool to issue an instruction into the downstream stages. This stage also maintains the instruction buffers. Instructions fetched from the cache in the fetch stage can be inserted into the instruction buffer for that thread if the downstream stages are not available. The thread-select logic decides which thread is active in a given cycle in the fetch and thread-select stages. As Figure 38.3 shows, the thread-select signals are common to fetch and thread-select stages. Therefore, if the thread -select stage chooses a thread to issue an instruction to the decode stage, the fetch stage also selects the same instruction to access the cache. The thread-select logic uses information from various pipeline stages to decide when to select or deselect a thread. For example, the thread-select stage can determine instruction type using a predecode bit in the instruction cache, while some traps are only detectable in later pipeline stages. Therefore, instruction type can cause deselection of a thread in the thread-select stage, while a late trap detected in the memory stage needs to flush a ll younger instructions from the thread and deselect itself during trap processing. Pipeline registers for the first two stages are replicated for each thread.

For single-cycle instructions such as ADD, Niagara implements full bypassing to younger instructions from the same thread to resolve RAW dependencies. Load instructions have a three-cycle latency before the results of the load are visible to the next instruction. Such long-latency instructions can cause pipeline hazards and resolving them requires stalling the corresponding thread until the hazard clears. So, in the case of a load, the next instruction from the same thread waits for two cycles for the hazards to clear.

In a multithreaded pipeline, threads competing for shared resources also encounter structural hazards. Resources such as the ALU that have a one-instruction-per-cycle throughput present no hazards, but the divider, which has a throughput of less than one instruction per cycle, presents a scheduling problem. In this case, any thread that must execute a DIV instruction has to wait until the divider is free. The thread scheduler guarantees fairness in accessing the divider by giving priority to the least recently executed thread. Although the divider is in use, other threads can use free resources such as the ALU, load-store unit, and so on.

The thread selection policy is to switch between available threads every cycle, giving priority to the least recently used thread. Threads can become unavailable because of long-latency instructions such as loads, branches, and multiply and divide. They also become unavailable because of pipeline stalls such as cache misses, traps, and resource conflicts. The thread scheduler assumes that loads are cache hits, and can therefore issue a dependent instruction from the same thread speculatively. However, such a speculative thread is assigned a lower priority for instruction issue as compared to a thread that can issue a non-speculative instruction.

Instructions from the selected thread go into the decode stage, which performs instruction decode and register file access. The supported execution units include an arithmetic logic unit (ALU), shifter, multiplier, and a divider. A bypass unit handles instruction results that must be passed to dependent instructions before the register file is updated. ALU and shift instructions have single cycle latency and generate results in the execute stage. Multiply and divide operations are long latency and cause a thread switch.

The load store unit contains the data TLB (DTLB), data cache, and store buffers. The DTLB and data cache access take place in the memory stage. Like the fetch stage, the critical path is set by the 64-entry, fully associative DTLB access. The load-store unit contains four 8-entry store buffers, one per thread. Checking the physical tags in the store buffer can indicate read after write (RAW) hazards between loads and stores. The store buffer supports the bypassing of data to a load to resolve RAW hazards. The store buffer tag check happens after the TLB access in the early part of write back stage.

Load data is available for bypass to dependent instructions late in the write back stage. Single-cycle instructions such as ADD will update the register file in the write back stage.

The L1 instruction cache is 16 Kbyte, 4-way set-associative with a block size of 32 bytes. A random replacement scheme is implemented for area savings, without incurring significant performance cost. The instruction cache fetches two instructions each cycle. If the second instruction is useful, the instruction cache has a free slot, which the pipeline can use to handle a line fill without stalling. The L1 data cache is 8 Kbytes, 4-way set-associative with a line size of 16 bytes, and implements a write-through policy. Even though the L1 caches are small, they significantly reduce the average memory access time per thread with miss rates in the range of 10 percent. Because commercial server applications tend to have large working sets, the L1 caches must be much larger to achieve significantly lower miss rates, so this sort of tradeoff is not favorable for area. However, the four threads in a thread group effectively hide the latencies from L1 and L2 misses. Therefore, the cache sizes are a good trade -off between miss rates, area, and the ability of other threads in the group to hide latency.

Niagara uses a simple cache coherence protocol. The L1 caches are write through, with allocate on load and no -allocate on stores. L1 lines are either in valid or invalid states. The L2 cache maintains a directory that shadows the L1 tags. The L2 cache also interleaves data across banks at a 64-byte granularity. A load that missed in an L1 cache (load miss) is delivered to the source bank of the L2 cache along with its replacement way from the L1 cache. There, the load miss address is entered in the corresponding L1 tag location of the directory, the L2 cache is accessed to get the missing line and data is then returned to the L1 cache. The directory thus maintains a sharers list at L1-line granularity. A subsequent store from a different or same L1 cache will look up the directory and queue up invalidates to the L1 caches that have the line. Stores do not update the local caches until they have updated the L2 cache. During this time, the store can pass data to the same thread but not to other threads; therefore, a store attains global visibility in the L2 cache. The crossbar establishes memory order between transactions from the same and different L2 banks, and guarantees delivery of transactions to L1 caches in the same order. The L2 cache follows a copy-back policy, writing back dirty evicts and dropping clean evicts. Direct memory accesses from I/O devices are ordered through the L2 cache. Four channels of DDR2 DRAM provide in excess of 20 Gbytes/s of memory bandwidth.

Each SPARC core has the following units:

•   Instruction fetch unit (IFU) includes the following pipeline stages – fetch, thread selection, and decode. The IFU also includes an instruction cache complex.

•  Execution unit (EXU) includes the execute stage of the pipeline.

   •   Load/store unit (LSU) includes memory and write-back stages, and a data cache complex.

•  Trap logic unit (TLU) includes trap logic and trap program counters.

•  Stream processing unit (SPU) is used for modular arithmetic functions for crypto.

•  Memory management unit (MMU).

•  Floating-point front end unit (FFU) interfaces to the FPU.

Thus, we see from the above details that the Niagara processor implements a thread-rich architecture designed to provide a high-performance solution for commercial server applications. The hardware supports 32 threads with a memory subsystem consisting of an on-board crossbar, level-2 cache, and memory controllers for a highly integrated design that exploits the thread-level parallelism inherent to server applications, while targeting low levels of power consumption.

Next, we shall discuss about IBM’s Cell Broad Band Engine (Cell BE) architecture.

**IBM’s Cell BE:** The Cell BE is a heterogeneous architecture developed by IBM, SCEI/Sony and Toshiba with the following design goals:

· An accelerator extension to Power o  Built on a Power ecosystem

o  Use best known system practices for processor design

· Setting a new performance standard

o   Exploit parallelism while achieving high frequency

o   Supercomputer attributes with extreme floating point capabilities

o   Sustain high memory bandwidth

· Design for natural human interaction

o   Photo-realistic effects

o   Predictable real-time response

o   Virtualized resources for concurrent activities

· Design for flexibility

o   Wide variety of application domains

o   Highly abstracted to highly exploitable programming models o Reconfigurable I/O interfaces

o  Virtual trusted computing environment for security

The Cell BE is a heterogeneous architecture because it has different types of cores. The Cell BE logically defines four separate types of functional components: the PowerPC Processor Element – one core (PPE), synergistic processor units – eight cores (SPU), the memory flow controller (MFC), and the internal interrupt controller (IIC). The computational units in the CBEA-compliant processor are the PPEs and the SPUs. Each SPU has a dedicated local storage, a dedicated MFC with its associated memory management unit (MMU), and a replacement management table (RMT). The combination of these components is called a Synergistic Processor Element (SPE). Figure 38.4 shows the architecture of Cell BE. The element interconnect bus (EIB) connects the various units within the processor.

![](img/Case%20Studies%20of%20Multicore%20Architectures%20II%20%E2%80%93%20Computer%20Architecture2-207.png)

**Power Processing Element:** The PPE is a 64 bit, “Power Architecture” processor, with a 32KB primary instruction cache, a 32KB primary data cache and a 512KB secondary cache. It is a dual issue processor that is dual threaded with in-order static issue. It acts as the control unit for the SPEs. It runs the OS and most of the applications but compute intensive parts of the OS and applications will be offloaded to the SPEs. It is a RISC architecture that uses considerably less power than other PowerPC devices, even at higher clock rates. The major functional units of the PPE are given in Figure 38.5. It is composed of three main units – the Instruction Unit (IU), which takes care of the fetch, decode, branch, issue and completion, the Fixed-Point Execution Unit (XU), which handles the fixed – point instructions and load/store instructions and the Vector Scalar Unit (VSU), which handles the vector and floating point instructions. The PPE pipeline is illustrated in Figure 38.6.

![](img/Case%20Studies%20of%20Multicore%20Architectures%20II%20%E2%80%93%20Computer%20Architecture2-208.png)

![](img/Case%20Studies%20of%20Multicore%20Architectures%20II%20%E2%80%93%20Computer%20Architecture2-209.png)

**Synergistic Processing Elements (SPEs):** The SPEs are eight in number. They follow SIMD style instruction set architecture and are optimized for power and performance of computational-intensive applications. The various functional units of the SPE are shown in Figure 38.7 and the SPE pipeline is indicated in Figure 38.8. An SPE can operate on sixteen 8-bit integers, eight 16-bit integers, four 32-bit integers, or four single-precision floating-point numbers in a single clock cycle, as well as a memory operation.

The SPEs have a local store memory for instructions and data. This 256KB local store is the largest component of the SPE and acts as an additional level of memory hierarchy. It is a single-port SRAM, capable of reading and writing through both narrow 128-bit and wide 128-byte ports. The SPEs operate on registers which are read from or written to the local stores. The local stores can access main memory in blocks of 1KB minimum (16KB maximum) but the SPEs cannot act directly on main memory (they can only move data to or from the local stores). Caches can deliver similar or even faster data rates but only in very short bursts (a couple of hundred cycles at best), the local stores can each deliver data at this rate continually for over ten thousand cycles without going to RAM.

![](img/Case%20Studies%20of%20Multicore%20Architectures%20II%20%E2%80%93%20Computer%20Architecture2-210.png)

Data and instructions are transferred between the local store and the system memory by asynchronous DMA commands, controlled by the Memory Flow Controller, which provides support for 16 simultaneous DMA commands. The SPE instructions insert DMA commands into queues which are then pooled as DMA commands into a single “DMA list” command. They support a 128 entry unified register file for improved memory bandwidth, and optimum power efficiency and performance .

The Element Interconnect Bus (EIB) provides the data ring for internal communication. You can form four 16 byte data rings of low latency, multiple simultaneous transfers at 96B/cycle peak bandwidth (at ½ CPU speed). The EIB is shown in Figure 38.9.

![](img/Case%20Studies%20of%20Multicore%20Architectures%20II%20%E2%80%93%20Computer%20Architecture2-211.png)

A big difference in Cells from normal CPUs is the ability of the SPEs in a Cell to be chained together to act as a stream processor. In one typical usage scenario, the system will load the SPEs with small programs , chaining the SPEs together to handle each step in a complex operation. For instance, the decoding of digital TV is a complex process that can be broken into a stream, as shown in Figure 38.10. The data would be passed off from SPE to SPE until the final processing is done. Another possibility is to partition the input data set and have several SPEs performing the same kind of operation in parallel. At 3.2 GHz, each SPE gives a theoretical 25.6 GFLOPS of single precision performance.

![](img/Case%20Studies%20of%20Multicore%20Architectures%20II%20%E2%80%93%20Computer%20Architecture2-213.png)

In order to deliver a significant increase in application performance in a power constrained environment, the Cell BE design exploits parallelism at all levels.

-   Data level parallelism – with the SPEs and SIMD instruction support
-   ILP – using a static scheduled and power aware micro-architecture
-   Compute-transfer parallelism – using programmable data transfer engines
-   TLP – with a multicore design approach and hardware multithreading on the PPE
-   Memory level parallelism – by overlapping transfers from multiple requests per core and from multiple cores

Thus, the Cell BE was designed based on the analysis of a broad range of workloads in areas such as cryptography, graphics transforms and lighting, physics, fast-Fourier transforms (FFT), matrix operations, and scientific workloads and is designed for graphics- and network-intensive jobs ranging from video games to complex imaging for the medical, defense, automotive and aerospace industries.

To summarize, we have dealt with two different styles of multicore architectures in this module:

-   UltraSPARC

–  Homogeneous

–  Multithreaded

–  Server applications

–  Power efficient

-   IBM’s Cell BE

–  Heterogeneous

–  Built for gaming applications

–  Exploits different types of parallelism

**Web Links / Supporting Materials**

1.  Computer Architecture – A Quantitative Approach , John L. Hennessy and David A. Patterson, 5th Edition, Morgan Kaufmann, Elsevier, 2011.

2.  IBM, Sony, Toshiba papers in ISSCC’05

1.  “A Streaming Processing Unit for a CELL Processor”, B. Flachs et. al.

1.  “The Design and Implementation of a First-Generation CELL Processor”, D. Pham et. al.

3.  “Microprocessor Report”,

1.  Reed Electronics Group, 2005, Jan. 31 & Feb. 14

4.  “IBM’s Cell Processor : The next generation of computing?”, D.K. Every, Shareware Press, Feb. 2005

5.  “Power Efficient Processor Architecture and The Cell Processor”, H.P. Hofstee, HPCA-11 2005

6.  “Power Efficient Processor Design and the Cell Processor”, IBM, 2005

7.  “Introducing the IBM/Sony/Toshiba Cell Processor”, J. H. Stokes, [http://arstechnica.com/](http://arstechnica.com/)

8.  “Cell Architecture Explained”, N. Blachford, http://www.blachford.info/

9.  Kunle Olukotun, Poonacha Kongetira, Kathirgamar Aingaran, “Niagara: A 32-Way Multithreaded Sparc Processor”, *IEEE Micro*, vol. 25, no. , pp. 21-29, March/April 2005, doi:10.1109/MM.2005.35

10.  [http://opensparc.sunsource.net/nonav/opensparct1.html](http://opensparc.sunsource.net/nonav/opensparct1.html)

11.  [http://www.sun.com/processors/UltraSPARC-T1/features.xml](http://www.sun.com/processors/UltraSPARC-T1/features.xml)

12.  [http://www.sun.com/servers/coolthreads/t1000/benchmarks.jsp](http://www.sun.com/servers/coolthreads/t1000/benchmarks.jsp)

13.  [http://news.com.com/Sun+begins+Sparc+phase+of+server+overhaul/2163-1010\_3-5983365.html](http://news.com.com/Sun+begins+Sparc+phase+of+server+overhaul/2163-1010_3-5983365.html)
