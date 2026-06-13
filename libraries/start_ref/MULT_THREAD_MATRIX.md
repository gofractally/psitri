# Multi-Thread ART Matrix 

We have several different approaches to art/database state managment that
are circling slightly different use cases and these use cases need to be
explicitly defined so we can make rational choices and keep each library
in its lane.

On top of it all is an abstract transaction-based API over an
ordered key-value store. At issue is differences in:

    1. number of readers, number of writers
    2. kind, shape, and lifetime of snapshots 
    3. durability and crash recovery
    4. fitting in RAM or going beyond RAM 

Techniques include: Copy on Write, MVCC, and Undo/Redo deltas

The best strategy for each changes, but there is overlap for the most
common pattern:

  - Transactions that must be applied in sequence, and which
      may abort in a single-threaded context.  This pattern
      can be supported by all engines generically. 

  - Snapshots can be supported by:
       a. copy on write w/ shared state
       b. MVCC with undo/redo buffers 

  - Parallel Writers can be supported by:
       a. mvcc w/ per node lock (or striped locks) + undo
          - so long as writes don't depend upon reads..
       b. independent tables 

  - performance is maximized by batching and sorting writes.
    

Database Engines Available
    - psitri - btree-like cow limited by 1 byte dividers,
            and cacheline sharing of compressed ptr addresses. 
    - heart - buckets(pages) using 256-way ART radix index (COW) 
    - hkv   - in-memory stl-api ART key/val store with
               custom allocator support - single threaded
      

The Matrix Dimensions:

    - forking snapshots needed?
    - number of readers vs number of writers
    - persistence vs transient 
    - transactional read/write

Single-Threaded Do/Undo History
-------------------------------
The highest performance comes from a single threaded library, it
impliments transactions by building a temporary delta and doing 
merge queries against that delta, on commit it patches the main
tree and optional logs an undo log that allows roll back to the
prior state by patching the tree again. 

- making this crash resistant during commit() phase requires
careful "atomic" update steps that are not easy to impliment; 

There is no ability to take "snapshots" of old versions for
readers to query unless those snapshots are in the form of
merge deltas that readers have to read through, slowing down
every read.

Single-Threaded MVCC
--------------------
Multiple versions are maintained and the writer only has to
annotate the final value with a version number, readers then
have to filter the vinal value based upon their version. This
approach works when read-snapshots can be slower and when the
writes are irreversalbe... the writer would have to impliment
a do/undo history with a policy choice of whether MVCC writes
are optimistic (requiring undo writes on abort) or pessistimistic,
    requiring MVCC writes 

Single-Threaded - COW 
---------------------
In this mode there are reference counts, and all modifications 
create a cow-to-root. There can be multiple versions live at the
same time and each version can be read by any number of threads,
the only requirment is that the root objects have atomic reference
counts and are returned to the write thread to release. 

If all reference counts are atomic, and the memory allocator is 
thread safe then any thread can release. 


Fixed Allocator vs Dynamic Alloc
--------------------------------
Most allocators allocate once and the object lives at that address forever, 
but his has problems with memory fragmentation and cache polution
as pages mix hot and cold data. SAL introduces movable memory which
allows background compaction and optimization at the cost of read
indirection during queries. 

While data is small this indirection hurts performance by introducing
an extra cold-cache per layer of descent, costing about 50ns, with
4 layers of descent this in and of itself limits throughput to under
1 million tps cold.

However, at scale, every cold disk page is 100x worse. In most datasets
there is a long-tail of cold data that is never read and that data needs
to get segregated from the hot data because every cold miss is just that
expensive. 

Now in performance sensitive production systems, they should ideally size
their RAM to fit their dataset and if they do that, then relocatable memory
doesn't pay because it introduces the very cold-cache misses it is trying
to avoid. 

Thus fixed-memory allocators are best when you have sufficient ram,
     and relocatable allocators are best when they save you from
     disk thrashing and/or the compactor is doing lots of expensive
     data reorg in background threads.







## Single Threaded Reader/Writer  


  - If there is only one writer, then no atomic operations are needed,
  and snapsh
