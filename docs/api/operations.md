# Lifetimes, concurrency, and durability

## Ownership and shutdown

A database owns its engine and mapped storage. Sessions, cursors, and snapshots
refer back to it. Their returned `shared_ptr`s and owned root references do not
extend the database object's lifetime. Destroy in this order:

1. Mutable tree handles and active transactions (commit or abort first).
2. Cursors, callback views, and owned tree snapshots.
3. Read and write sessions.
4. The database.

The examples declare the database first so C++ reverse-order destruction does
this naturally. Do not return a cursor or snapshot from a function whose local
database is about to close. Root ownership keeps a *version* alive; it does not
keep its database mapping alive. A cursor from an attached snapshot pins the
root version independently, but still depends on the database and session.

A snapshot remains frozen across subsequent commits. Copying it retains the
same version, not a deep copy of all data. Long-lived snapshots retain old
nodes and can increase disk usage, allocation pressure, and reclamation work.
Drop obsolete revisions deliberately. In-memory snapshots are not persistent
revision records; the Large Engine can store their owned subtrees in a manifest.

## Concurrency contract

Use one writer thread, with the database opened and write session created on
that thread. Serialize write transactions and scope each to completion before
starting another. Native SAL supports separate writers for independent root
tables; the shared engine API uses one serialized writer. The Large Engine database
adapter shares an engine write session across its write-session handles.

Each reader thread creates, uses, and destroys its own read session and cursor.
Keep sessions, snapshots, and cursors on their creating thread. The database
may be shared with reader threads provided it outlives them all. Join readers
before database destruction. Opening a read session does not freeze the root;
creating a cursor does. A new cursor sees a new committed snapshot; an existing
cursor never silently advances to a newer commit.

For portable use, keep **one live cursor per read session**. The Small Engine currently
uses one hazard slot per read session: creating a second cursor overwrites the
first cursor's protection. Use another read session if you need another live
cursor. Do not pass a writer's owned snapshot to another thread; use that
thread's read session to acquire its own snapshot. This contract does not
promise a shared snapshot transfer API between threads.

The Large Engine also supports **serialized handoff** of an inactive session
and its retained handle graph between threads. One shared synchronization gate
must cover every native operation and every handle copy or destruction that
touches that session, including cursors and owned snapshots. The requirement is
that two threads are never active on one SAL session simultaneously. Finish
each mutable transaction on one thread before releasing the gate; do not carry
a borrowed view or active residency scope across a handoff. Keep the database
alive until the entire handle graph is destroyed. A wrapper can enforce this
with a common mutex and opaque retained handles. This profile-specific pattern
does not provide concurrent access to one session and does not change the
portable creating-thread rule above.

### Read path and synchronization

**Practically lock-free:** once a snapshot is acquired, tree traversal does
not take node mutexes.
Readers follow immutable snapshot data while the writer changes private paths.
A reader does not wait for the writer's transaction to finish.

The Large Engine's `sal::read_lock` is an **atomic residency marker**, not a blocking
reader-writer mutex. Entering the read scope publishes the session's recycling
position; leaving clears it. The compactor can relocate objects while retaining
their old storage until readers finish. Long read scopes defer segment reuse,
and can eventually limit compaction progress; they do not make the reader wait
for a compactor lock. Node resolution loads the object's location atomically.

The Small Engine follows fixed addresses. Its read-session cursor acquires a snapshot by
publishing an atomic hazard slot and validating the root, retrying if publication
races with a commit. No relocation pin is needed during traversal.

| Boundary | Synchronization |
|---|---|
| Large Engine: acquire a published snapshot | Shared per-root mutex protects loading and retaining the root |
| Large Engine: publish a root | Per-root writer mutex and exclusive root-publication mutex |
| Small Engine: acquire a read-session cursor | Atomic hazard publication and root validation |
| Small Engine: acquire an owned root / publish roots | Root-publication mutex |
| Small Engine: start a write transaction | Per-root writer mutex held until commit or abort |
| Allocation and maintenance | Atomic fast paths plus locks or waits on some segment-claim, growth, sync, and release paths |

Root acquisition can therefore wait behind publication, even though an already
acquired snapshot can continue reading. Writer serialization and publication
are separate from reading the tree. **Mutex-free snapshot traversal** is the basis of the
practically lock-free design; the whole database does not promise formal lock-free
or wait-free progress. Page faults, storage I/O, and allocation can also wait.

A callback's byte view is borrowed only for the call. The Large Engine retains the atomic
residency marker for a borrowed view; a long callback can delay storage reuse.
Copy data before queueing it to another thread, doing slow I/O, or awaiting
anything. Native `pinned_value()` also holds a residency marker and is intended
for short scopes; it is outside the common API.

Do not mix multiple writable opens of the same directory or assume a shared
multi-process writer protocol. Read-only opens in the Large Engine use a shared advisory
lock and refuse while an exclusive writer owns the database. The Small Engine rejects
read-only opens. Open only a matching engine and compatible disk format.

## Configuration

Choose `Engine::config`, not another engine's config type. Set it before open.
`cfg.sync_mode` is the common setting. Other fields are profile-specific:

| Profile / field | Meaning |
|---|---|
| Small Engine: `max_segments` | Ordinary arena reservation in 8 MiB segments; default 4,096 = 32 GiB. Companion large arena reserves up to `min(max_segments, 512)` segments of 2 GiB. |
| Small Engine: `background_release` | Use the release thread (default true), or reclaim inline |
| Large Engine: `max_pinned_cache_size_mb` | Budget for pinned SAL segments; not a total RSS cap |
| Large Engine: `enable_read_cache` | Enable read-driven hot object promotion; profile default false |
| Large Engine: `read_cache_window_sec` | Access-decay interval used for hot object promotion |
| Large Engine: `max_database_size` | Address-space/storage ceiling, default 32 TiB; platform/filesystem constraints apply |

Both profiles default to `sync_mode::none`. The Large Engine's profile defaults disable
`write_protect_on_commit` and `enable_read_cache`. Other SAL runtime settings
are configurable through the profile config and forwarded by the factory. The explicit
`open_mode` controls read-only operation and overrides `cfg.read_only`.

For example, before opening a database using the Large Engine:

```cpp
psitri::big_engine::config cfg;
cfg.sync_mode = psitri::sync_mode::fsync;
cfg.max_pinned_cache_size_mb = 256;
cfg.enable_read_cache = true;
```

Budget for metadata, writer segments, unpinned OS cache, retained snapshots,
application memory, and temporary compaction space as well as this pinned
budget. Do not set it equal to all available RAM. SAL may clamp the effective
budget to platform limits; for diagnostics inspect
`db->underlying_allocator().get_runtime_config()` with no concurrent config
mutation. This is a diagnostic specific to the Large Engine, not the generic API.

Environment overrides `HEART_READ_CACHE=1`, `HEART_READ_CACHE_SEC`,
and `HEART_PINNED_MB` take precedence over these respective fields when set.
Clear them when validating application configuration. Layout experiments and
`engine_for_testing()` are not supported application tuning APIs.

## Durability

Publication, OS flushing, and crash recovery are separate properties. A commit
makes a root visible to new readers. The chosen sync policy requests how
storage is flushed; it does not by itself prove recovery correctness.

| Policy | Intent |
|---|---|
| `none` | No per-commit synchronization request; application supplies any required replay/recovery guarantee |
| `mprotect` | Memory write protection, not a durability guarantee |
| `msync_async` | Asynchronous flush hint |
| `msync_sync` | Blocking mapped-page flush |
| `fsync` | Request filesystem synchronization before returning |
| `full` | Request the platform's strongest supported device synchronization |

Select policy for the deployment's failure model and storage hardware. Blockchain
applications may reconstruct database state from a replicated, authoritative
block log and choose to synchronize derived state less frequently. Specify the
log's durability, replication acknowledgment, and replay window separately.
Full device synchronization is a policy choice, not a universally preferable
setting. Compare [matching durability tiers](../../website/content/fair-comparison.md#choose-the-durability-contract).
There is no universal power-loss guarantee across devices. Do not treat `db->sync()`
as a portable replacement for configured transaction durability: the Small Engine's
standalone `sync()` is currently a no-op, so it is not part of the core concept.

For the Large Engine's SAL storage, requested `msync`, `fsync`, and
`F_FULLFSYNC` operations retry `EINTR`. A terminal flush failure emits a
durability diagnostic and terminates the process with exit code **74**
(`sal::durable_io_failure_exit_code`). It does not return a successful commit,
unwind a transaction, run exit handlers, or allow later writes. A root or its
recovery record may already have reached the kernel, so such a failure cannot
be treated as a recoverable transaction rejection. Acknowledge a durable write
only after its configured commit returns successfully.

After a failed, unacknowledged commit, do not assume that the previous root was
restored. Reopen may find the complete old or complete new state; an actual
storage fault can also require recovery or operator repair. Validate the store
using the supported engine recovery procedures and the application's
authoritative log. A read-only open does not repair the original store.
This failure contract does not strengthen `none` or `mprotect` into durable
policies, and does not change the Small Engine's failure handling.

Transactions in the Small Engine sync their companion arena's heads and continuations before
the ordinary arena and root publication, using the configured policy. This
ordering also covers contiguous values in the companion arena.

The API suite verifies **clean close/reopen** with `fsync`. The Large Engine's large-value
suite also exercises refcount reconstruction and explicit full recovery.
Focused Large Engine tests inject ordinary flush errors and interruptions,
verify no acknowledgement after terminal failure, and check consistent,
unchanged read-only reopen without an additional power cut or corruption.
Comprehensive power-loss, arbitrary process-kill, disk-full, storage-corruption,
and sustained beyond-RAM validation is incomplete. Select the library against your
application's failure model and test recovery on the target storage stack.
The Small Engine exposes low-level rebuild machinery, but its database factory does not
perform a complete root walk/rebuild.

Use `create_only` for new databases and `open_existing` for expected existing
ones so missing data is not mistaken for a fresh successful open. Keep external
backups and engine/revision metadata in your deployment process. Never switch
the engine type and point it at existing files; migrate via logical data export
and import or a separately validated converter.
