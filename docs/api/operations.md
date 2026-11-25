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

Roots are independent. Each writer thread creates its own write session and
uses it only on that thread; write sessions working on different roots run at
the same time and do not wait for each other.

A transaction holds its root's writer from `start_transaction` until it commits
or aborts. A second transaction on the same root, from any session, waits and
then starts from what the first published, so no commit is lost and nothing is
retried. Starting a second transaction on a root from the thread that already
has one open on it is a caller error. The Small Engine reports it:
`start_transaction` throws `std::logic_error`, as it does for a second
transaction on a session that already has one open, and `open_root` throws for
a root at or below one the transaction already holds. Do not count on the
Large Engine detecting it: there the call waits for itself and never returns. Additional roots opened with `open_root` wait for their
writer when opened in ascending index order.

A Small Engine session is not thread-safe, and it is not tied to the thread
that created it: a write session or read session may be used from any thread
as long as only one thread uses it at a time, which the caller arranges. A
transaction holds its root's writer lock on behalf of the thread that started
it, so commit or abort it on that thread; hand a session over between
transactions. Builds with assertions enabled stop the process with a message
when two threads are inside one write session at once.

A session keeps its database open: the store closes when the last handle to
the database and the last session are gone. Trees, transactions and cursors
are lighter and do not; each must be destroyed before the session it came
from. (The Large Engine's `shared_tree` belongs to the store, not to a
session, and must be destroyed before the store closes.)

Readers never wait for a writer's work: they are excluded only for the moment
a root is published. `tests/api/writers.cpp` checks these rules on both engines.

Each reader thread creates, uses, and destroys its own read session and cursor.
Keep sessions, snapshots, and cursors on their creating thread. The database
may be shared with reader threads provided it outlives them all. Join readers
before database destruction. Opening a read session does not freeze the root;
creating a cursor does. A new cursor sees a new committed snapshot; an existing
cursor never silently advances to a newer commit.

A read session may hold several cursors, each on the snapshot it was created
on. Do not pass a writer's owned snapshot to another thread; use that thread's
read session to acquire its own. The portable contract does not promise a
snapshot that can be handed between threads; the Large Engine's `shared_tree`
is such a handle.

Trees, transactions and cursors made through a Large Engine write session
refer to that session and must not outlive it; a `shared_tree` does not and
may. The Large Engine also supports **serialized handoff** of an inactive
session and its retained handle graph between threads. One shared synchronization gate
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
| Large Engine: start a write transaction | Per-root writer mutex held until commit or abort |
| Large Engine: publish a root | Exclusive root-publication mutex |
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
read-only opens: `open` with `open_mode::read_only` throws `std::runtime_error`
saying the engine does not support it, before creating or touching any file.
While a Small Engine store is open its files are locked exclusively, so a second
open of the same directory, from the same or another process, throws. Open only a
matching engine and compatible disk format.

## Configuration

Choose `Engine::config`, not another engine's config type. Set it before open.
`cfg.sync_mode` is the common setting. Other fields are profile-specific:

| Profile / field | Meaning |
|---|---|
| Small Engine: `max_segments` | Ordinary arena ceiling in 8 MiB segments; default 4,096 = 32 GiB. Writes that need more throw `storage_full`; see [bounding and watching storage](engines.md#bounding-and-watching-storage). Companion large arena reserves up to `min(max_segments, 512)` segments of 2 GiB. |
| Small Engine: `background_release` | Use the release thread (default true), or reclaim inline |
| Small Engine: `write_protect_on_commit`, `checksum_on_commit` | Accepted for source compatibility and ignored |
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

### Small Engine: stopping without closing

A Small Engine store records whether its last owner closed it.
`database::closed_cleanly(dir)` reads that without opening the store, and
`db->recovered_on_open()` says whether this open found it not closed. When it
was not, `open` rebuilds the free map and the reference counts from the
published roots before it returns: these are derived files that are never
synced and that, after a stop, still describe the dead writer's unfinished
transaction. The rebuild reads every node once, so its cost follows the size
of the live data. If the walk reaches storage outside the arena, `open` throws
and leaves the files as they are; it does not come back empty.

What `tests/api/small_engine.cpp` shows, and nothing beyond it: a writer
running committed transactions was killed with `SIGKILL` at random moments
2,202 times, repeatedly on the same store. After every kill the store reported
that it had not been closed, reopened, held every transaction whose `commit()`
had returned and at most the one that was in flight, matched its model in a
full forward scan with exact row and index counts, and kept its storage use
where it had settled; after the last kill, removing every key drained the
arenas to zero.

| `sync_mode` | After the process is killed (operating system keeps running) | After power loss or an operating-system crash |
|---|---|---|
| `none` | Every committed transaction present (tested) | Not tested. Nothing is flushed by commit: expect to lose recent transactions, and the root slot can reach storage before the nodes it points to, which open may or may not be able to detect |
| `mprotect` | Not tested. The Small Engine applies no page protection; it flushes data pages as `msync_sync` does but not the root slot | Not tested |
| `msync_async` | Every committed transaction present (tested) | Not tested. Flushes are only requested, so the same exposure as `none` applies |
| `msync_sync` | Every committed transaction present (tested) | Not tested. Commit waits for `msync(MS_SYNC)` of the data pages, then of the root slot |
| `fsync` | Every committed transaction present (tested) | Not tested. As `msync_sync`, plus `fsync` of the data and root files before commit returns |
| `full` | Every committed transaction present (tested) | Not tested. As `fsync`, with the platform's device flush (`F_FULLFSYNC` on macOS) |

A process kill leaves the operating system holding every page the process
wrote, which is why all modes pass that test alike; the modes differ only in
what has reached the device, and that has not been exercised by cutting power.
Flush errors are not checked on the Small Engine's data pages. A transaction
that commits several roots writes one root slot after another, so a stop
between them leaves some roots new and others old.

Use `create_only` for new databases and `open_existing` for expected existing
ones so missing data is not mistaken for a fresh successful open. Keep external
backups and engine/revision metadata in your deployment process. Never switch
the engine type and point it at existing files; migrate via logical data export
and import or a separately validated converter.
