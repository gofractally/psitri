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
revision records; Big can store their owned subtrees in a manifest.

## Concurrency contract

Use one writer thread, with the database opened and write session created on
that thread. Serialize write transactions and scope each to completion before
starting another. Native engine internals support differing concurrency
patterns; those are not promises of the portable API.

Each reader thread creates, uses, and destroys its own read session and cursor.
Keep sessions, snapshots, and cursors on their creating thread. The database
may be shared with reader threads provided it outlives them all. Join readers
before database destruction. Opening a read session does not freeze the root;
creating a cursor does. A new cursor sees a new committed snapshot; an existing
cursor never silently advances to a newer commit.

For portable use, keep **one live cursor per read session**. Small currently
uses one hazard slot per read session: creating a second cursor overwrites the
first cursor's protection. Use another read session if you need another live
cursor. Do not pass a writer's owned snapshot to another thread; use that
thread's read session to acquire its own snapshot. This contract does not
promise a shared snapshot transfer API between threads.

A callback's byte view is borrowed only for the call. Big holds a residency
lock while the callback runs; a long callback can delay compaction. Copy data
before queueing it to another thread, doing slow I/O, or awaiting anything.
Similarly, native `pinned_value()` locks are for short scopes; they are outside
the common API.

Do not mix multiple writable opens of the same directory or assume a shared
multi-process writer protocol. Big read-only opens use a shared advisory
lock and refuse while an exclusive writer owns the database. Small rejects
read-only opens. Open only a matching engine and compatible disk format.

## Configuration

Choose `Engine::config`, not another engine's config type. Set it before open.
`cfg.sync_mode` is the common setting. Other fields are profile-specific:

| Profile / field | Meaning |
|---|---|
| Small: `max_segments` | Ordinary arena reservation in 8 MiB segments; default 4,096 = 32 GiB. Companion large arena reserves up to `min(max_segments, 512)` segments of 2 GiB. |
| Small: `background_release` | Use the release thread (default true), or reclaim inline |
| Big: `max_pinned_cache_size_mb` | Budget for pinned SAL segments; not a total RSS cap |
| Big: `enable_read_cache` | Enable read-driven hot object promotion; profile default false |
| Big: `read_cache_window_sec` | Access-decay interval used for hot object promotion |
| Big: `max_database_size` | Address-space/storage ceiling, default 32 TiB; platform/filesystem constraints apply |

Both profiles default to `sync_mode::none`. Big's profile defaults disable
`write_protect_on_commit` and `enable_read_cache`, preserving its historical
baseline. It otherwise inherits SAL's runtime settings. The factory now forwards
all fields to SAL; previously it forwarded only sync mode. The explicit
`open_mode` controls read-only operation and overrides `cfg.read_only`.

For example, before opening a Big database:

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
mutation. This is a Big-specific diagnostic, not the generic API.

Historical benchmark overrides `HEART_READ_CACHE=1`, `HEART_READ_CACHE_SEC`,
and `HEART_PINNED_MB` take precedence over these respective fields when set.
Clear them when validating application configuration. Layout experiments and
`engine_for_testing()` are not supported application tuning APIs.

## Durability

Publication, OS flushing, and crash recovery are separate properties. A commit
makes a root visible to new readers. The chosen sync policy requests how
storage is flushed; it does not by itself prove recovery correctness.

| Policy | Intent |
|---|---|
| `none` | No synchronous durability request; suitable for disposable benchmarks |
| `mprotect` | Memory write protection, not a durability guarantee |
| `msync_async` | Asynchronous flush hint |
| `msync_sync` | Blocking mapped-page flush |
| `fsync` | Request filesystem synchronization before returning |
| `full` | Request the platform's strongest supported device synchronization |

Select policy for the deployment's failure model and storage hardware. There
is no universal power-loss guarantee across devices. Do not treat `db->sync()`
as a portable replacement for configured transaction durability: Small's
standalone `sync()` is currently a no-op, so it is not part of the core concept.

Small transactions sync their companion arena's heads and continuations before
the ordinary arena and root publication, using the configured policy. This
ordering also covers contiguous values in the companion arena.

The API suite verifies **clean close/reopen** with `fsync`, not power failure or
arbitrary process-kill recovery. Neither engine is certified for those scenarios
by this API change. Release qualification must still cover interrupted commits,
allocator/refcount reconstruction, reuse after recovery, disk-full/I/O failures,
and prolonged compaction with data beyond RAM. Small exposes low-level rebuild
machinery but the database factory does not perform a complete root walk/rebuild.
Do not turn a passing C++ concept or clean-reopen test into a stronger claim.

Use `create_only` for new databases and `open_existing` for expected existing
ones so missing data is not mistaken for a fresh successful open. Keep external
backups and engine/revision metadata in your deployment process. Never switch
the engine type and point it at existing files; migrate via logical data export
and import or a separately validated converter.
