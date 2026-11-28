# start_ref: status

`start_ref` is the storage library behind the **Small Engine**
(`psitri::small_engine` = `start_ref::kv::engine`). The user-facing contract is
in `docs/api/`; this page records what the library is today and what is known
to be missing. Design background is in `SPEC.md` (allocator and durability
model) and `MULT_THREAD_MATRIX.md`.

## What it is

| Layer | File | Role |
|---|---|---|
| Free map | `include/start_ref/freeset.hpp` | One bitmap tree per 8 MiB segment; runs of 1 to 64 lines |
| Arena | `include/start_ref/arena.hpp` | `data.part`, `free.part`, `refs.part`, `meta.part`; 128-byte lines at fixed addresses; one u8 reference count per line; 64 reader slots; 16 root slots |
| Large values | `include/start_ref/large_value.hpp` | The same arena at a 32 KiB grain in `large/`; values over about 8 KiB, chunked above 2 MiB |
| Tree | `include/start_ref/hkv/` | Radix tree with copy-on-write and line-granular sharing |
| Database | `include/start_ref/kv.hpp` | `database`, `write_session`, `read_session`, `transaction`, `cursor`, owned `tree` snapshots |

Nothing is relocated or compacted. Freed lines are reused where they are; a
segment is claimed again once an eighth of its 8-line groups are free.

## Behaviour that is tested

- The shared API suites `tests/api/` (`api-conformance-tests`) and the
  library's own `test/` (`start_ref-tests`).
- `tests/api/small_engine.cpp` (`small-engine-tests`):
  - process kill at random points, repeated on one store, every sync mode: no acknowledged transaction lost, tree matches its
    model, no storage leaked;
  - recovery when root slots share nodes and a snapshot and a transaction
    were abandoned;
  - storage use level under churn at constant live size;
  - reaching `max_segments` mid-transaction: exception, rollback, store usable;
  - the 1,024-byte key limit at the boundary;
  - a session's second transaction, misordered `open_root`, and a 65th
    reader are refused with an exception;
  - write and read sessions, and a transaction in progress, moving between
    threads with serialized use (root writer locks are semaphores owned by
    the transaction);
  - the session that reached the limit writes again after aborting; the
    limit is reached at 100% use for small values; running out of room for a
    multi-line node is `storage_full`; a writer ahead of the releaser waits;
  - a multi-root commit whose reference is refused publishes nothing;
  - a store filled to refusal one upsert at a time is emptied one remove at
    a time (removes allocate from a reserve), can drop its root, and can be
    reopened with a higher limit;
  - room held by a version whose reader has gone, or by an idle session's
    segment, is found before the store is declared full; an allocation that
    can never fit gives up after a bounded number of reclaim rounds;
  - two writers aborting large transactions at once (the release queue has
    many producers);
  - what a reader sees of a two-root commit;
  - read-only open refused; second open of an open store refused;
  - more owned snapshots of one version than a writer reference can join.
  - a failed `msync` of data pages in a flushing mode ends the process with
    exit code 74 before the root is stored; `sync_mode::mprotect`,
    `write_protect_on_commit` and `checksum_on_commit` are refused at open
    (`tests/api/small_engine_crash.cpp`).

## Open on unclean stop

`meta.part` carries a clean mark, cleared at open and set at close.
`database::closed_cleanly(dir)` reads it. An open that finds it clear runs
`kv_tree::recover_mark` from each root slot and rebuilds `free.part` and
`refs.part` for both arenas (`database::recover_`); `recovered_on_open()`
reports that. The walk mirrors the release cascade in `cow_destroy_dead_`: a
change to how nodes own their lines has to be made in both.

## Known gaps

- The failure defended against is the death of the process with the
  operating system still running (`tests/api/small_engine_crash.cpp`). Power
  loss and operating-system failure are not defended against, by decision:
  nothing orders a root slot behind its nodes on the device in `none` and
  `msync_async`, nodes and root slots carry no checksum, and with two write
  sessions a flushing commit can return with some of its pages not yet
  flushed (dirty bits are per segment and cleared by whichever session syncs
  it) (#11, #35).
- A commit of several roots stores the root slots one after another: a
  process that dies between two stores leaves the earlier roots new and the
  later ones old.
- The recovery walk does not cover the compiled-out page layout (#14).
- Typed values registered with `register_value_types` hold references the
  arena does not know; recovery calls no hooks and reports how many typed
  values it found (`typed_values_found_by_recovery()`); reconciling what they
  refer to is the application's (#14).
- `database::sync()` is a no-op; durability follows the configured mode at
  commit.
- Read-only open is not implemented. (A store that is open is locked: a
  second open of its directory throws.)
- A multi-root commit publishes its root slots one at a time.
- Each writer that is in a transaction needs a segment to itself; with more
  simultaneous writers than segments the extra writers' ordinary writes are
  refused at once. Removes do not need a segment.
- A single transaction that removes very many keys from a full store can use
  up the remove reserve before it commits (about 9,800 small keys on a
  two-segment store in the tests); `reserve_bytes` sizes it.
- The reserve's guaranteed runs cover node lengths of 2 to 16 lines exactly
  (a remove needs at most 14: it copies routers and key prefixes, never
  leaves) and keep a few 64-line runs as margin; a guaranteed run is repaid only by
  a freed run at least as long, so a remove that built a longer node than any
  it freed would leave that length owed (no tested shape does).
- While the reserve is owed a run of some length, every free of a run that
  long or longer takes the reserve's mutex (gofractally/psitri#36).
- At the ceiling a session takes free lines from any segment, which gives up
  the placement policy's low write amplification for as long as it lasts.
- Reverse cursor steps are a series of forward probes (cost follows key
  length); there is no backward link.
- The large-value arena has no ceiling of its own below
  `min(max_segments, 512)` segments of 2 GiB (#13).
