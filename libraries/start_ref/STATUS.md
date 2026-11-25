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
  - process kill at random points, repeated on one store, every sync mode
    except `mprotect`: no acknowledged transaction lost, tree matches its
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

## Open on unclean stop

`meta.part` carries a clean mark, cleared at open and set at close.
`database::closed_cleanly(dir)` reads it. An open that finds it clear runs
`kv_tree::recover_mark` from each root slot and rebuilds `free.part` and
`refs.part` for both arenas (`database::recover_`); `recovered_on_open()`
reports that. The walk mirrors the release cascade in `cow_destroy_dead_`: a
change to how nodes own their lines has to be made in both.

## Known gaps

- Power loss and operating-system crash are untested for every sync mode. The
  rebuild makes the derived files irrelevant to that case, but nothing checks
  that the data pages a root points to reached storage (#11).
- `msync` results on data pages are ignored (`arena::sync_segment_`) (#11).
- `sync_mode::mprotect` protects nothing here, and
  `write_protect_on_commit` / `checksum_on_commit` are ignored (#12).
- There is no checksum on nodes or root slots; a damaged image is detected
  only when the recovery walk leaves the arena (#11).
- The recovery walk does not cover the compiled-out page layout (#14).
- Typed values registered with `register_value_types` hold references the
  arena does not know; recovery does not replay their retain hooks (#14).
- `database::sync()` is a no-op; durability follows the configured mode at
  commit.
- Read-only open is not implemented. (A store that is open is locked: a
  second open of its directory throws.)
- A multi-root commit publishes its root slots one at a time.
- Each writer that is in a transaction needs a segment to itself; with more
  simultaneous writers than segments the extra ones are refused.
- A single transaction that removes very many keys from a full store can use
  up the remove reserve before it commits.
- At the ceiling a session takes free lines from any segment, which gives up
  the placement policy's low write amplification for as long as it lasts.
- Reverse cursor steps are a series of forward probes (cost follows key
  length); there is no backward link.
- The large-value arena has no ceiling of its own below
  `min(max_segments, 512)` segments of 2 GiB (#13).
