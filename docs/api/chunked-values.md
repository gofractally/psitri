# Large values in Small and Big

`psitri::small_engine` and `psitri::big_engine` store byte values up to
**4,294,967,295 bytes (4 GiB − 1 byte)** using the existing `upsert` and `get`
API. Both expose this limit as `Engine::max_value_size`. Unit tests allocate,
read, replace, and free values through **512 MiB** interleaved with smaller
records. Big uses segment-aware chunks; Small uses fixed-address allocations
of at most 2 MiB. The required engine concept is unchanged, so experimental
engines keep their existing interfaces and limits.

```cpp
using E = psitri::big_engine;
auto tx = writer->start_transaction(0);
tx.upsert("file", bytes); // string_view; input remains owned by the caller
tx.commit();

auto cursor = reader->create_cursor(0);
auto range = cursor.get("file", 30ull * 1024 * 1024, 4ull * 1024 * 1024);
cursor.get("file", [](std::string_view part) {
    consume(part); // one contiguous view, valid during this callback
}, offset, length);

// Big's native holder; the owning and callback calls above work on both engines.
cursor.find("file");
auto held = cursor.pinned_value(offset, length);
consume(held.view()); // borrowed or assembled; valid while held lives
```

Omitting offset reads from zero; omitting length reads the remainder. Reads
clamp at EOF and distinguish an empty range from a missing key. See
[range reads](range-reads.md) for the complete contract.

## Big's storage format

All external values use one packed `key_val_node` format, defined in
[key_val_node.hpp](../../libraries/sal_art_ref/include/sal_art_ref/key_val_node.hpp).
Only continuation payloads use a second type, `value_chunk`, defined in
[large_value.hpp](../../libraries/sal_art_ref/include/sal_art_ref/large_value.hpp).
There are no legacy-format readers or separate wide-value/descriptor types.

```text
key_val_node (29-byte fixed header)
+------------------------------------+
| SAL/versioned base       20 bytes   |
| Logical value length      uint32   |
| Key length                uint16   |
| Continuation count        uint16   | excludes this first chunk
| Value type                 uint8   | bytes or subtree
+------------------------------------+
| Optional full key                  |
| Directory[count]                   | {logical_start:uint32, child:uint32}
| First payload                      |
+------------------------------------+
              | directory references
              v
value_chunk (20-byte fixed header)
+------------------------------------+
| SAL allocation header    16 bytes   |
| Payload length            uint32   |
| Payload                            |
+------------------------------------+
```

Zero continuations means a contiguous value. Otherwise, the first directory
entry's logical start is the first payload's length; adjacent starts and the
logical total determine continuation lengths. Each entry occupies **8 bytes**.
Subtree records have no continuations and contain their nested root reference.
Byte records small enough to fit inside Big leaf pages keep the existing inline
representation.

A page owns the head. The head owns one reference to each continuation, with no
self-reference. Copying a page or retaining a snapshot shares the head and its
payloads. Destroying the last owner releases all continuations. Construction
allocates the head first, then transfers each continuation from a temporary RAII
owner into its directory. The directory is complete before publication or sync.
SAL does not compact these unsynced allocations, even after segment rollover.

Starting in a fresh segment, a 512 MiB value uses **17 allocations/control
blocks**: one head with its first payload and 16 continuations. That is **136
bytes of control blocks**, excluding the owning page/root. There is no separate
directory allocation. Directory metadata can still require an additional chunk
at a payload-capacity boundary.

## Small's storage format

Small keeps its existing allocator: a run occupies at most one 64-bit freeset
word, with **64 × 32 KiB = 2 MiB** of physical storage. Logical values larger
than a leaf allocation are split across runs in the same large-value arena.
Live allocations never relocate, so directory entries contain direct 32-bit
grain indices. There are **no SAL control blocks** or residency locks.

The existing native leaf retains its 32-bit physical slot length. Ordinary
byte slots remain `[uint16 type=0][payload]`. Chunked byte slots use internal
type 65,535 and this layout, defined in
[large_value.hpp](../../libraries/start_ref/include/start_ref/large_value.hpp):

```text
Native leaf / first chunk
+------------------------------------+
| Key suffix length          uint16   | existing leaf header: 8 bytes
| Key prefix length / flags  uint16   |
| Physical slot length       uint32   |
+------------------------------------+
| Internal type = 65535      uint16   | chunked slot header: 8 bytes
| Logical value length       uint32   |
| Continuation count         uint16   | excludes this first chunk
| Directory[count]                    | {logical_start:uint32, grain:uint32}
| First payload                       |
+------------------------------------+
| Key bytes (suffix or full key)      |
+------------------------------------+
              | direct grain indices
              v
Continuation allocation
+------------------------------------+
| Payload (up to 2 MiB)               | no per-chunk header
| Padding to a 32 KiB grain           |
+------------------------------------+
```

The head reserves space for the full input key so later leaf splits can change
its key representation without exceeding the allocation limit. Each directory
entry is **8 bytes**; adjacent starts and the logical total determine payload
lengths. The head fills its remaining space with data; continuations hold up to
2 MiB each. A **512 MiB value uses 257 allocations**: the head plus 256
continuations, with a 2,048-byte directory embedded in the head. This is why
an 8-bit continuation count is insufficient for Small.

Snapshots share the head. When a tree edit copies the head, its continuations
are retained independently. A continuation at the allocator's writer refcount
threshold is cloned into another fixed allocation. Destroying the final head
owner releases its continuation references; the final reference to a continuation
reclaims its entire run. Preparation and clone failures roll back references
acquired before the failure. Continuation data and heads are synced before
publishing a committed root.

The internal representation is hidden from byte callers: `value_type()` and
`get_any().type()` report raw bytes (type zero) for either encoding. Registered
typed-object IDs are 1 through 65,534; IDs zero and 65,535 are reserved. Native
`get_any().bytes(offset, length)` supports the same range rules and owns any
crossing-range assembly until its next byte read or destruction. Copies share
an existing assembly buffer. The transaction must still protect the underlying
slot for subsequent reads. Chunking applies to the maintained radix layout;
the gated `PART_CBUCKET` layout remains experimental and rejects large values.

## Limits shared by both profiles

Lengths are checked before narrowing or mutation. The 16-bit continuation
count accommodates both physical layouts without changing public read names.

**512 MiB is the largest payload exercised by the current allocation tests.**
Placement arithmetic is tested through `UINT32_MAX`, and a protected virtual
mapping verifies that a 4 GiB input is rejected without reading its bytes.
Available disk, address space, configuration, and input/output memory still
constrain practical capacity.

Native `value_size()` in both profiles, plus Big's `get(key, std::string*)`
and displaced-value size returns, use signed 64-bit results to accommodate the
full unsigned-32-bit range while preserving negative sentinels. Range offsets and lengths remain 64-bit. The
required public concept and its owning/callback result types are unchanged.

This is the new library's storage format; compatibility with development-era
files is not retained. Big and Small do not share serialized formats or raw IDs.

## Big segment placement

SAL uses 32 MiB segments, 128-byte allocation grains, a 64-byte footer, and a
terminal-marker reserve. A fresh allocation can occupy **32 MiB minus 256 bytes**.
Subtract the selected record's header and key to get its usable payload capacity.
A continuation has a 20-byte header, giving a fresh payload capacity of
**33,554,156 bytes**. The head subtracts its 29-byte header, key, and directory
instead.

Let `V` be payload bytes and `T = 2 MiB`. Whole-value capacity subtracts the
head header and key; chunked head capacity also subtracts the directory.

1. If the whole value fits in the current segment, put it there.
2. If it only fits whole in a fresh segment and `V > T`, roll over and put it there.
3. A smaller value may consume a useful tail and continue in the next segment.
   Tails below **64 KiB of payload after directory overhead** are skipped when
   a split would be needed.
4. If the value cannot fit whole in a fresh segment, compare tail-first and
   fresh-first layouts. Choose the fewest chunks larger than `T`, then the fewest
   total chunks, then the least skipped tail. A tie consumes the current tail.

For each candidate, middle continuations fill their segments and the final
continuation stores only the remainder. Directory overhead is included before
scoring, rather than treating the head as a raw continuation. With `A` available
allocation bytes, `H = 29 + key_length`, and continuation capacity `C`, the
smallest continuation count satisfies:

```text
V <= (A - H - 8*N) + N*C
N  = ceil((V - (A - H)) / (C - 8))  // when V > A - H
first_payload = A - H - 8*N
```

A candidate without room for its directory and first payload is discarded.
The score counts large chunks and all chunks, including the head; skipped space
breaks ties. The comparison does not subdivide into arbitrary 2 MiB pieces.

In these approximate examples, `C` is a full continuation's payload capacity
and `R` is the usable tail after metadata:

| Example, ignoring small header differences | Placement |
|---|---|
| Repeated 3 MiB values | Ten share a segment; the next starts another segment |
| `V = C + 1 MiB`, `R = 16 MiB` | Start fresh: full chunk plus 1 MiB |
| `V = C + 1 MiB`, `R = 1 MiB` | Consume tail: 1 MiB plus full chunk |
| `V = C + 1 MiB`, `R = C - 1 MiB` | Consume tail: same chunk counts, less skipped space |

Capacity comes from `allocator_session::current_allocation_capacity()`, not
`segment::free_space()` (which includes reserved bytes). Selection and allocation
run consecutively on the session's owner thread. The directory occupies the
start of the head allocation; no separate allocation interrupts the payload
sequence. Subsequent records can reuse the final segment's remaining space.

**Finalization/sync credits skipped tails exactly once.** Placement only requests
rollover through `next_allocation_segment()` or ordinary allocation. It does not
add free bytes, invent a freed object, or duplicate terminal-marker accounting.
The freed tail contributes to normal compaction eligibility. Tests check exact
free-byte totals after rollover and repeated sync, including already-synced tails.

## Reading chunks and Big relocation

Each directory entry stores a continuation's logical start. Binary search finds
the selected continuation, or selects the head when the offset precedes the
first entry. Only intersecting payloads are resolved and copied. A range ending
at a boundary does not read the next child. Ranges inside the head borrow its
payload directly, just as ranges inside a continuation do.

A range inside one allocation is borrowed directly. Big protects its address
with a residency lock; Small relies on its fixed address and the owning
snapshot or transaction. In both profiles, a crossing range assembles exactly the requested bytes into one owned string. Owning string
reads take that buffer without a second full copy; callback reads call once.
Big's `pinned_value()` holds either the borrowed pin or shared ownership of the buffer,
so holder copies and moves preserve the view.

The root/snapshot keeps the logical value alive. Big crossing reads copy the needed
directory entries while pinned, release that pin, then resolve and protect each
chunk separately while copying. No raw directory pointer survives across an
unlock. A positioned page cursor also re-resolves its stable page/child IDs on
each read, rather than trusting a cached address after compaction.

Ordinary compaction relocates a whole immutable allocation behind the same
control block. The head's directory moves with its first payload; continuations
move independently. Compaction preserves chunk boundaries and ownership. Re-chunking,
partial writes, deduplication, and streaming writes are not implemented.
`upsert` still requires a contiguous caller-owned input, and a full owning read
requires memory for the entire result. Use ranges to bound read memory.

## Validation and remaining qualification

[Big's large-value tests](../../tests/api/large_values.cpp) cover:

- 65,536 bytes; 2 MiB − 1, exact, and + 1; 3, 8, 15, and 16 MiB;
  fresh contiguous capacity − 1, exact, and + 1; 64, 128, and 512 MiB.
- Mixed small inserts/deletes, whole reads, odd chunk boundaries, range callbacks,
  borrowed/owned holders, EOF, exceptions, overwrite, shrink, and full reclamation.
- Long external keys, page splits, subtrees, snapshots, savepoint/transaction abort,
  clean writable/read-only reopen, refcount rebuild, and explicit full recovery.
- Observed relocation of the head/directory and of a continuation while another
  thread reads an old snapshot.
- Embedded first-payload borrowing, exact-boundary reads, two allocations for a
  two-chunk value, checksums/write protection, and 4 GiB rejection before mutation.
- Repeated 3 MiB packing, tail-first/fresh-first decisions, final-tail reuse,
  and exact skipped-tail accounting.

[Small's large-value tests](../../tests/api/small_large_values.cpp) cover:

- The exact contiguous-leaf boundary and adjacent sizes; 2, 3, 4, 16, 64,
  128, and 512 MiB values among smaller records, followed by full reclamation.
- Whole reads, every continuation boundary, direct borrowing, one-call crossing
  callbacks, EOF, exceptions, holder copy/move, and the protected 4 GiB rejection.
- Single-large-leaf roots, snapshot copies, leaf splits, saturated continuation
  refcounts, concurrent readers, multi-root reads, rollback, and durable reopen.
- Arena exhaustion during continuation preparation and head allocation, with
  no leaked chunks or change to the prior value.

The Big tests' 128 MiB setting bounds the requested pinned cache; it is not a measurement
or bound on process RSS or the OS page cache. These tests do not establish power-loss
atomicity, disk-full behavior, performance versus other engines, or production
beyond-RAM qualification. Those remain release qualification work for Psibase.
