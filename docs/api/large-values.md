# Impact assessment: 24-bit large-value lengths

Historical pre-implementation assessment, 2026-09-17. The original analysis
below describes the former 65,535-byte Big limit and the proposed 24-bit change.
It is superseded by [implemented large-value storage](chunked-values.md):
32-bit logical lengths, 16-bit continuation counts, 8-byte directory entries
inside the first chunk, and tests through 512 MiB in both Small and Big.
Small retains its 2 MiB physical allocator and uses direct-index continuations;
Big uses segment-aware SAL allocations. Both have a 4 GiB − 1 byte logical limit. Development-era disk compatibility is not required. The discussion
below is historical; its alternatives and compatibility requirements do not
specify the current implementation. Binary units are used throughout.

## Updated direction

The latest direction is **segment-aware variable-sized chunks with split
avoidance**. A value over 2 MiB that fits a fresh segment moves there whole
only if it would otherwise split. Values larger than one segment compare using
the current tail against starting fresh, favoring fewer chunks over 2 MiB,
then fewer total chunks, then less unused tail space. This supersedes the
intermediate quarter-segment ceiling and unconditional tail filling.
The header/descriptor records each child's logical range so reads can locate
the correct chunk despite unequal sizes. See the
[chunked-value design](chunked-values.md) for allocation, offset indexing,
ownership, compaction, and control-block costs.

A 24-bit field can represent a contiguous value below 16 MiB, but cannot
represent a near-32-MiB chunk. The implemented design uses checked 32-bit lengths
for both chunks and logical values. The former 24-bit proposal below explains
why widening only the original 16-bit field was insufficient.

The current Psibase checkout inspected at
`/Users/dlarimer/psibase-main-business` defaults `ConfigRow::maxValueSize` to
`8 << 20`, and `NativeFunctions::verifyConfigRow` allows up to `15 << 20`.
Triedent already uses a 24-bit object size in `triedent::object_header`;
`value_node::data_size()` derives the value size after subtracting key and node
metadata. Thus 2 MiB is a useful minimum test, but it is insufficient for full
compatibility with Psibase's currently permitted data.

## Localized representation change

The affected field is
[`key_val_node::value_len`](../../libraries/sal_art_ref/include/sal_art_ref/key_val_node.hpp).
Its current packed record is:

| Part | Current | Direct widening |
|---|---:|---:|
| `versioned` base | 20 bytes | 20 bytes |
| Key length | 2 bytes | 2 bytes |
| Value length | 2 bytes | 3 bytes |
| Value type | 1 byte | 1 byte |
| Header total | 25 bytes | 26 bytes |

The representable value range grows from 0..65,535 to 0..16,777,215 bytes.
Change the constructor's narrowing store, getter/field encoding, limit check,
layout assertion, and associated tests. Perform all validation and allocation
size arithmetic before narrowing to a stored field. Bitfields silently truncate
out-of-range assignments; making one wider does not make it checked.

Only external records grow. The 128-byte inline-entry threshold, small inline
value-length fields, leaf-page offsets, radix routing, control-block IDs, and
2-byte page reference codes can stay as they are. The page reference code
identifies an external object; it does not encode the object's payload length.
For ordinary short keys, the existing `vt_extern_val` path keeps the full suffix
in the page and puts only the value into `key_val_node`. Longer suffixes use
`vt_extern`, storing the full key with the value.

An alternative packs key length into 11 bits, value length into 24, and type
into 5, retaining the current 25-byte total. Eleven key bits cover the inclusive
1,024-byte key limit, and five type bits exceed the current bytes/subtree needs.
This is also a format change and couples the metadata layout more tightly to
current key/type limits. Prefer the straightforward 26-byte form unless
measurements justify the tighter packing. An explicit three-byte length codec
makes byte order clear; a C++ bitfield needs compiler/layout checks.

## Allocation capacity and cost

[SAL configuration](../../libraries/sal/include/sal/config.hpp) currently declares
32 MiB segments and a 16 MiB maximum-object budget. The new chunk path must
reconcile that constant/assertion and enforce the actual allocation capacity.
Allocation sizes are already 32-bit and the allocation header has a 25-bit
size. The current segment's `can_alloc()` geometry permits a fresh-segment
allocation of 32 MiB minus 256 bytes, including the object's own header.
No wider object pointers or segment size is necessary. Logical values larger
than one physical allocation use chunks; native `int32_t` size-reporting
helpers need attention for values beyond 2 GiB.

A standalone layout probe compiled with the checkout's LLVM 22 build flags
confirmed 25 / 26 / 25 bytes for current / widened / repacked headers. The
following are calculated allocation sizes, rounded to SAL's 128-byte grain:

| User payload | External key bytes | Widened allocation |
|---|---:|---:|
| 2 MiB | 0 | 2,097,280 bytes |
| 2 MiB | 1,024 | 2,098,304 bytes |
| 8 MiB | 1,024 | 8,389,760 bytes |
| 15 MiB | 1,024 | 15,729,792 bytes |

All four examples fit a fresh segment and the proposed 24-bit value field.
The 8 MiB and 15 MiB values roll over whole if needed; exactly 2 MiB remains
on the small-value side of the split-avoidance threshold. Above 16,777,215
payload bytes, this particular direct-record format
must switch to chunked representation or use a wider length field. A chunk
format with a 32-bit payload length can then use almost a full segment per
chunk. Subtract object headers from actual allocatable space; account for
alignment and terminal/sync reserves. Enforce limits before narrowing lengths
or entering allocator paths that cannot report failures safely.

For most external records, the extra header byte does not change rounded
allocation size. At a grain boundary it adds one 128-byte grain. Inline reads
and tree fan-out are unaffected. Exact throughput effects are unmeasured.

Large payloads themselves have costs regardless of field width: inserting or
replacing a value copies its bytes, an owned `get<std::string>` copies them
again, checksum verification hashes them, and compaction relocates whole
objects. Unchanged external values are refcount-shared when pages are copied;
copying a snapshot does not copy every large value. Old snapshots can retain
entire older payloads after replacement. Callback reads avoid the owned-copy
allocation, but hold a residency lock for the callback's duration. Segment-tail
waste and compactor bandwidth warrant measurement at these sizes.

SAL's read-cache promotion currently excludes objects larger than 4 KiB; widening
the value field does not change that policy. This does not prevent writes into
pinned writer segments or normal OS caching of larger objects.

## Persistence compatibility

A direct widening moves the type byte and payload offset, so existing external
records cannot be interpreted as the new structure. Repacking the metadata is
incompatible too, even though its total size stays 25 bytes. Old binaries must
not open new-format data, and new binaries must reject or explicitly decode
old-format records.

SAL's persisted type registry checks registered type IDs, not the byte layout
behind an unchanged ID. Reusing the old `key_val` type ID without an additional
format check would bypass that protection. Choose either an explicit format
revision plus logical migration, or a distinct external-record type with
version-aware readers and lifecycle hooks. The latter requires updating the
external-key resolver, value reads, subtree handling, and refcount/recovery
visitors for both formats. Merely changing `sizeof` is not a migration plan.

## Other profiles and integration limits

- **Small:** its value length is already 32-bit, but the large allocator caps
  the entire object at 64 × 32 KiB = 2 MiB. Leaf metadata, key suffix, and the
  typed-value envelope count against that budget. An exact 2 MiB user payload
  therefore cannot fit one allocation. Support it through profile-specific
  chunking or a separate allocator-capacity change; widening Big alone does
  neither. Existing error text still says 0.5 MB and is stale.
- **Experimental tiering:** the hot tier adds a one-byte data/tombstone envelope
  and stores values through Small (or another selected engine). Widening only
  Big does not raise the composition's effective limit. The flush path copies
  the data into Big again; account for its transient buffers and write traffic.
- **MDBX shim:** the non-GC/non-RC branch currently sets a 524,288-byte object
  budget, subtracting key/header overhead before accepting writes. This hardcoded
  shim check remains a blocker even if the selected native backend accepts
  larger data. Replace it with capacity-aware backend limits when implementing
  the wider support; retain the separate DUPSORT/key restrictions.
- **Native API:** append defaulted offset/length arguments to byte reads while
  preserving existing calls. Callback reads borrow a range within one chunk or
  assemble only the selected bytes when it crosses boundaries, keeping the
  one-callback contract. Native scoped holders need the same owned fallback.
  See the [range-read contract](range-reads.md). Streaming needs a separate
  extension; native signed 32-bit size helpers require an audit for multi-gigabyte
  logical values. Keep payload limits distinct from physical allocation limits.

## Required qualification

Test exact payload lengths 65,535, 65,536, 65,537; 2 MiB − 1, 2 MiB, 2 MiB + 1;
8 MiB; and 15 MiB, plus the 24-bit representation boundary and rejection just
beyond the selected public limit. Also qualify the wider chunk representation
at its actual segment-dependent boundaries, as described in the chunked design.
Exercise empty/short keys and 1,024-byte keys, both external-record variants,
insert/replacement/shrink/remove, explicit and implicit abort, and old snapshots
held across replacements. Read through point lookups, both scan directions,
callbacks, and owning buffers. Include zero and high bytes in payloads.

Then test clean reopen, format rejection/migration, checksums, relocation under
concurrent readers, revision/subtree retention, process interruption around
commit, disk-full behavior, and sustained large-value churn beyond RAM. A
representable length and a layout probe establish feasibility, not those
runtime guarantees. No new storage behavior or performance results were tested
in this assessment.
