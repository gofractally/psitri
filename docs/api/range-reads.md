# Value ranges and borrowed-or-owned reads

Small and Big implement offset/length byte reads over contiguous and chunked
values. Both transparently assemble ranges that cross chunk boundaries.
The optional `psitri::range_engine` concept detects ranges without changing
`psitri::engine` or the core adb concepts.

## Calls

```cpp
auto whole = tx.get<std::string>(key);
auto suffix = tx.get<std::string>(key, offset);
auto part = tx.get<std::string>(key, offset, length);
bool found = tx.get(key, [](std::string_view bytes) {
    consume(bytes); // valid only during this call
}, offset, length);

auto current = cursor.value<std::string>(offset, length);
bool positioned = cursor.get_value([](std::string_view bytes) {
    consume(bytes);
}, offset, length);

// Big's native holder also supports ranges.
auto held = cursor.pinned_value(offset, length);
consume(held.view()); // valid while held lives
```

Offset defaults to zero, and length defaults to `psitri::read_all` (also
`adb::read_all`, equal to `UINT64_MAX`). Zero length selects an empty range.
Use nonnegative byte offsets. Owning point reads constrain offset to an integral
argument, so `get(key, 0)` selects a range, not the native output-pointer overload.
Callback arguments must be callable with `std::string_view`.

The same semantics apply to transactions, cursor point and positioned reads,
and native tree/root/editor convenience methods. Small's registered typed-object
decoding is separate from this byte API. Engines without ranges continue to
satisfy their original concepts; select capabilities using `if constexpr`.

Both profiles provide native range traversal, with no virtual dispatch, type
erasure, or runtime capability adapter. Contiguous values still borrow/copy
directly; only fragmented values need directory traversal and assembly.

## Range semantics

For an existing byte value of size `V`:

```text
available = offset < V ? V - offset : 0
count = min(length, available)
```

Subtraction avoids overflow from adding offset and length. Ranges exclude
headers, keys, and allocation padding. A range starting at a chunk boundary
starts in the next chunk; one ending there excludes the next chunk.

| Situation | Owning read | Callback read |
|---|---|---|
| Present, nonempty range | Engaged optional with exactly `count` bytes | `true`, one callback |
| Present, empty value or zero length | Engaged empty string | `true`, one empty callback |
| Present, offset at or beyond EOF | Engaged empty string | `true`, one empty callback |
| Missing key | `nullopt` | `false`, no callback |
| Cursor at end/reverse-end | `nullopt` from `value()` | `false` from `get_value()` |

Reads extending beyond EOF return a short result. Subtree/non-byte behavior is
unchanged. Callback and allocation exceptions propagate; callbacks are never
retried. A full read or large requested range still needs enough contiguous
output memory. This is not a streaming API.

## Borrowing and boundary fallback

| Selected range | Behavior |
|---|---|
| Fits in one record/chunk | Borrow a subview under residency protection; no payload copy |
| Crosses a chunk boundary | Assemble exactly `count` bytes and invoke the callback once |
| Empty | Return an empty view without visiting payload chunks |

A 4 KiB crossing range uses a 4 KiB payload buffer regardless of total value size.
Both binary-search continuation starts. Big copies the needed metadata while
pinned, and protects each intersecting chunk separately while copying. The
owning root retains the same logical version throughout. Assembly-only residency
locks are released before invoking a callback with owned bytes. An outer lock
explicitly held by the caller remains the caller's responsibility. Small uses
direct arena indices; the cursor's snapshot or write transaction keeps its
fixed allocations alive, so no residency lock is needed.

Use owning `std::string` results or callbacks for portable reads. Small's native
`get<std::string_view>()` and `value<std::string_view>()` can borrow within one
chunk, but throw `std::invalid_argument` for crossing ranges: a bare view cannot
own the fallback buffer. Small's `get_any().bytes(offset, length)` keeps that
buffer in the `any_value` holder until its next byte read or destruction. Keep
the holder alive and its transaction valid; a view from a temporary holder is
unsafe. The callback API handles both paths transparently.

Big's `value_ref` preserves `view()`, `data()`, `size()`, `empty()`, `str()`, and
implicit conversion to the value view. `is_borrowed()` reports which path was
used. Borrowed holders should remain short-lived because they delay reclamation.
Owned holders share their assembly buffer on copy and hold no residency lock.
Both preserve their view across copy/move. A view extracted from a temporary
holder expires with it. An invalid/subtree position returns an empty holder;
use optional/callback reads when the found/missing distinction matters.

The native `value(read_lock&)` overload borrows contiguous storage under
the caller's lock. For a crossing value it uses cursor-owned scratch, valid until
the next raw-view read or cursor destruction. Direct-tree view reads use
tree-owned scratch with the same next-read restriction. Prefer the portable
callback/owning forms or `pinned_value()` for explicit lifetime management.

## Compatibility and tests

Ordinary calls keep full-value defaults, owning/callback result types, one-callback
behavior, and missing/empty semantics. The required concepts are unchanged.
Native `value_size()` in both profiles and Big size-returning methods use
signed 64-bit results; code naming exact
member-function pointer signatures must be updated. Defaulted range parameters
also mean that exact native member-function types are not frozen. This is a
source API, not a binary ABI or a promise that holder layout never changes.

The former 65,536-byte rejection test now verifies a successful round trip.
The [shared conformance tests](../../tests/api/conformance.cpp) cover ranges on
both profiles. The [Big](../../tests/api/large_values.cpp) and
[Small](../../tests/api/small_large_values.cpp) large-value tests cover
whole values through 512 MiB, every actual chunk boundary, callback counts,
borrowed/owned holders, exceptions, and old snapshots. Big additionally tests
relocation under reads; Small tests saturated refcounts and allocation failures.
Storage placement, capacity, and qualification limits are documented in
[large values](chunked-values.md).
