# API reference (version 1)

Include `<psitri/api.hpp>` and constrain generic code with `template
<psitri::engine E>`. This page specifies the common byte-value contract of the
Small Engine and Large Engine profiles in their default layouts. A concept checks C++
expressions; the [behavioral tests](../../tests/api/conformance.cpp) check their
meaning. Backend extensions are described separately.

## Names and roles

| Public name | Meaning |
|---|---|
| `psitri::small_engine` | Small Engine: `start_ref::kv::engine` |
| `psitri::big_engine` | Large Engine: `sal_art_ref::radix::engine` |
| `psitri::default_engine` | `big_engine` in API version 1 |
| `psitri::database_for<E>` | `E::database` |
| `psitri::config_for<E>` | `E::config` |
| `psitri::snapshot_for<E>` | Owned tree returned by `E::write_session::get_root` |
| `psitri::open_mode` | `sal::open_mode` |
| `psitri::sync_mode` | `sal::sync_type` |

An engine bundle names `database`, `config`, `write_session`, `read_session`,
`transaction`, and `cursor`. It is a compile-time choice; there is no runtime
engine registry. Returned session handles are `std::shared_ptr`s, but they do
**not** keep the database alive. See the [ownership rules](operations.md).

## Database and sessions

| Expression | Return / behavior |
|---|---|
| `E::database::open(path, mode, cfg)` | `std::shared_ptr<E::database>`; throws on failure |
| `db->start_write_session()` | `std::shared_ptr<E::write_session>` |
| `db->start_read_session()` | `std::shared_ptr<E::read_session>` |
| `ws->start_transaction(root)` | `E::transaction`, scoped read/write work on that root |
| `ws->get_root(root)` | Copyable, owned snapshot of that root's committed state |
| `snapshot.snapshot_cursor()` | `E::cursor` over that frozen state |
| `rs->create_cursor(root)` | `E::cursor` over the committed state at creation |

`create_only` rejects an existing engine database; `open_existing` rejects a
missing database; `create_or_open` does either. All preserve existing content.
`read_only` is available in the Large Engine; the Small Engine rejects it with a
`std::runtime_error` that says it is unsupported, without touching the directory. Never
open another engine's files as a conversion mechanism. Existence checks use
engine marker files and do not provide a universal format detector or an atomic
multi-process create protocol.

Root indices `0..7` form the portable range; the Small Engine has sixteen roots and the Large Engine
has 1,024. A root index is a slot, not a revision number. Use root `0` unless you
need the multi-root extension. Supplying out-of-range indices is a caller error.
A default-constructed empty tree may be assigned with `set_root({})`; it is not
an attached snapshot on which to create a cursor. A tree returned by
`get_root()` is attached and readable even when the root contains no keys.

## Transactions

| Expression | Result |
|---|---|
| `tx.upsert(key, value)` | Insert or replace copied bytes; ignore backend-specific return |
| `tx.remove(key)` | `int`: `0` removed, `-1` absent (not a boolean) |
| `tx.get<std::string>(key)` | `std::optional<std::string>`, including uncommitted writes |
| `tx.get(key, callback)` | `bool`; invokes callback exactly once if found, never if absent |
| `tx.commit()` | Publishes the working root; applies configured sync policy |
| `tx.abort()` | Discards uncommitted work |
| transaction destructor | Aborts unfinished work |

| `tx.snapshot_cursor()` | A cursor on the root as this transaction has changed it so far; a snapshot, like any cursor |
| `tx.remove_range_counted(low, high)` | Removes every key in `[low, high)`, including uncommitted ones; returns how many. An empty `low` is the beginning; an empty `high` leaves the range empty (pass a key above every key to remove to the end). The same in the engines' transactions and in those that keep their changes aside |

Call commit or abort at most once and then destroy the transaction. Do not
retain mutable root handles beyond their transaction. Only one transaction may
be active on a write session. A transaction holds its root's writer until it
commits or aborts; see the [concurrency contract](operations.md#concurrency-contract).

### Transactions that keep their changes aside

`<psitri/transactions.hpp>` adds two transaction types that work on any engine.
Both keep their changes in memory and read through them, so point reads and
cursors in either direction see the transaction's own writes; both have
`save()`, `rollback(savepoint)` and `release(savepoint)` for nested work.

| Type | Root writer | Commit |
|---|---|---|
| `transient_transaction<Snapshot>` | never taken; any number run at once | never; `reset()` discards the changes |
| `delta_transaction<Engine>` | held from construction until commit or abort | applies the changes in key order, then publishes once |

The engine's own transaction changes a private copy of the tree as it goes.
Use it when the work is expected to commit and a transient or delta transaction
when it may well be thrown away. `tests/api/transactions.cpp` drives all three
against a reference map.
Missing keys are ordinary results, not exceptions. Failed allocation, invalid
open modes, or I/O may throw `std::exception` subclasses. The Small Engine
throws `small_engine::storage_full` when a write would exceed `max_segments`
(abort and carry on), `std::invalid_argument` for an oversized key or value,
and `std::logic_error` for a second transaction on a session that has one open.
A node can carry at most 254 references, of which transactions and write
sessions may take it to 190. `commit()` throws `std::runtime_error` if a root
it would publish is at that ceiling (only possible when the transaction was
given a tree by `set_root` that some 190 snapshots also hold); nothing is
published, the transaction stays open, and it can be aborted or committed
again once references are released. Committing a root the transaction did not
change takes no reference. Copying an owned snapshot, or taking a cursor from
it, takes one and throws at 254; point reads through an owned snapshot
(`snapshot.get`) take none. Catch failures outside
the transaction scope so rollback runs during unwinding. A commit failure can
have an uncertain persistence outcome; do not automatically replay a
non-idempotent transaction or continue using that database instance.

## Cursors

Every positioning method below returns `true` when it lands on an entry.
`key()` is valid only on an entry; it returns a borrowed view, invalidated by
cursor movement or destruction. Copy it to retain it.

| Method | Meaning |
|---|---|
| `seek_begin()` / `seek_last()` | Position on the first / last key |
| `lower_bound(key)` | First key **greater than or equal to** the argument |
| `upper_bound(key)` | Last key **less than or equal to** the argument: a floor lookup |
| `next()` / `prev()` | Move forward / backward, returning false at the boundary |
| `is_end()` / `is_rend()` | After the last / before the first entry |
| `value<std::string>()` | Owned optional copy at the current position |
| `get_value(callback)` | Scoped byte view at current position; false at either end |
| `get<std::string>(key)` | Owned optional point read on the cursor's snapshot |
| `get(key, callback)` | Scoped point read on the cursor's snapshot |
| `is_subtree()` | Whether the positioned entry is a native subtree, an extension |

**`upper_bound` is not `std::map::upper_bound`.** For keys `a,c,e`,
`lower_bound("b")` lands on `c`, `upper_bound("b")` on `a`, and
`upper_bound("c")` on `c`. Do not infer cursor movement from point reads;
portable code positions explicitly with bounds before scanning. A newly created
cursor supports point reads immediately; position it before calling `key()`.
Do not rely on a moved cursor retaining its prior position.

Keys compare lexicographically as unsigned bytes, with a prefix before its
extensions. Numeric ordering requires an order-preserving application encoding
(for example, fixed-width unsigned big-endian), not host-endian serialization.
Empty keys and values are supported. There is no built-in collation, integer
encoding, record serializer, prefix index, or duplicate-value convention.

Callbacks receive `std::string_view`, valid only during the callback. Keep them
short; do not mutate the same database, save a view, or wait for a writer from a
callback. `get_value` does not invoke its callback on a native subtree; use
the Large Engine's subtree operations to read that entry. A typed value (below) read
as bytes throws `invalid_type` in both engines. The common API does
not interpret application bytes as backend ownership handles: a value's kind is
the engine's own record of how it was stored, never something read out of the
bytes a caller stored.

The optional [range-read capability](range-reads.md), detected by
`psitri::range_engine<E>`, is implemented by both engines. Use
`get<std::string>(key, offset, length)`, `get(key, callback, offset, length)`,
`value<std::string>(offset, length)`, or `get_value(callback, offset, length)`.
Omitting offset keeps the original full read; omitting length reads the
remainder (`psitri::read_all`). Ranges stop at EOF and preserve the distinction
between an empty result and a missing key. Callback reads borrow the selected
bytes without copying when the range fits one allocation. Both profiles
assemble crossing ranges and invoke the callback once. The Large Engine also supports
`pinned_value(offset, length)` with the same transparent fallback.

## Size and feature boundaries

The common key envelope is at most **1,024 bytes**, with roots **0..7**.
Both maintained profiles support a **4 GiB − 1 byte** logical value limit,
exposed as `Engine::max_value_size`, with allocation tests through **512 MiB**.
Practical database and memory limits may be lower. Oversized values throw
`std::invalid_argument` before input bytes are read or the tree changes. Both engines
check the 1,024-byte key limit before mutation and throw `std::invalid_argument`
for a longer key (`psitri::small_engine::max_key_size` names the limit); reading
or removing such a key is an ordinary miss. The Small Engine uses fixed-address
chunks of at most 2 MiB; the Large Engine uses segment-aware chunks of almost 32 MiB.
See [large-value storage](chunked-values.md) for layouts and ownership.

Native `value_size()` in both profiles and the Large Engine's other size-returning methods
use signed 64-bit results, preserving negative miss/subtree sentinels. The
required concept's owning, callback, and remove return types are unchanged.
Each profile embeds its continuation directory and first payload in its head;
their physical formats differ. Pin an engine and compatible format when opening
persisted data.

`adb::multi_root_transaction` adds `primary()` and `open_root(index)` editors,
with `upsert`, `remove`, `get`, and `set_root(tree)`. The latter assigns a tree
and returns the displaced owned tree. Open distinct additional roots in
ascending order. Keep all trees in the same database. Publication across root
slots is **not promised atomic**; neither publication order nor savepoints
spanning roots is a common guarantee. For atomic revision manifests, put
related subtrees beneath a single Large Engine root.

On the Small Engine, concretely: the roots of one transaction are published
one after another, in ascending index order. Each root on its own always
shows a committed state. A reader that takes cursors on two roots from a read
session can get the first root as of this commit and the second as of the
previous one, and whole commits can fall between any two cursors. Two owned
snapshots from `get_root` are never torn that way within one commit, but
other commits can still land between the two calls. After an unclean stop the
root slots can likewise be a mix of the last commit and the one before. Data
that must change together belongs under one root, distinguished by key prefix.

Native subtrees, detached editors, and savepoints are Large Engine extensions.
Registered typed values and typed test adapters are outside the common byte
contract. Detect optional capabilities
explicitly; never assume they follow from `psitri::engine<E>`.

### Typed values in the Large Engine

A typed value is a type number (1 to 65,534) and a token of at most 256 bytes,
for a type that registered `retain` and `release` callbacks in
`sal_art_ref::g_value_types` before any value of it is stored. It is written
with `upsert_typed(key, type_number, token)` on a transaction, a tree handle or a
detached write transaction, and read with `get_any(key, fn)` or a cursor's
`value_any(fn)` / `value_typenum()`, where `fn(type_number, bytes)` receives the
type number (zero for plain bytes) and the token or the bytes. Every byte read
(`get`, `value`, `get_value`, `pinned_value`) of a typed value throws
`sal_art_ref::invalid_type`.

- The kind is stored by the engine beside the value. Plain bytes are stored as
  given and never inspected, whatever they begin with.
- Storing hands the engine one reference of the token's owner: `upsert_typed`
  does not call `retain`. If it throws, nothing was stored and the reference is
  still the caller's.
- `retain` runs when the stored bytes are physically duplicated (a leaf page is
  copied because a snapshot shares it, or split); `release` runs when a copy
  dies (overwritten, removed, aborted, or its page or record destroyed). Either
  may run on the engine's background threads: they must be thread-safe and must
  not block or call back into the database.
- Opening and recovering a database calls neither.
- Typed values need the default page layout, and do not pass through the
  kept-aside change stores of `psitri::transactions`: those carry bytes only.
