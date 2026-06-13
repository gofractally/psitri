# start_ref multi-root + snapshot — the design (Dan)

start_ref and sal_art_ref (heart) are the SAME root/snapshot model: a node is
referenced by its INDEX, and a tree / snapshot / root handle is an index PLUS
a refcount held on it. The ONLY difference:
- sal_art_ref control block = {refcount, location}  (location = relocation
  indirection; sal can move nodes).
- start_ref control block   = {refcount} only        (arena is address-stable
  — the node never moves, so there is no location to indirect).

So start_ref multi-root is a MECHANICAL PORT of heart's tree / tree_handle /
transaction (already green), substituting start_ref's per-line refcount
(arena add_ref / cow_retain / cow_release, keyed by line index) for sal's
smart_ptr. There is no novel design — copy heart's shapes.

## The one model (mirrors heart::radix tree_handle._working smart_ptr)
- `tree` = {db, line_ptr root, count} holding EXACTLY ONE ref on `root`
  (cow_retain at construct, cow_release at destruct/move). Move-only.
  This is get_root's return, set_root's exchange type, AND the per-root
  working handle inside a transaction.
- The working root HOLDS ITS OWN REF (the wedge in the first attempt was
  pretending it didn't): retain the committed root into the working at txn
  start; on each COW the engine returns a fresh ref-1 new root — ADOPT it
  (no retain) and cow_release the pre-image (the give() pattern,
  [[feedback_psitri_smartptr_pattern]]).
- commit: TRANSFER the working's ref into _live_root[idx] (move, NOT a fresh
  cow_retain) and retire the previous _live_root[idx]. No double-ref.
- abort: drop the working (release diverged), restore _live_root[idx] into the
  working (retain).
- get_root(idx): cow_retain(_live_root[idx]) -> a `tree` (independent ref;
  COW keeps it readable across later commits -> frozen snapshot).
- set_root(t): `tree old = move(working); working = move(t); return old;` —
  pure ref-handle moves, identical to heart::tree_handle::set_root.

## Why the first attempt wedged
start_ref's working slot historically held NO ref (the ref lived only in
_live_root, established by commit's cow_retain). Mixing that with explicit-ref
owned trees made set_root either leak or double-free. The fix is to make the
working slot ref-holding too (heart's model) so every generation has exactly
one owner end-to-end.

## Gate
The drain census (kv-compat "memory drained: N/N") + the bank balance check
catch any ref imbalance. Build against KV_ONLY=start_ref + start_ref-tests'
require_drained, then the full battery.
