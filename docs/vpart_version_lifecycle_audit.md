# vpart version-lifecycle audit (Dan's pause, 2026-06-12)

Trigger: with NO flushing and NO background pipeline (tiered knobs disabled,
pure writer + probes), inserts degrade 164 ns -> 5.3 us by 61M keys and the
arena holds 260 B/key (16 GB for ~6 GB of live tree). The writer is being
blocked by vpart itself. Separately, the codebase has grown TWO parallel
lifecycle mechanisms where the design calls for one.

## 1. What exists today (two mechanisms, neither is the spec'd model)

**A. Published-version pins (readers).** A cursor publishes
`(version<<32|root)` into a registry slot (seq_cst publish-validate), the
drain POLLS all slots to compute a floor = min pinned version, and frees
supersession chains with death <= floor. Slot scan per drain pass; one
parked reader blocks every younger chain; floor computation is O(slots)
polling, not event-driven.

**B. Frozen-root handles (the new tree_root).** shared_ptr-counted handle
over {version, root}; last drop pushes the root to a REAPER thread that
deep-cascades the whole tree. Event-driven and correct, but it bypasses the
version sequence entirely: the reaper frees by reachability (cascade), not
by version, and coexists awkwardly with A (a warm-leg cursor holds BOTH a
pin slot and a handle).

## 2. The spec'd model (Dan)

ONE logical object per published version — the smart_ptr/tree_root
equivalent, not sal's smart_ptr but logically similar:

    version_handle { version seq, root, atomic refcount }

- Every holder (reader cursor, writer's current root, merger's frozen warm,
  crash-slot) owns a ref. Copy = ref++, drop = ref--.
- Commit publishes version v+1 and DROPS the writer's ref on v.
- refcount -> 0 pushes the VERSION (not the root) onto the GC queue.
- GC (one thread, event-fed, in version order): when version v dies and all
  versions < v are already dead+processed, release "everything possible with
  that version gone" — exactly v's supersession chain (the husks superseded
  AT v are visible only to pins < v... corrected: chain v is releasable when
  no LIVE version < v remains). A frozen tree that was never superseded is
  the degenerate case: its handle death releases the whole tree through the
  same queue (the reaper lane becomes a GC case, not a separate mechanism).

Properties: exact, event-driven (no slot polling), one mechanism for hot
versions and frozen trees, and the version sequence gives the floor for free
(a dead-version min-heap / ordered ring instead of slot scans).

## 3. Why the writer degrades ALONE (the no-flush 5 us)

Measured chain of causes, in order of likely contribution:

1. **Drain throughput vs batch-1000 production.** At ~2.5M inserts/s the
   writer produces ~1 chain line per ~31 ops plus husk traffic; the drain is
   one thread doing per-entry `vdrain_entry` + per-run frees + ring pushes.
   The arena's 260 B/key says ~10 GB of chains were waiting at 61M keys: the
   drain LOST the race long before any pin existed. Throughput, not floors.
2. **Arena growth poisons claim_.** claim_ is O(num_segments) under one
   mutex per window claim (most-empty scan, two passes with the fallback
   bar). At 16 GB = 2000 segments; at 26 GB (the tiered 1B attempt) = 3300.
   Once the arena is big, every window refill is us-class — the 4-6 us
   steady-state. Growth and slow claims feed each other.
3. **Recycle-ring capacity.** The writer adopts <= 512 recycled runs per
   txn from one SPSC ring; at batch=1000 the writer consumes lines faster
   than it adopts them back, so even drained memory returns slowly,
   pushing allocation onto fresh segments (growth -> cause 2).

## 4. Proposed rework (pending sign-off — no code until then)

1. **version_handle table** (Section 2) replacing both mechanisms: a small
   ring of {seq, root_raw, atomic<uint32_t> refs} indexed by seq window.
   Reader cursors hold handles (slot publishing retired or kept only as the
   drain's fast floor hint); tree_root becomes an alias of version_handle.
2. **GC unification**: one GC thread consuming the dead-version queue in
   order; per version: release its chain; frozen-tree death = cascade. The
   reaper lane and chain drain merge.
3. **GC throughput**: batch chain processing per pass; the cascade already
   bypasses per-key work; if one thread still trails, shard by segment.
4. **claim_ index**: replace the O(nseg) scan with a small free-space
   ordered index (e.g., 16-bucket histogram of segments by free_groups8 ->
   pop from the emptiest bucket; updated on free/claim) — O(1) claims at
   any arena size.
5. **Ring sizing/adoption**: adopt budget proportional to batch size, or a
   second consumer pass at window refill.

## 5. Unit tests required (Dan)

- version_handle: copy/drop across threads; refcount exactness under
  contention; commit drops the writer's ref on the prior version.
- GC order: version v's chain is NOT released while any handle < v lives;
  released exactly once after; frozen-tree degenerate case.
- Leak gates: free-lines accounting returns to baseline after (a) pure
  churn, (b) freeze/flush/drop cycles, (c) parked-reader release.
- Writer-isolation: batch-1000 insert-only run holds vpart-native ins ns to
  150M+ keys with a bounded arena (THE regression gate for this audit —
  fails today at ~46M).

## 6. Concept extension (LOCKED after Dan's review, 2026-06-12)

psitri's multi-root transaction design REVIEWED AND ADOPTED: ascending-index
lock acquisition enforced by assert (deadlock-free by construction), ONE
tree_handle abstraction for primary/additional roots and subtrees, abort
rolls back all roots, ordered (not atomic) cross-root publish — the concept
documents "ordered durability"; rotation makes the crash window idempotent
by publishing warm before hot.

The exchange primitive (Dan: swap_trees makes no sense):

    tree handle.set_root(tree)   // assign, RETURN the displaced tree

with `tree` the engine's owned-tree value type (psitri::tree exists; heart
has tree values; vpart/part implement as an owning root handle — in vpart
this IS the refcounted version-root: dropping the returned tree feeds the
version GC, unifying set_root with the standalone lifecycle).

Rotation, entirely in concept vocabulary, one multi-root transaction:

    auto h0 = tx.primary();              // hot
    auto h1 = tx.open_root(1);           // warm slot (empty in steady state)
    h1.set_root(h0.set_root(tree{}));    // hot reborn empty; warm = frozen
    tx.commit();                         // publish order: warm, then hot

Enforcement (Dan): tiered is templated on the concept ONLY and its test
suite instantiates heart/heart, vpart/heart, part/heart. vpart passes the
full engine conformance body standalone. No engine customization for tiered.

## 7. Composable subtrees — the generic snapshot story (Dan, locked)

Every engine supports subtree composition as part of the concept. Snapshots
are NOT a mechanism: they are trees stored as values.

    handle.snapshot() -> tree                 // O(1) shared root of the live tree
    handle.upsert_subtree(key, tree)          // store a tree as a value
    handle.is_subtree(key) / cursor.is_subtree()
    handle.open_subtree(key) -> tree_handle   // COW into a stored subtree
    handle.create_subtree(key) -> tree_handle

User snapshot management = upsert_subtree("snapshots/<name>", h.snapshot());
remove(key) deletes a snapshot; iteration lists them. The engine's lifecycle
machinery (refcounts in psitri/heart/part; version GC + version_handle in
vpart) is what makes the shared roots safe — invisible at the API.

Engine status:
- psitri: FULL (invented the model: tree-as-value, open/create_subtree).
- heart:  FULL (subtree-as-value landed: value_type discriminator,
          leaf-backed, ref-cascade clean; cursor is_subtree/subtree_cursor).
- vpart:  MISSING — vhkv has no subtree value kind (cursor is_subtree()
          stubs false). Needs: a subtree-value encoding in the leaf, the
          stamp-GC treating stored roots as references (cascade/share rules
          — the version_handle work is the prerequisite), snapshot() as a
          shared committed root.
- part:   MISSING — same build against refcounts (cheaper: ref++ on store).

Sequencing in #31: version_handle/ordered GC FIRST (subtree sharing rides
it), then the subtree value kind, then snapshot(). Conformance: the engine
test body's subtree cases must pass on vpart/part standalone.

Tiered design note (#32): a subtree value crossing the flush boundary means
the merger materializes a TREE from W into B — both speak the concept, so
the merger recurses: create_subtree in B, replay the W subtree through it.
Cost is proportional to subtree size; fine for v1, revisit if hot.

## 8. SUPERSEDES §7's migration note — snapshot identity, typed values (Dan)

NO copy-construction of subtrees at migration — subtree values share state
under the hood; materializing copies breaks identity and ends badly.

**Use case (blockchain):** block-state snapshots stored in another tree,
indexed by block hash:

    auto blk_root = db.get_root(0);          // snapshot of the state tree
    db.root[1].insert(hash, blk_root);       // a tree stored as a value

**The tiered identity model.** A snapshot taken in the hot layer must
materialize IN ORDER in the base layer. The merger merges up to the first
snapshot boundary, captures the heart tree corresponding to the vpart tree
at that point, and the two acquire SHARED IDENTITY:

    ttree = { vpart_tree_ref + heart_tree_ref }   // active (unmerged) state
          | { resolved heart_tree_ref }            // after its boundary merges

- tiered owns a PERSISTENT INDEX of unique ttree handles (lock-free
  bitset-style allocator, sal-idiom): id -> pair-or-resolved. Refcounted.
- Retained active versions commit in order; at each snapshot boundary the
  merger captures the sal::smart_ptr of the heart root and stores it in the
  registry, dropping the vpart component.
- When the merger encounters a vpart LEAF whose value is a ttree id, it
  looks up the registry and stores the RESOLVED heart identity in its
  place. Identity mapping, never tree copying.
- Ordered materialization makes snapshot-to-snapshot DIFF merging natural
  in vpart: nodes with stamp > S_prev are exactly the S_prev -> S_next
  delta (the stamps ARE the diff index).
- Subtree values are OPAQUE TO VPART: just value bytes written by the
  higher abstraction. vpart needs NO subtree node kind for tiered.

**The API hole (Dan): typed values with lifecycle hooks.** The concept
handles POD; a refcounted object stored as a value needs the engine to call
its lifecycle. Sal-like type registry on the database:

    db.register_type_ops(type_id, retain_fn, release_fn)
    value slot = (T, blob); T = 0 = POD (default, no hooks)

Engines invoke retain at exactly the points subtrees are retained today
(leaf copy/share) and release where subtrees release (leaf destroy: drain /
reap / abort / overwrite). Native subtrees in psitri/heart become the
built-in case of the SAME mechanism; user types (tiered's ttree ids) supply
their own hooks. In vpart the hook points already exist as customization
seams (copy_value / v_destroy in vhkv); the work is the T byte in the value
encoding + routing through the registry — including from the RECLAIM
contexts (drain/reaper call release hooks too).

Sequencing change for #31: the vhkv subtree-value-kind from §7 is NOT
needed; build instead (a) version GC, (b) typed-value registry + hooks,
(c) claim index. tiered (#32) builds the ttree registry + ordered snapshot
materialization on top.

## 9. Public API review: typed values via POD tokens + traits (Dan's sketch)

    value slot  = { typenum, pod bytes }          // pod = trivially copyable token
    storing     : t_pod take_pod(T&&)             // CONSUMES the handle's reference
    reading     : T make_from_pod<T>(const t_pod&, bool retain = true)  // MINTS a ref
    traits      : T::type_id, t_pod::retain(session*, const t_pod&),
                  t_pod::release(session*, const t_pod&)
    registration: db.register_value_types<sal::smart_ptr<X>, user_ref<Y>>()
    engine fires retain on node clone, release on node death (incl. GC threads)

**User-perspective verdict: right shape.** One namespace of storable types;
upsert(key, std::move(handle)) / get<T>(key) is symmetric with bytes values;
take/make names the ownership transfer explicitly (move in = donate your
ref; get out = your own retained ref); traits registration is one
compile-checked line. Subtrees become the natural fallout: a tree handle is
just a registered type whose pod is its root address.

**What generalizes cleanly (subtree APIs subsumed):**
- upsert_subtree(key, tree)      -> upsert(key, std::move(tree))
- get_subtree(key)               -> get<tree>(key)
- is_subtree(key)/cursor.is_subtree() -> value_type() == tree::type_id
  (generic typenum query — ALSO what the tiered merger needs for ttree ids)
- raw smart_ptr<alloc_header> overloads -> the same take_pod path
- recursive destroy: release hooks fire uniformly; the NATIVE tree type's
  release cascades; a user type's release does whatever it owns

(Transparent cursor descent into subtrees STRUCK from this review — it
exists in no engine and is an anti-feature: iterating a snapshots-by-hash
tree must yield one row per snapshot, not explode each snapshot's contents;
key semantics and recursion are unsolvable; typed values cannot support it.
All descent is EXPLICIT: subtree_cursor() to read, open_subtree() to edit.)

**What does NOT generalize (and shouldn't):** the STRUCTURAL composition
layer — open_subtree(key)/create_subtree(key) in-place COW editing and
transparent cursor descent INTO subtrees. Those require the engine to
understand the pod as a tree of its own kind (descend it, COW it, write the
new root back atomically with the parent txn). They remain tree-typed APIs
LAYERED ON TOP of the generic storage: the native tree type is simply an
implicitly-registered type the engine additionally knows structurally.
Nothing breaks — the storage layer they sit on becomes the generic one.

**Contract details to lock:**
1. typenum space: 0 = plain bytes (no hooks, today's fast path); native
   tree type implicitly registered; collisions assert at registration.
2. pod bytes are memcpy'd to an aligned temp before hook calls (no
   alignment contract on in-leaf storage).
3. Reading a value with the WRONG pod type is "INVALID TYPE", never
   "not found" (the key exists — reporting absence lies) and never a
   reinterpret. Three outcomes everywhere: found / not-found / invalid-type.
   Throwing forms (get<T>, cursor.value<T>()) throw a bad-variant-access-
   style invalid_type error; status forms return a distinct code (e.g.
   -2 invalid_type vs -1 not-found). value_type() remains the dispatch
   query for code that legitimately handles multiple types. NOTE: psitri's
   current "subtree values read as not-found" behavior is a deviation to
   correct during its green-up.
4. release may be DEFERRED to GC threads (vpart drain/reaper; sal release
   thread): hooks are thread-safe, non-blocking, take the GC's session*.
5. Overwrite releases the old pod; abort releases pods taken by the dying
   txn (create-log symmetry — same as subtrees today).
6. tiered: W and B must register the same typenums (the merger reads
   typenum in W, writes in B); ttree ids are durable tokens by design.
7. Engines without native sessions pass session* = their GC context;
   types that need no session ignore it.

## 10. any_value — the dynamic read form (Dan, locked)

    database::any_value = { typenum, pod payload (inline), session/read-lock }

Returned by get/find WITHOUT a type argument; the lock-bundling proxy
(heart's pinned_value lesson) generalized to typed values. Three read forms:

    get<T>(key)        -> retained T            | throws invalid_type
    get(key, lambda)   -> scoped view           | lock held for the lambda
    get(key)           -> any_value             | lock held while it lives

cursor.is_type<T>() mirrors holds_alternative (bytes = type 0).
any_value surface: type(), is<T>(), as<T>() (materializes via make_from_pod,
RETAINS — the only civilian exit), bytes() (typenum 0 only; invalid_type
otherwise — a typed pod exposed as bytes would be a forgeable, unbalanced
handle), raw_pod() (expert/merger path: zero-copy identity flow, balanced
because the typed STORE form upsert(key, type, raw_pod) fires the retain
hook — exactly the engine-internal clone-retain semantic).
Contract: holding any_value holds a read lock — scoped inspection, not
storage; engines bundle their native lock type (vpart: the cursor pin;
heart/sal: the session read lock).

## 11. One value path: (typenum, bytes) + traits; psio as the default codec

The unification (Dan): EVERY value flows through the same shape —

    value slot = (typenum, bytes)

- std::string / std::vector<uchar> / std::string_view: built-in storable
  types with NULL retain/release bodies (the trivial registrations). The
  "bytes fast path" stops being a parallel mechanism — it is typenum 0..N
  with no hooks; engines fast-path it internally, same perf as today.
- Refcounted handle types: tiny token pod + real hooks (§9).
- REFLECTED user types: psio/fracpack (pssz) is the DEFAULT CODEC — a
  traits adapter ABOVE the engine (psio_storable<T>: type_id, encode on
  store, psio::view<T> zero-copy on read). If it reflects, it stores:
      tx.upsert(k, my_struct);
      auto v = tx.get<psio::view<MyStruct>>(k);   // zero-copy, lock-scoped
      auto o = tx.get<MyStruct>(k);               // owning decode
- LAYERING: the engine knows ONLY (typenum, bytes) + the hook registry.
  The codec lives above as a traits provider — the shape concept stays
  format-agnostic; no engine ever depends on psio.

any_value REVISED (supersedes the inline-copy wording in §10): it is a
lock-bundled POINTER to the in-tree bytes — {typenum, view into the leaf,
read lock} — never a copy. Zero-copy reads generalize to every type;
as<T>() materializes (decode, or token-copy + retain) only on demand.
psio views are unaligned-safe by design (wire-format access); hook calls
still receive aligned temp copies of small tokens.

Contracts added:
- typenum STABILITY across processes and builds (values persist): ids are
  user-assigned or schema-hashed; the registry asserts collisions and
  engines persist nothing that depends on registration order.
- views are read-only and lock-scoped (the any_value holding rule).

## 12. psio/pssz study notes + the final any_value form (Dan)

Studied ~/gofractally/psio (cpp + pssz-spec). The load-bearing facts:
- pssz: SSZ-derived; adaptive offset widths (pssz_<W>, auto from
  max_encoded_size<T>); CANONICAL deterministic encoding (equal values =
  equal bytes -> content-addressable stored values, snapshot identity /
  dedup for free); single-pass encode; DWNC frozen-layout types collapse to
  memcpy (reflected structs store at raw-bytes cost); validators check
  without decoding (free read-path integrity).
- view<T, Fmt, Store>: one view template per shape, formats plug in via
  view_layout::traits; sub-views share the backing span; and the Store
  STORAGE-POLICY PARAMETER is the lock-bundling seam — the engine supplies
  a read-lock-holding storage policy and view<T, pssz, db_read_lock> over
  the in-leaf span IS the zero-copy typed read; lifetime bound to the lock
  by construction.

FINAL any_value (supersedes §10/§11 composition): the slot already stores
(typenum, bytes), so

    any_value = { const value_slot*, read_lock }

type() reads the typenum THROUGH the pointer; bytes()/raw_pod() are spans
into the slot; as<T>() dispatches on the slot typenum (decode / view /
token-retain). No copies of anything, ever.
