#pragma once
#include <sal/allocator_session.hpp>
#include <sal/smart_ptr.hpp>
#include <atomic>
#include <cassert>

namespace sal
{
   /**
    * @brief A lock-free, thread-safe shared slot for smart_ptr — "global_ptr".
    *
    * Any thread may store into it, any thread may load from it, and a refcount
    * owned by one thread is transferred to another with a guarantee that no
    * count is ever lost or duplicated. This is the cross-thread handoff
    * primitive: sessions stay single-owner; an atomic_ptr is how a reference
    * crosses between them.
    *
    *     sal::atomic_ptr<my_node> global;          // shared slot
    *     // thread A (its own session):
    *     global.store(ses_a->smart_alloc<my_node>(...));   // publish
    *     // thread B (its own session):
    *     sal::smart_ptr<my_node> p = global.load(ses_b);   // p owns a ref
    *
    * ── THE WORD ────────────────────────────────────────────────────────────
    *
    *     [ addr : 32 | epoch : 16 | count : 16 ]   one std::atomic<uint64_t>
    *
    *   addr   the ptr_address currently published (null_ptr_address if empty).
    *   count  readers "in flight": they have claimed the address but not yet
    *          settled their reference (see protocol). count sits in the low
    *          bits so a claim is a single fetch_add(1).
    *   epoch  bumped by every store/exchange so a stale reader's settle can
    *          never succeed against a *different generation* that happens to
    *          republish the same address (the recycled-address ABA).
    *
    * ── WHY A PLAIN { load; retain; } IS BROKEN ─────────────────────────────
    * Loading the address and retaining it are two steps. Between them a writer
    * can swap the slot and drop the old object's last reference — the retain
    * then lands on freed (or recycled) memory. Every correct design closes the
    * gap by making the SLOT itself hold claims that readers take in one RMW.
    * This is external (split) reference counting — the same family as
    * folly::atomic_shared_ptr — simplified by two sal properties: addresses
    * are 32-bit (the whole triple fits one u64), and control-block retain is
    * already a sessionless relaxed fetch_add any thread may perform.
    *
    * ── THE PROTOCOL ────────────────────────────────────────────────────────
    * Invariant: while the slot publishes addr A, the slot itself owns exactly
    * ONE strong reference on A, and `count` equals the number of claims taken
    * against this (addr, epoch) generation that have not yet settled.
    *
    * load(ses):
    *   1. CLAIM:  w = word.fetch_add(1)  — one RMW yields the address AND a
    *      transient claim. From this instant A cannot die (writer side below).
    *   2. RETAIN: ses->retain(A) — safe under the claim's protection.
    *   3. SETTLE: CAS {A, e, c} -> {A, e, c-1} to withdraw the claim.
    *      - Success: done — the retain from step 2 is the owned reference.
    *      - The slot changed (addr or epoch differs): a writer harvested this
    *        claim and ALREADY credited one strong reference on A for us
    *        (writer step 2). We now hold two (credit + our retain), so give
    *        one back — safe, the count is >= 2 so this cannot free — and own
    *        the remainder.
    *      Either way the caller ends with exactly one owned reference.
    *
    * store/exchange(p):
    *   1. PRE-CREDIT: read the word {A, e, C}; retain(A) C times — one credit
    *      per in-flight reader. The credits MUST be on A's strong count before
    *      the publish lands: a reader observes "slot changed" the instant the
    *      CAS below succeeds, balances its own retain with a release, and
    *      walks away owning the credit — if the credit were added after the
    *      publish (harvest-after), the reader's reference could be the last,
    *      drop the count to zero, and free A under the writer's late retain.
    *      Skipped when A is null (claims against an empty slot are vacuous).
    *   2. PUBLISH: CAS the word {A, e, C} -> {p.addr, e+1, 0}, donating p's
    *      reference to the slot. The CAS compares the FULL word, so it fails
    *      if any new claim arrived since step 1 — then return the exact
    *      pre-credits taken (releasing counts one owns is always sound, even
    *      to zero; retaining freed memory never is) and retry from a fresh
    *      read. (A CAS loop, not a blind exchange, also because epoch must be
    *      read-modify-written to stay monotonic.)
    *   3. The slot's own reference on old A is returned to the caller as a
    *      smart_ptr (exchange) or released (store).
    *
    * ── WHY NO COUNT IS LOST (the accounting) ───────────────────────────────
    * All word operations are RMWs on one atomic, so they are totally ordered.
    * Between two publishes, count = (#claims) - (#fast settles): exactly the
    * readers still in flight. A fast-settling reader removed its own unit and
    * self-retained; a reader that misses the window is included in C and is
    * credited by the harvest. Each in-flight reader consumes exactly one
    * credit (it observes the slot changed and skips its own settle). The
    * transient overshoot on A's strong count is bounded by the number of
    * concurrently in-flight readers — the same slack the control block
    * already reserves (max_ref_count = 2^18 - max_threads).
    *
    * ── EPOCH SUFFICIENCY ───────────────────────────────────────────────────
    * A settle compares the full 64-bit word, so it can only succeed against
    * the same (addr, epoch, count) triple it claimed. Republishing even the
    * SAME address bumps the epoch, forcing stale settles onto the (correct)
    * credit path. The 16-bit epoch would have to wrap to the same value — with
    * the same address republished and the same count in flight — within one
    * reader's claim-to-settle window (a handful of instructions) to be
    * confused: 65,536 publishes to one slot inside that window. If that ever
    * keeps anyone up at night, the word widens to a 128-bit CAS; the protocol
    * is unchanged.
    *
    * ── USAGE CONTRACT ──────────────────────────────────────────────────────
    * - The slot transfers LIFETIME, not exclusivity. Objects published here
    *   are visible to other threads: the publisher must be done mutating them
    *   (committed / logically read-only), or readers race on the bytes. Not
    *   asserted — is_read_only() tracks segment write-protection, which is
    *   legitimately off in many configs (write_protect_on_commit = false).
    *   Lifetime is what this class manages; to read the bytes, dereference the
    *   returned smart_ptr through your own session as usual (read_lock pins
    *   residency against the compactor exactly as for any smart_ptr).
    * - Every operation takes/uses a session: load(ses) binds the result to the
    *   caller's session; store/exchange use the session already inside the
    *   incoming smart_ptr (asserted) to release/return the previous value.
    * - reset(ses) before the atomic_ptr is destroyed (or destroy it empty).
    *   The destructor asserts emptiness in debug: it has no session to
    *   release a leftover value with, and an atomic_ptr must not outlive its
    *   allocator anyway. In release, a non-empty slot's reference leaks.
    * - Destroying the atomic_ptr concurrently with load/store on it is UB,
    *   like destroying any object in use.
    */
   template <typename T = alloc_header>
   class atomic_ptr
   {
      // [ addr:32 | epoch:16 | count:16 ] — count in the low bits: claim = +1.
      static constexpr uint64_t count_one  = 1;
      static constexpr uint64_t count_mask = 0xFFFFull;
      static constexpr uint64_t epoch_one  = 1ull << 16;
      static constexpr uint64_t slot_mask  = ~count_mask;  // addr+epoch: the generation

      static constexpr uint64_t pack(ptr_address a, uint64_t epoch_and_count) noexcept
      {
         return (uint64_t(*a) << 32) | epoch_and_count;
      }
      static constexpr ptr_address addr_of(uint64_t w) noexcept
      {
         return ptr_address(uint32_t(w >> 32));
      }

      mutable std::atomic<uint64_t> _w{pack(null_ptr_address, 0)};

     public:
      atomic_ptr() noexcept = default;

      /// Adopt an initial value: p's reference is donated to the slot.
      /// (No session needed — there is no previous value to settle.)
      explicit atomic_ptr(smart_ptr<T> p) noexcept : _w(pack(p.address(), 0)) { p.take(); }

      atomic_ptr(const atomic_ptr&)            = delete;
      atomic_ptr(atomic_ptr&&)                 = delete;
      atomic_ptr& operator=(const atomic_ptr&) = delete;
      atomic_ptr& operator=(atomic_ptr&&)      = delete;

      ~atomic_ptr() noexcept
      {
         // Must be reset() (or never set) before destruction — see usage
         // contract. A live value here has no session to be released with.
         assert(addr_of(_w.load(std::memory_order_relaxed)) == null_ptr_address &&
                "atomic_ptr destroyed while holding a value — reset(ses) first");
         assert((_w.load(std::memory_order_relaxed) & count_mask) == 0 &&
                "atomic_ptr destroyed with readers in flight");
      }

      /// Acquire a reference to the current value, bound to the CALLER's
      /// session. Returns an empty smart_ptr if the slot is empty. Lock-free;
      /// never blocks a concurrent store.
      [[nodiscard]] smart_ptr<T> load(const allocator_session_ptr& ses) const noexcept
      {
         // CLAIM — one RMW: address + transient protection.
         const uint64_t w = _w.fetch_add(count_one, std::memory_order_acquire);
         assert((w & count_mask) != count_mask && "atomic_ptr claim count saturated");
         const ptr_address a = addr_of(w);

         if (a == null_ptr_address)
         {
            // Withdraw the (vacuous) claim. If the generation moved first, the
            // publisher skipped the harvest for null — nothing to undo.
            uint64_t cur = w + count_one;
            while ((cur & slot_mask) == (w & slot_mask) &&
                   !_w.compare_exchange_weak(cur, cur - count_one, std::memory_order_acq_rel,
                                             std::memory_order_acquire))
            {
            }
            return smart_ptr<T>();
         }

         // RETAIN under the claim's protection, then SETTLE.
         ses->retain(a);
         uint64_t cur = w + count_one;
         for (;;)
         {
            if ((cur & slot_mask) != (w & slot_mask))
            {
               // The slot changed: the publisher PRE-credited one reference on
               // `a` for this claim before its publish could be observed, so we
               // hold credit + our own retain — give one back (the count is
               // >= 2: credit and retain both landed before this release).
               ses->release(a);
               return smart_ptr<T>(ses, a, false);
            }
            if (_w.compare_exchange_weak(cur, cur - count_one, std::memory_order_acq_rel,
                                         std::memory_order_acquire))
               return smart_ptr<T>(ses, a, false);  // fast path: our retain is the ref
         }
      }

      /// Publish `p` (donating its reference) and return the PREVIOUS value as
      /// an owned smart_ptr bound to p's session (empty smart_ptr if the slot
      /// was empty). p must carry a session even when null — it is needed to
      /// credit in-flight readers of the previous value.
      [[nodiscard]] smart_ptr<T> exchange(smart_ptr<T> p) noexcept
      {
         assert(p.session().get() != nullptr &&
                "exchange/store needs a session to settle the previous value");
         allocator_session_ptr ses = p.session();

         constexpr uint64_t epoch_mask = count_mask << 16;
         uint64_t           w          = _w.load(std::memory_order_acquire);
         for (;;)
         {
            const ptr_address old_a = addr_of(w);
            const uint64_t    c     = w & count_mask;
            // PRE-CREDIT the in-flight readers (see protocol: the credits must
            // be on the strong count BEFORE the publish is observable).
            if (old_a != null_ptr_address)
               for (uint64_t i = 0; i < c; ++i)
                  ses->retain(old_a);
            if (_w.compare_exchange_strong(w, pack(p.address(), (w + epoch_one) & epoch_mask),
                                           std::memory_order_acq_rel, std::memory_order_acquire))
            {
               p.take();  // the slot owns the new value now
               // The slot's reference on the old value goes to the caller.
               return smart_ptr<T>(ses, old_a, false);
            }
            // Lost the race (a claim arrived or another writer published):
            // return the exact pre-credits taken and retry from the fresh `w`.
            // Releasing counts we own is sound even if one of them is the last.
            if (old_a != null_ptr_address)
               for (uint64_t i = 0; i < c; ++i)
                  ses->release(old_a);
         }
      }

      /// Publish `p`, releasing the previous value through p's session.
      void store(smart_ptr<T> p) noexcept { (void)exchange(std::move(p)); }

      /// Empty the slot, releasing the previous value through `ses`.
      void reset(const allocator_session_ptr& ses) noexcept
      {
         (void)exchange(smart_ptr<T>(ses, null_ptr_address, false));
      }

      /// Racy peek at the published address — diagnostics only. The value may
      /// be stale before this returns; it confers NO protection. Use load().
      [[nodiscard]] ptr_address address_hint() const noexcept
      {
         return addr_of(_w.load(std::memory_order_relaxed));
      }
   };

}  // namespace sal
