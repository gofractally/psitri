#pragma once
#include <sal/allocator_session.hpp>
#include <sal/smart_ptr.hpp>
#include <atomic>
#include <cassert>

namespace sal
{
   /**
    * @brief A lock-free shared slot for smart_ptr — "global_ptr".
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
    *     [ addr : 32 | lock : 1 | epoch : 15 | consumed : 16 ]   one atomic u64
    *
    *   addr      the ptr_address currently published (null_ptr_address = empty).
    *   consumed  how many of the slot's PRE-PAID credits readers have taken.
    *             Low bits, so a load's claim is a single fetch_add(1).
    *   epoch     bumped by every publish; distinguishes generations for the
    *             empty-slot count fixup (and diagnostics).
    *   lock      serializes the SLOW paths only (publish, credit re-arm).
    *             Loads on the fast path never take it.
    *
    * ── WHY A PLAIN { load addr; retain addr; } IS BROKEN ───────────────────
    * Loading the address and retaining it are two steps. Between them a writer
    * can swap the slot and drop the old object's last reference — the retain
    * then lands on freed (or recycled — the ABA class) memory. Every correct
    * design closes the gap by making the SLOT hand out references in ONE RMW.
    * This is external (split) reference counting, the same architecture as
    * folly::atomic_shared_ptr, simplified by two sal properties: addresses are
    * 32 bits (the whole state fits one u64) and control-block retain/release
    * are raw atomics any thread may perform through its own session.
    *
    * ── THE PROTOCOL: PRE-PAID CREDITS ──────────────────────────────────────
    * Invariant (slot unlocked, addr = A ≠ null):
    *
    *     strong(A)  >=  1 (the slot's own base ref)  +  (bias − consumed)
    *
    * Publishing BANKS `bias` extra references on A up front (the credits).
    * A load claims one credit with a single fetch_add on `consumed` — the
    * reference it walks away with was already on A's strong count, so there
    * is no load-then-retain window at all: loads are WAIT-FREE, one RMW, and
    * never touch the word again (no ABA exposure on the read side).
    *
    * A claim is VOID — confers nothing, claimant must retry — when its
    * fetch_add's PRE-value had the lock bit set, or consumed >= bias, or a
    * null address. Void +1s physically land in the count field, but every
    * void claimant OBSERVED its own voidness (its pre-value says so), and the
    * count is rewritten absolutely by the next lock holder, so they are
    * harmless: physical count >= bias always means "real consumption = bias,
    * remainder void".
    *
    * Slow paths take the lock bit (writer-vs-writer only; loads in the fast
    * path are unaffected):
    *
    * publish (store/exchange/reset), under the lock:
    *   1. Settle the OLD value: real consumed C = min(count, bias); release
    *      the (bias − C) unconsumed credits. Safe: the slot still holds the
    *      base ref, so these releases cannot free A early; they are counts the
    *      slot itself banked.
    *   2. Arm the NEW value: bank `bias` credits on it BEFORE it is exposed.
    *   3. Expose: one release-store of {new_addr, epoch+1, consumed = 0,
    *      unlocked}. p's reference is donated as the new base ref.
    *   4. The old base ref is returned to the caller (exchange) or released
    *      (store/reset).
    *
    * credit re-arm (a load found consumed >= bias), under the lock:
    *   Bank `bias` fresh credits on A, then store {A, epoch, consumed = 1,
    *   unlocked} — the re-arming reader takes credit #1. The banking is safe
    *   because the base ref can only be surrendered by a publish, publishes
    *   need the lock, and we hold it: A is pinned for the duration. (This
    *   pin-under-lock is what makes re-arm sound; without the lock, banking
    *   races a concurrent publish's teardown — a retain on freed memory.)
    *
    * ── WHY NO COUNT IS LOST (the accounting) ───────────────────────────────
    * Credits move in only three ways: banked under the lock (+bias), consumed
    * by a fast load (one fetch_add, observed valid), or released as
    * unconsumed remainder at publish (bias − C). Fast loads keep exactly the
    * one credit they consumed; void claims keep nothing and are excluded from
    * C by the min(count, bias) rule. The slot's transient contribution to an
    * object's strong count is bounded by bias + 1 — budget against the
    * control block's 18-bit field (max_ref_count = 2^18 − max_threads) when
    * sizing how many slots may publish the same object.
    *
    * ── COSTS / PROGRESS ────────────────────────────────────────────────────
    *   load fast path :  1 RMW, wait-free. Never blocks on writers.
    *   load slow path :  every bias-th load of an unchanging value re-arms
    *                     under the lock (~bias retains).
    *   publish        :  lock + ~bias retains on new + up to bias releases on
    *                     old. Deliberate trade: stores pay O(bias) relaxed
    *                     RMWs so that loads are a single RMW — right for
    *                     read-hot slots (roots, config, handoff points).
    *   Claims that land while the lock is held spin-retry; lock sections are
    *   short and bounded (no loads, no allocation inside).
    *
    * ── USAGE CONTRACT ──────────────────────────────────────────────────────
    * - The slot transfers LIFETIME, not exclusivity. Published objects are
    *   visible to other threads: the publisher must be done mutating them
    *   (committed / logically read-only), or readers race on the bytes. Not
    *   asserted — is_read_only() tracks segment write-protection, which is
    *   legitimately off in many configs (write_protect_on_commit = false).
    *   To read the bytes, dereference the returned smart_ptr through your own
    *   session as usual (read_lock pins residency against the compactor
    *   exactly as for any smart_ptr).
    * - Every operation goes through a session: load(ses) binds the result to
    *   the CALLER's session; store/exchange/reset use the session inside the
    *   incoming smart_ptr (asserted non-null) for the credit accounting.
    * - reset(ses) before destruction (or destroy it never-set): the
    *   destructor has no session to settle a leftover value with, and an
    *   atomic_ptr must not outlive its allocator anyway. Asserted in debug;
    *   in release a non-empty slot's references leak.
    * - Destroying the atomic_ptr concurrently with operations on it is UB,
    *   like destroying any object in use.
    */
   template <typename T = alloc_header>
   class atomic_ptr
   {
     public:
      /// Credits banked per published value. Each slot transiently adds up to
      /// bias + 1 to its object's strong count (18-bit budget, see class doc).
      static constexpr uint32_t credit_bias = 256;

     private:
      // [ addr:32 | lock:1 | epoch:15 | consumed:16 ]
      static constexpr uint64_t count_one  = 1;
      static constexpr uint64_t count_mask = 0xFFFFull;
      static constexpr uint64_t epoch_one  = 1ull << 16;
      static constexpr uint64_t epoch_mask = 0x7FFFull << 16;
      static constexpr uint64_t lock_bit   = 1ull << 31;

      static constexpr uint64_t pack(ptr_address a, uint64_t epoch_count_lock) noexcept
      {
         return (uint64_t(*a) << 32) | epoch_count_lock;
      }
      static constexpr ptr_address addr_of(uint64_t w) noexcept
      {
         return ptr_address(uint32_t(w >> 32));
      }
      static constexpr uint64_t next_epoch(uint64_t w) noexcept
      {
         return (w + epoch_one) & epoch_mask;
      }

      mutable std::atomic<uint64_t> _w{pack(null_ptr_address, 0)};

      /// Acquire the slow-path lock; returns the pre-lock word (its count is
      /// the authoritative consumption snapshot — claims landing after the
      /// successful CAS observe the lock bit and void themselves).
      uint64_t lock_() const noexcept
      {
         // fetch_or, NOT a CAS loop: reader claims fetch_add this word
         // continuously, so a CAS acquirer can be starved indefinitely
         // (measured: the MT hammer livelocked — writers never won the lock,
         // readers never saw writers_left drop). fetch_or wins immediately
         // and atomically returns the authoritative pre-lock snapshot; if
         // another locker beat us, spin on plain loads and re-or.
         for (;;)
         {
            const uint64_t w = _w.fetch_or(lock_bit, std::memory_order_acq_rel);
            if (!(w & lock_bit))
               return w;  // we own the lock; w is the pre-lock word
            uint64_t s;
            do
               s = _w.load(std::memory_order_acquire);
            while (s & lock_bit);
         }
      }

     public:
      atomic_ptr() noexcept = default;

      /// Adopt an initial value: p's reference becomes the base ref, and
      /// `credit_bias` credits are banked through p's own session — BEFORE the
      /// address is published, so a claim can never consume an unbanked credit.
      explicit atomic_ptr(smart_ptr<T> p) noexcept : _w(pack(null_ptr_address, 0))
      {
         if (p.is_valid())
         {
            const allocator_session_ptr& ses = p.session();
            for (uint32_t i = 0; i < credit_bias; ++i)
               ses->retain(p.address());
         }
         _w.store(pack(p.address(), 0), std::memory_order_release);
         p.take();
      }

      atomic_ptr(const atomic_ptr&)            = delete;
      atomic_ptr(atomic_ptr&&)                 = delete;
      atomic_ptr& operator=(const atomic_ptr&) = delete;
      atomic_ptr& operator=(atomic_ptr&&)      = delete;

      ~atomic_ptr() noexcept
      {
         // Must be reset() (or never set) before destruction — see usage
         // contract: there is no session here to settle a leftover value with.
         assert(addr_of(_w.load(std::memory_order_relaxed)) == null_ptr_address &&
                "atomic_ptr destroyed while holding a value — reset(ses) first");
      }

      /// Acquire a reference to the current value, bound to the CALLER's
      /// session; empty smart_ptr if the slot is empty. Fast path: one RMW,
      /// wait-free — the fetch_add consumes a pre-paid credit that is already
      /// on the object's strong count.
      [[nodiscard]] smart_ptr<T> load(const allocator_session_ptr& ses) const noexcept
      {
         for (;;)
         {
            const uint64_t    w = _w.fetch_add(count_one, std::memory_order_acquire);
            const ptr_address a = addr_of(w);

            if (a == null_ptr_address)
            {
               // Vacuous claim on an empty slot. Withdraw it so empty-load
               // storms cannot carry the 16-bit count into the epoch bits; if
               // the generation moves (or a lock holder rewrote the count
               // absolutely — count 0 means our +1 is already erased), it is
               // gone already and there is nothing to undo.
               uint64_t cur = w + count_one;
               while ((cur & ~count_mask) == (w & ~count_mask) && (cur & count_mask) != 0 &&
                      !_w.compare_exchange_weak(cur, cur - count_one, std::memory_order_relaxed,
                                                std::memory_order_relaxed))
               {
               }
               return smart_ptr<T>();
            }

            if (!(w & lock_bit) && (w & count_mask) < credit_bias) [[likely]]
               return smart_ptr<T>(ses, a, false);  // consumed credit = our ref

            if (w & lock_bit)
            {
               // Void claim during a slow path. Spin on plain LOADS until the
               // lock clears before re-claiming — re-claiming while locked
               // would pump void +1s into the count (and, sustained for a
               // whole lock section, overflow it into the epoch bits). This
               // bounds voids to ~one per reader per lock section.
               uint64_t s;
               do
                  s = _w.load(std::memory_order_acquire);
               while (s & lock_bit);
               continue;
            }

            // Credits exhausted: re-arm under the lock. The base ref pins the
            // published object for as long as we hold the lock (publishes also
            // need it).
            const uint64_t    lw = lock_();
            const ptr_address la = addr_of(lw);
            // RE-VALIDATE on the locked snapshot: another re-arm or a publish
            // may have won the race. Re-arming on anything but a genuinely
            // exhausted bank would store count = 1 and ERASE the real
            // consumption record — the next publish would then over-release
            // the "unconsumed" remainder and free the object under its
            // readers' credits.
            if (la == null_ptr_address || (lw & count_mask) < credit_bias)
            {
               _w.store(lw, std::memory_order_release);  // unlock, preserving valid claims
               continue;                                 // retry with a fresh claim
            }
            for (uint32_t i = 0; i < credit_bias; ++i)
               ses->retain(la);
            // consumed = 1: this load takes credit #1 of the fresh bank. The
            // old bank is fully consumed (count >= bias; the excess is void
            // claims) — those credits are now their readers' own references.
            _w.store(pack(la, (lw & epoch_mask) | count_one), std::memory_order_release);
            return smart_ptr<T>(ses, la, false);
         }
      }

      /// Publish `p` (donating its reference) and return the PREVIOUS value as
      /// an owned smart_ptr bound to p's session (empty if the slot was
      /// empty). p must carry a session even when null — the credit
      /// accounting for the previous value needs one.
      [[nodiscard]] smart_ptr<T> exchange(smart_ptr<T> p) noexcept
      {
         assert(p.session().get() != nullptr &&
                "exchange/store/reset needs a session for credit accounting");
         allocator_session_ptr ses = p.session();

         const uint64_t    w     = lock_();
         const ptr_address old_a = addr_of(w);

         // Settle the old value's credits: real consumption is capped at bias
         // (anything beyond is void claims); release the unconsumed remainder.
         // The slot's base ref is still in place, so this cannot free old_a.
         if (old_a != null_ptr_address)
         {
            const uint64_t consumed =
                (w & count_mask) < credit_bias ? (w & count_mask) : credit_bias;
            for (uint64_t i = consumed; i < credit_bias; ++i)
               ses->release(old_a);
         }
         // Arm the new value BEFORE exposing it.
         if (p.is_valid())
            for (uint32_t i = 0; i < credit_bias; ++i)
               ses->retain(p.address());

         _w.store(pack(p.address(), next_epoch(w)), std::memory_order_release);
         p.take();                                // donated: the new base ref
         return smart_ptr<T>(ses, old_a, false);  // the old base ref → caller
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
