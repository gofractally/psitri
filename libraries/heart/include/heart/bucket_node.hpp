#pragma once
#include <cstring>

#include <heart/inline_value.hpp>
#include <heart/paged.hpp>
#include <heart/radix_type_ids.hpp>
#include <heart/smart_address.hpp>  // tail_ptr
#include <ucc/round.hpp>

namespace heart
{
   /**
    * The "last-mile" terminal: a bucket of variable-length (suffix, value) entries kept
    * as a 2-3-4 B-tree *inside this node's own tail heap*. It is a plain `paged` node — a
    * page like any other (header + tail heap), NOT a radix node: it terminates the radix,
    * it doesn't route. It collapses the sparse, early-diverging terminal region that a
    * pure byte-radix over-nodes (each radix hop costs 2 random cachelines; a bucket
    * resolves up to a page-full of keys in one self-contained, COW-as-a-unit object).
    *
    * The 2-3-4 nodes and the `inline_value` entries live in the tail heap, linked by
    * `tail_ptr` (16-bit offset-from-tail). Inserts are memmove-free (split a small node,
    * never shift a big array); find / lower_bound / in-order are ordered (full-suffix
    * compare), so it composes with the engine's lower_bound/upper_bound. When the page
    * can no longer fit an insert, insert() reports overflow and the caller splits it into
    * a radix node over child buckets.
    *
    * (v1: full suffixes are stored inline via inline_value, so a single entry's
    * suffix+value must fit inline_value::max_inline_size; oversized values spill to a
    * standalone leaf_node referenced by the entry — wired in a later step.)
    */
   struct bucket_node : paged
   {
      static constexpr sal::header_type type_id = radix_type_ids::bucket;

      /// A 2-3-4 node: up to 3 entries and (if internal) up to 4 children, all tail_ptr.
      struct node234
      {
         uint8_t                n;     // entries in use, 0..3
         uint8_t                leaf;  // 1 if no children
         tail_ptr<inline_value> ent[3];
         tail_ptr<node234>      kid[4];
      };

      tail_ptr<node234> root;  // is_null() => empty bucket

      // ── page lifecycle (concrete; a bucket is sizeof(bucket_node) + a tail heap) ─────
      static constexpr uint32_t page_bytes = 4096;  // target page; fills then splits to radix
      static constexpr uint32_t head_bytes() noexcept { return uint32_t(sizeof(bucket_node)); }
      uint32_t head_size() const noexcept { return uint32_t(sizeof(bucket_node)); }
      uint32_t free_heap() const noexcept { return this->free_bytes(head_end()); }
      static uint32_t alloc_size() noexcept
      {
         return ucc::round_up_multiple<sal::cacheline_size>(page_bytes);
      }
      // No SAL children (entries are inline tail_ptr), so destroy is the alloc_header no-op.
      template <typename F>
      void for_each_child(F&&) const
      {
      }

      bucket_node(uint32_t asize, sal::ptr_address_seq seq) noexcept
      {
         this->init_page(asize, type_id, seq);
         root = tail_ptr<node234>{0};
      }

      // ── tail-heap allocation (tail_ptr is offset-from-tail; 0 == null) ──────────────
      const char* head_end() const noexcept
      {
         return reinterpret_cast<const char*>(this) + sizeof(bucket_node);
      }
      tail_ptr<node234> alloc_node(bool leaf) noexcept
      {
         if (!this->can_alloc_tail(uint16_t(sizeof(node234)), head_end()))
            return {0};
         this->alloc_tail(uint16_t(sizeof(node234)));
         tail_ptr<node234> tp = this->template tail_pos<node234>();
         node234*          x  = this->resolve(tp);
         x->n    = 0;
         x->leaf = leaf ? 1 : 0;
         return tp;
      }
      tail_ptr<inline_value> alloc_entry(const uint8_t* s, int slen, const uint8_t* v,
                                         int vlen) noexcept
      {
         const int need = inline_value::byte_size(slen, vlen);
         if (!this->can_alloc_tail(uint16_t(need), head_end()))
            return {0};
         this->alloc_tail(uint16_t(need));
         tail_ptr<inline_value> tp = this->template tail_pos<inline_value>();
         inline_value*          iv = this->resolve(tp);
         iv->suffix_len = uint8_t(slen);
         iv->value_len  = uint8_t(vlen);
         iv->value_type = 0;
         std::memcpy(iv->data, s, size_t(slen));
         std::memcpy(iv->data + slen, v, size_t(vlen));
         return tp;
      }

      // entry vs (s,slen): -1 / 0 / 1
      int cmp(tail_ptr<inline_value> e, const uint8_t* s, int slen) const noexcept
      {
         const inline_value* iv = this->resolve(e);
         const int           el = iv->suffix_len;
         const int           m  = el < slen ? el : slen;
         const int           c  = m ? std::memcmp(iv->data, s, size_t(m)) : 0;
         if (c)
            return c < 0 ? -1 : 1;
         return el == slen ? 0 : (el < slen ? -1 : 1);
      }

      // ── find ────────────────────────────────────────────────────────────────────
      const inline_value* find(const uint8_t* s, int slen) const noexcept
      {
         tail_ptr<node234> x = root;
         while (!x.is_null())
         {
            const node234* nx = this->resolve(x);
            int            i  = 0;
            while (i < nx->n && cmp(nx->ent[i], s, slen) < 0)
               ++i;
            if (i < nx->n && cmp(nx->ent[i], s, slen) == 0)
               return this->resolve(nx->ent[i]);
            if (nx->leaf)
               return nullptr;
            x = nx->kid[i];
         }
         return nullptr;
      }

      // ── lower_bound: smallest entry with suffix >= (s,slen); null if none ───────────
      tail_ptr<inline_value> lower_bound(const uint8_t* s, int slen) const noexcept
      {
         tail_ptr<node234>      x    = root;
         tail_ptr<inline_value> best = {0};
         while (!x.is_null())
         {
            const node234* nx = this->resolve(x);
            int            i  = 0;
            while (i < nx->n && cmp(nx->ent[i], s, slen) < 0)
               ++i;
            if (i < nx->n)
            {
               best = nx->ent[i];  // tightest separator >= s
               if (cmp(nx->ent[i], s, slen) == 0)
                  return nx->ent[i];
            }
            if (nx->leaf)
               break;
            x = nx->kid[i];
         }
         return best;
      }

      // ── insert (top-down preemptive split): 1 inserted, -1 updated, 0 OVERFLOW ──────
      // split full child i of x (x must not be full); false on overflow.
      bool split_child(tail_ptr<node234> xp, int i) noexcept
      {
         node234*          x  = this->resolve(xp);
         tail_ptr<node234> yp = x->kid[i];
         node234*          y  = this->resolve(yp);  // full: y->n == 3
         tail_ptr<node234> zp = alloc_node(y->leaf);
         if (zp.is_null())
            return false;
         x = this->resolve(xp);  // (alloc doesn't move existing blocks, but keep refs fresh)
         y = this->resolve(yp);
         node234* z = this->resolve(zp);
         z->n      = 1;
         z->ent[0] = y->ent[2];
         if (!y->leaf)
         {
            z->kid[0] = y->kid[2];
            z->kid[1] = y->kid[3];
         }
         const tail_ptr<inline_value> median = y->ent[1];
         y->n = 1;  // y keeps ent[0], kid[0..1]
         for (int k = x->n; k > i; --k)
            x->ent[k] = x->ent[k - 1];
         for (int k = x->n + 1; k > i + 1; --k)
            x->kid[k] = x->kid[k - 1];
         x->ent[i]     = median;
         x->kid[i + 1] = zp;
         x->n++;
         return true;
      }
      int insert(const uint8_t* s, int slen, const uint8_t* v, int vlen) noexcept
      {
         if (root.is_null())
         {
            root = alloc_node(true);
            if (root.is_null())
               return 0;
         }
         if (this->resolve(root)->n == 3)  // grow height
         {
            tail_ptr<node234> nr = alloc_node(false);
            if (nr.is_null())
               return 0;
            this->resolve(nr)->kid[0] = root;
            root                      = nr;
            if (!split_child(root, 0))
               return 0;
         }
         tail_ptr<node234> xp = root;
         for (;;)
         {
            node234* nx = this->resolve(xp);
            int      i  = 0;
            while (i < nx->n && cmp(nx->ent[i], s, slen) < 0)
               ++i;
            if (i < nx->n && cmp(nx->ent[i], s, slen) == 0)  // duplicate -> update value
            {
               tail_ptr<inline_value> e = alloc_entry(s, slen, v, vlen);
               if (e.is_null())
                  return 0;
               this->resolve(xp)->ent[i] = e;
               return -1;
            }
            if (nx->leaf)
            {
               tail_ptr<inline_value> e = alloc_entry(s, slen, v, vlen);
               if (e.is_null())
                  return 0;
               nx = this->resolve(xp);
               for (int k = nx->n; k > i; --k)
                  nx->ent[k] = nx->ent[k - 1];
               nx->ent[i] = e;
               nx->n++;
               return 1;
            }
            if (this->resolve(nx->kid[i])->n == 3)  // split before descending
            {
               if (!split_child(xp, i))
                  return 0;
               nx      = this->resolve(xp);
               int c   = cmp(nx->ent[i], s, slen);
               if (c == 0)
               {
                  tail_ptr<inline_value> e = alloc_entry(s, slen, v, vlen);
                  if (e.is_null())
                     return 0;
                  this->resolve(xp)->ent[i] = e;
                  return -1;
               }
               if (c < 0)
                  ++i;
            }
            xp = this->resolve(xp)->kid[i];
         }
      }

      // ── in-order visit: f(const inline_value&) ──────────────────────────────────────
      template <typename F>
      void walk(tail_ptr<node234> xp, F&& f) const
      {
         if (xp.is_null())
            return;
         const node234* nx = this->resolve(xp);
         for (int i = 0; i < nx->n; ++i)
         {
            if (!nx->leaf)
               walk(nx->kid[i], f);
            f(*this->resolve(nx->ent[i]));
         }
         if (!nx->leaf)
            walk(nx->kid[nx->n], f);
      }
      template <typename F>
      void for_each(F&& f) const
      {
         walk(root, f);
      }
   } __attribute__((packed));
}  // namespace heart
