#pragma once
// Phase-1 perf prototype: a path-compressed radix tree built on heart's
// radix_full<256> + smart_address, bump-allocated in one arena with
// ptr_address-style 32-bit offsets. SINGLE-VERSION, in-place (one writer,
// everything is the "current" version) — no MVCC/COW yet. Its only job is to
// exercise the radix dispatch and give honest random insert/find timings to
// iterate against; MVCC, the adaptive tiers, and SAL come next.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <span>
#include <sys/mman.h>

#include <heart/radix_full.hpp>
#include <heart/smart_address.hpp>

namespace heart::proto
{
   using sal::ptr_address;

   class radix_tree
   {
     public:
      explicit radix_tree(size_t reserve_bytes = (3ull << 30))
      {
         _base = static_cast<uint8_t*>(
             ::mmap(nullptr, reserve_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0));
         _cap  = reserve_bytes;
         _used = 8;  // offset 0 unused; null is smart_address 0xFFFFFFFF
         _root = smart_address::null();
      }
      ~radix_tree() { if (_base) ::munmap(_base, _cap); }
      radix_tree(const radix_tree&) = delete;
      radix_tree& operator=(const radix_tree&) = delete;

      size_t bytes_used() const noexcept { return _used; }

      void insert(std::span<const uint8_t> key, uint64_t value) { insert(&_root, key, 0, value); }

      bool find(std::span<const uint8_t> key, uint64_t& out) const noexcept
      {
         smart_address sa    = _root;
         uint32_t      depth = 0;
         while (!sa.is_null())
         {
            const uint8_t* node = _base + *sa.as_pointer();
            if (node[0] == LEAF)
            {
               auto* l = reinterpret_cast<const leaf*>(node);
               if (l->klen == key.size() && std::memcmp(l->key, key.data(), l->klen) == 0)
               {
                  out = l->value;
                  return true;
               }
               return false;
            }
            auto* n = reinterpret_cast<const inner*>(node);
            if (n->nprefix > key.size() - depth)
               return false;
            if (n->nprefix && std::memcmp(n->prefix(), key.data() + depth, n->nprefix) != 0)
               return false;
            depth += n->nprefix;
            if (depth == key.size())
            {
               if (n->has_term)
               {
                  out = n->term;
                  return true;
               }
               return false;
            }
            sa = n->children.find(key[depth]);
            ++depth;
         }
         return false;
      }

     private:
      enum : uint8_t { LEAF = 0, INNER = 1 };

      struct leaf
      {
         uint8_t  type;  // LEAF @0
         uint8_t  _pad;
         uint16_t klen;
         uint32_t _pad2;
         uint64_t value;    // @8
         uint8_t  key[];    // @16
      };
      struct inner
      {
         uint8_t  type;      // INNER @0
         uint8_t  has_term;  // @1
         uint16_t nprefix;   // @2
         uint32_t _pad;      // @4
         uint64_t term;      // @8
         radix_full<256> children;  // @16 (1024 B)
         uint8_t*        prefix() noexcept { return reinterpret_cast<uint8_t*>(this) + sizeof(inner); }
         const uint8_t*  prefix() const noexcept
         {
            return reinterpret_cast<const uint8_t*>(this) + sizeof(inner);
         }
      };
      static_assert(sizeof(inner) == 16 + 1024);

      uint32_t bump(uint32_t n) noexcept
      {
         _used     = (_used + 7) & ~size_t(7);
         uint32_t o = uint32_t(_used);
         _used += n;
         assert(_used <= _cap && "radix_tree arena exhausted");
         return o;
      }
      template <class T>
      T* at(uint32_t off) noexcept { return reinterpret_cast<T*>(_base + off); }

      smart_address make_leaf(std::span<const uint8_t> key, uint64_t val) noexcept
      {
         uint32_t off = bump(uint32_t(sizeof(leaf) + key.size()));
         leaf*    l   = at<leaf>(off);
         l->type = LEAF;
         l->klen = uint16_t(key.size());
         l->value = val;
         std::memcpy(l->key, key.data(), key.size());
         return smart_address::from_pointer(ptr_address(off));
      }
      smart_address make_inner(const uint8_t* pfx, int plen) noexcept
      {
         uint32_t off = bump(uint32_t(sizeof(inner) + plen));
         inner*   n   = at<inner>(off);
         n->type = INNER;
         n->has_term = 0;
         n->nprefix = uint16_t(plen);
         n->term = 0;
         n->children.init();
         std::memcpy(n->prefix(), pfx, plen);
         return smart_address::from_pointer(ptr_address(off));
      }

      void insert(smart_address* slot, std::span<const uint8_t> key, uint32_t depth, uint64_t val) noexcept
      {
         if (slot->is_null())
         {
            *slot = make_leaf(key, val);
            return;
         }
         uint8_t* node = _base + *slot->as_pointer();

         if (node[0] == LEAF)
         {
            auto* l = reinterpret_cast<leaf*>(node);
            if (l->klen == key.size() && std::memcmp(l->key, key.data(), l->klen) == 0)
            {
               l->value = val;  // update in place
               return;
            }
            // split: longest common prefix of the two keys from `depth`
            int lcp  = 0;
            int maxp = int(std::min<size_t>(l->klen, key.size())) - int(depth);
            while (lcp < maxp && l->key[depth + lcp] == key[depth + lcp])
               ++lcp;
            smart_address inner_sa = make_inner(key.data() + depth, lcp);
            auto*         ni       = at<inner>(*inner_sa.as_pointer());
            uint32_t      d2       = depth + lcp;
            if (d2 == l->klen)
            {
               ni->has_term = 1;
               ni->term     = l->value;
            }
            else
               ni->children.set_branch(l->key[d2], *slot);  // old leaf keeps its slot value
            if (d2 == key.size())
            {
               ni->has_term = 1;
               ni->term     = val;
            }
            else
               ni->children.set_branch(key[d2], make_leaf(key, val));
            *slot = inner_sa;
            return;
         }

         auto*    n    = reinterpret_cast<inner*>(node);
         uint8_t* pfx  = n->prefix();
         int      p    = 0;
         int      maxp = int(std::min<size_t>(n->nprefix, key.size() - depth));
         while (p < maxp && pfx[p] == key[depth + p])
            ++p;
         if (p < n->nprefix)
         {
            // prefix mismatch -> split this node's prefix one level up
            smart_address ni_sa = make_inner(key.data() + depth, p);
            auto*         ni     = at<inner>(*ni_sa.as_pointer());
            uint8_t       old_b  = pfx[p];
            int           newpl  = n->nprefix - (p + 1);
            std::memmove(pfx, pfx + p + 1, newpl);  // n keeps the tail of its prefix
            n->nprefix = uint16_t(newpl);
            ni->children.set_branch(old_b, *slot);
            uint32_t d2 = depth + p;
            if (d2 == key.size())
            {
               ni->has_term = 1;
               ni->term     = val;
            }
            else
               ni->children.set_branch(key[d2], make_leaf(key, val));
            *slot = ni_sa;
            return;
         }
         depth += n->nprefix;
         if (depth == key.size())
         {
            n->has_term = 1;
            n->term     = val;
            return;
         }
         insert(&n->children.branches[key[depth]], key, depth + 1, val);
      }

      uint8_t*      _base = nullptr;
      size_t        _cap  = 0;
      size_t        _used = 0;
      smart_address _root = smart_address::null();
   };
}  // namespace heart::proto
