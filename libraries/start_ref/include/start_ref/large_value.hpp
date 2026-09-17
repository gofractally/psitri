#pragma once
#include <adb/value_range.hpp>
#include <algorithm>
#include <limits>
#include <memory>
#include <span>
#include <start_ref/arena.hpp>
#include <string>
#include <vector>

namespace start_ref::large_value_detail
{
   inline constexpr uint16_t chunked_type     = UINT16_MAX;  // internal raw-byte representation
   inline constexpr uint64_t max_value_size   = UINT32_MAX;
   inline constexpr uint32_t chunk_capacity   = large_max_object_bytes;
   inline constexpr uint32_t leaf_header_size = 8;

   struct entry
   {
      uint32_t start;
      uint32_t grain;
   } __attribute__((packed));
   struct header
   {
      uint16_t type;
      uint32_t value_size;
      uint16_t continuation_count;
   } __attribute__((packed));
   static_assert(sizeof(entry) == 8 && sizeof(header) == 8);

   inline bool chunked(std::string_view slot) noexcept
   {
      uint16_t type;
      std::memcpy(&type, slot.data(), sizeof(type));
      return type == chunked_type;
   }
   struct layout
   {
      uint32_t first_size;
      uint16_t continuation_count;
   };
   constexpr layout plan(uint32_t value_size, uint32_t key_size)
   {
      const uint64_t overhead = uint64_t(leaf_header_size) + key_size + sizeof(header);
      if (overhead >= chunk_capacity)
         throw std::invalid_argument("start_ref: key exceeds the large-value head capacity");
      const uint32_t available = chunk_capacity - uint32_t(overhead);
      if (value_size <= available)
         return {value_size, 0};
      constexpr uint32_t net = chunk_capacity - sizeof(entry);
      const uint64_t     n   = (uint64_t(value_size) - available + net - 1) / net;
      if (n > UINT16_MAX || n * sizeof(entry) >= available)
         throw std::invalid_argument("start_ref: large-value directory exceeds head capacity");
      return {available - uint32_t(n * sizeof(entry)), uint16_t(n)};
   }

   // A view of the physical slot. The owning root keeps both this head and all
   // continuations alive; Small never relocates a live allocation.
   class head_view
   {
      std::string_view _slot;
      const header*    _head;

     public:
      explicit head_view(std::string_view slot)
          : _slot(slot), _head(reinterpret_cast<const header*>(slot.data()))
      {
         if (slot.size() < sizeof(header) || _head->type != chunked_type ||
             !_head->continuation_count)
            throw std::runtime_error("start_ref: invalid chunked-value header");
         const size_t metadata = sizeof(header) + size_t(_head->continuation_count) * sizeof(entry);
         if (metadata >= slot.size() || entries().front().start != slot.size() - metadata ||
             entries().back().start >= _head->value_size)
            throw std::runtime_error("start_ref: invalid chunked-value directory");
      }
      uint32_t               size() const noexcept { return _head->value_size; }
      std::span<const entry> entries() const noexcept
      {
         return {reinterpret_cast<const entry*>(_slot.data() + sizeof(header)),
                 _head->continuation_count};
      }
      std::string_view first() const noexcept
      {
         return _slot.substr(sizeof(header) + entries().size_bytes());
      }
      uint32_t end(size_t i) const noexcept
      {
         return i + 1 == entries().size() ? size() : entries()[i + 1].start;
      }
      uint32_t length(size_t i) const noexcept { return end(i) - entries()[i].start; }
   };

   inline uint32_t units(uint32_t bytes) noexcept
   {
      return (bytes + large_grain_bytes - 1) / large_grain_bytes;
   }
   inline void release(large_arena& store, uint32_t grain, uint32_t bytes) noexcept
   {
      store.release(grain,
                    [&](line_idx)
                    {
                       const auto n = units(bytes);
                       if (n > 1)
                          store.release_run(grain + 1, n - 1);
                    });
   }
   inline void release_head(large_arena& store, std::string_view slot) noexcept
   {
      const head_view h(slot);
      for (size_t i = 0; i < h.entries().size(); ++i)
         release(store, h.entries()[i].grain, h.length(i));
   }

   // Called only on a newly copied, private head. Share each continuation until
   // its byte refcount reaches the writer band, then clone it using this writer's
   // allocation session. On failure roll back precisely the references acquired.
   inline void copy_head(large_session& session, char* slot, size_t size)
   {
      auto&           store = session.store();
      const head_view h({slot, size});
      auto*           entries = reinterpret_cast<entry*>(slot + sizeof(header));
      size_t          owned   = 0;
      try
      {
         for (; owned < h.entries().size(); ++owned)
         {
            const auto old = entries[owned].grain;
            if (store.retain(old))
               continue;
            const auto bytes = h.length(owned);
            const auto grain = session.alloc_lines(units(bytes));
            std::memcpy(store.get(grain), store.get(old), bytes);
            entries[owned].grain = grain;
         }
      }
      catch (...)
      {
         for (size_t i = 0; i < owned; ++i)
            release(store, entries[i].grain, h.length(i));
         throw;
      }
   }

   // Continuations are prepared before touching the tree. The gather write
   // transfers their ownership into exactly one leaf, including if a later
   // tree-building allocation throws and destroys that leaf through its hook.
   class source
   {
      large_arena&       _store;
      std::string_view   _value;
      layout             _layout;
      std::vector<entry> _entries;
      mutable bool       _adopted = false;

     public:
      source(large_session& session, uint32_t key_size, std::string_view value)
          : _store(session.store()), _value(value), _layout(plan(uint32_t(value.size()), key_size))
      {
         assert(value.size() <= max_value_size && _layout.continuation_count);
         _entries.reserve(_layout.continuation_count);
         uint64_t offset = _layout.first_size;
         try
         {
            while (offset < value.size())
            {
               const uint32_t bytes =
                   uint32_t(std::min<uint64_t>(value.size() - offset, chunk_capacity));
               const auto grain = session.alloc_lines(units(bytes));
               _entries.push_back({uint32_t(offset), grain});
               std::memcpy(_store.get(grain), value.data() + offset, bytes);
               offset += bytes;
            }
         }
         catch (...)
         {
            discard();
            throw;
         }
         assert(_entries.size() == _layout.continuation_count);
      }
      source(const source&) = delete;
      ~source()
      {
         if (!_adopted)
            discard();
      }
      size_t size() const noexcept
      {
         return sizeof(header) + _entries.size() * sizeof(entry) + _layout.first_size;
      }
      void write(char* dest) const noexcept
      {
         assert(!_adopted);
         const header h{chunked_type, uint32_t(_value.size()), uint16_t(_entries.size())};
         std::memcpy(dest, &h, sizeof(h));
         std::memcpy(dest + sizeof(h), _entries.data(), _entries.size() * sizeof(entry));
         std::memcpy(dest + sizeof(h) + _entries.size() * sizeof(entry), _value.data(),
                     _layout.first_size);
         _adopted = true;
      }

     private:
      void discard() noexcept
      {
         for (const auto& e : _entries)
            release(_store, e.grain,
                    uint32_t(std::min<uint64_t>(_value.size() - e.start, chunk_capacity)));
      }
   };

   class range
   {
      std::shared_ptr<std::string> _owned;
      std::string_view             _view;

     public:
      explicit range(std::string_view v) : _view(v) {}
      explicit range(std::string v)
          : _owned(std::make_shared<std::string>(std::move(v))), _view(*_owned)
      {
      }
      std::string_view view() const noexcept { return _view; }
      bool             is_borrowed() const noexcept { return !_owned; }
      template <class S>
      S copy_or_take() &&
      {
         if constexpr (std::same_as<S, std::string>)
            if (_owned && _owned.use_count() == 1)
               return std::move(*_owned);
         if constexpr (std::same_as<S, std::string_view>)
            if (_owned)
               throw std::invalid_argument(
                   "start_ref: a crossing range needs an owning result or callback");
         return S(_view.data(), _view.size());
      }
   };

   inline range read(const large_arena* store,
                     std::string_view   slot,
                     uint64_t           offset,
                     uint64_t           length)
   {
      if (!chunked(slot))
         return range(adb::detail::value_slice(slot.substr(2), offset, length));
      if (!store)
         throw std::runtime_error("start_ref: missing large-value arena");
      const head_view h(slot);
      offset = std::min<uint64_t>(offset, h.size());
      length = std::min<uint64_t>(length, uint64_t(h.size()) - offset);
      if (!length)
         return range(std::string_view{"", 0});
      const auto   entries = h.entries();
      const auto   next  = std::upper_bound(entries.begin(), entries.end(), offset,
                                            [](uint64_t n, const entry& e) { return n < e.start; });
      const size_t first = size_t(next - entries.begin());  // 0=head, i+1=continuation i
      auto         piece = [&](size_t index) -> std::string_view
      {
         if (!index)
            return h.first();
         const size_t   i     = index - 1;
         const auto     e     = entries[i];
         const uint32_t bytes = h.length(i);
         const uint32_t n     = units(bytes);
         if (!bytes || bytes > chunk_capacity || is_sentinel(e.grain) ||
             uint64_t(e.grain) + n > uint64_t(store->num_segments()) * lines_per_segment ||
             e.grain % 64 + n > 64)
            throw std::runtime_error("start_ref: invalid continuation range");
         return {static_cast<const char*>(store->get(e.grain)), bytes};
      };
      const uint32_t start = first ? entries[first - 1].start : 0;
      auto           bytes = piece(first);
      const auto     local = size_t(offset - start);
      if (local > bytes.size())
         throw std::runtime_error("start_ref: invalid continuation order");
      if (length <= bytes.size() - local)
         return range(bytes.substr(local, size_t(length)));
      std::string result(size_t(length), '\0');
      size_t      written  = 0;
      uint64_t    expected = start;
      for (size_t i = first; written < length && i <= entries.size(); ++i)
      {
         const uint32_t begin = i ? entries[i - 1].start : 0;
         if (begin != expected)
            throw std::runtime_error("start_ref: noncontiguous directory");
         const auto   part  = piece(i);
         const size_t skip  = i == first ? local : 0;
         const size_t count = std::min(part.size() - skip, size_t(length) - written);
         std::memcpy(result.data() + written, part.data() + skip, count);
         written += count;
         expected += part.size();
      }
      if (written != length)
         throw std::runtime_error("start_ref: incomplete large-value range");
      return range(std::move(result));
   }
}  // namespace start_ref::large_value_detail
