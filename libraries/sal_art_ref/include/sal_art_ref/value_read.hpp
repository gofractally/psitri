#pragma once
#include <adb/value_range.hpp>
#include <memory>
#include <optional>
#include <sal/read_lock.hpp>
#include <sal_art_ref/large_value.hpp>
#include <stdexcept>
#include <string>

namespace sal_art_ref::radix
{
   // A short-lived borrowed range pins SAL residency. A crossing range instead
   // owns its assembly buffer, and holds no residency lock during the callback.
   // Copies share owned storage, so their views remain valid across moves/copies.
   class value_ref
   {
      std::optional<sal::read_lock> _lock;
      std::shared_ptr<std::string>  _owned;
      std::string_view              _v;

     public:
      value_ref(const sal::read_lock& lock, std::string_view value) : _lock(lock), _v(value) {}
      explicit value_ref(std::string value)
          : _owned(std::make_shared<std::string>(std::move(value))), _v(*_owned)
      {
      }
                       operator std::string_view() const noexcept { return _v; }
      std::string_view view() const noexcept { return _v; }
      const char*      data() const noexcept { return _v.data(); }
      size_t           size() const noexcept { return _v.size(); }
      bool             empty() const noexcept { return _v.empty(); }
      std::string      str() const { return std::string(_v); }
      bool             is_borrowed() const noexcept { return !_owned; }
      template <class T>
      T copy_or_take() &&
      {
         if constexpr (std::same_as<T, std::string>)
            if (_owned && _owned.use_count() == 1)
               return std::move(*_owned);
         return T(_v.data(), _v.data() + _v.size());
      }
   };

   namespace value_detail
   {
      struct stored_value
      {
         std::string_view bytes;
         sal::ptr_address directory = sal::null_ptr_address;
      };
      inline stored_value from_external(const key_val_node& record, sal::ptr_address addr)
      {
         if (record.chunked())
            return {{}, addr};
         const auto v = record.value_bytes();
         return {{reinterpret_cast<const char*>(v.data()), v.size()}, sal::null_ptr_address};
      }
      inline stored_value from_external(sal::allocator_session& ses, sal::ptr_address addr)
      {
         return from_external(*ses.get_ref<key_val_node>(addr).obj(), addr);
      }
      // The caller must retain the owning tree/value for this whole operation.
      // `lock` is transferred in (never leave a hidden outer read lock around the
      // assembly loop). Directory metadata is copied before releasing its pin.
      inline value_ref read(sal::allocator_session&        ses,
                            stored_value                   value,
                            uint64_t                       offset,
                            uint64_t                       length,
                            std::optional<sal::read_lock>& lock)
      {
         if (value.directory == sal::null_ptr_address)
            return value_ref{*lock, adb::detail::value_slice(value.bytes, offset, length)};
         const auto* header = ses.get_ref<sal::alloc_header>(value.directory).obj();
         if (header->type() != key_val_node::type_id || header->size() < sizeof(key_val_node))
            throw std::runtime_error("invalid large-value head type/size");
         const auto*    head     = static_cast<const key_val_node*>(header);
         const uint64_t metadata = sizeof(key_val_node) + uint64_t(head->key_len) +
                                   uint64_t(head->continuation_count) * sizeof(chunk_entry);
         if (!head->chunked() || head->is_subtree() || head->key_len > key_val_node::max_key_len ||
             metadata >= head->size())
            throw std::runtime_error("invalid large-value head bounds");
         const auto     entries = head->chunks();
         const uint32_t total   = head->value_len;
         if (!entries.front().start || metadata + entries.front().start > head->size() ||
             entries.back().start >= total)
            throw std::runtime_error("invalid large-value first payload bounds");
         offset = std::min<uint64_t>(offset, total);
         length = std::min<uint64_t>(length, total - offset);
         if (!length)
            return value_ref{*lock, std::string_view{"", 0}};

         // Upper bound identifies the next continuation; the selected chunk is
         // its predecessor, or the head itself when offset precedes entry zero.
         const auto next =
             std::upper_bound(entries.begin(), entries.end(), offset,
                              [](uint64_t n, const chunk_entry& e) { return n < e.start; });
         struct piece
         {
            sal::ptr_address child;
            uint32_t         start;
            uint32_t         end;
         };
         const piece first{next == entries.begin() ? value.directory : (next - 1)->child,
                           next == entries.begin() ? 0 : (next - 1)->start,
                           next == entries.end() ? total : next->start};
         if (first.start > offset || first.end <= offset)
            throw std::runtime_error("invalid large-value directory order");
         auto resolve = [&](piece e) -> std::string_view
         {
            if (e.end <= e.start || e.end > total || !ses.is_resolvable(e.child))
               throw std::runtime_error("invalid large-value child/range");
            const auto* p = ses.get_ref<sal::alloc_header>(e.child).obj();
            if (!e.start)
            {
               if (e.child != value.directory || p->type() != key_val_node::type_id)
                  throw std::runtime_error("invalid large-value first chunk");
               const auto first = static_cast<const key_val_node*>(p)->first_bytes();
               return {reinterpret_cast<const char*>(first.data()), first.size()};
            }
            if (p->size() < sizeof(value_chunk) || p->type() != value_chunk::type_id)
               throw std::runtime_error("invalid large-value continuation type/size");
            const auto* c = static_cast<const value_chunk*>(p);
            if (c->payload_size != e.end - e.start ||
                uint64_t(sizeof(value_chunk)) + c->payload_size > c->size())
               throw std::runtime_error("invalid large-value chunk length");
            return {reinterpret_cast<const char*>(c->data), c->payload_size};
         };
         if (length <= first.end - offset)
            return value_ref{*lock,
                             resolve(first).substr(size_t(offset - first.start), size_t(length))};

         const uint64_t     end = offset + length;  // clamped above: cannot overflow
         std::vector<piece> needed;
         const auto         stop =
             std::lower_bound(next, entries.end(), end,
                              [](const chunk_entry& e, uint64_t n) { return e.start < n; });
         needed.reserve(1 + size_t(stop - next));
         needed.push_back(first);
         for (auto i = next; i != stop; ++i)
            needed.push_back({i->child, i->start, i + 1 == entries.end() ? total : (i + 1)->start});
         lock.reset();
         std::string result(size_t(length), '\0');
         size_t      written      = 0;
         uint32_t    previous_end = first.start;
         for (const auto& e : needed)
         {
            auto pin = ses.lock();
            if (e.start != previous_end)
               throw std::runtime_error("noncontiguous large-value directory");
            const auto     bytes = resolve(e);
            const uint64_t local = offset > e.start ? offset - e.start : 0;
            const size_t count = size_t(std::min<uint64_t>(bytes.size() - local, length - written));
            std::memcpy(result.data() + written, bytes.data() + local, count);
            written += count;
            previous_end = e.end;
         }
         if (written != length)
            throw std::runtime_error("incomplete large-value range");
         return value_ref{std::move(result)};
      }
   }  // namespace value_detail
}  // namespace sal_art_ref::radix
