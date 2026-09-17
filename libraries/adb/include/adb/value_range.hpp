#pragma once

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

namespace adb
{
   inline constexpr uint64_t read_all = std::numeric_limits<uint64_t>::max();

   namespace detail
   {
      template <class F>
      concept value_callback = requires(F& fn, std::string_view value) { fn(value); };

      struct value_consumer
      {
         void operator()(std::string_view) const noexcept {}
      };

      // Clamp by subtraction, never offset + length (which can overflow).
      constexpr std::string_view value_slice(std::string_view value, uint64_t offset,
                                             uint64_t length) noexcept
      {
         const auto start = std::min<uint64_t>(offset, value.size());
         value.remove_prefix(static_cast<size_t>(start));
         const auto count = std::min<uint64_t>(length, value.size());
         value.remove_suffix(value.size() - static_cast<size_t>(count));
         return value;
      }

      // Opt-in, empty mixins for engines whose scoped reads return contiguous
      // values. Existing full-read methods remain untouched. Explicit-object
      // parameters preserve the derived reader's constness without a vtable,
      // stored reader pointer, or wrapper object. A chunked engine must supply
      // native range methods that visit only intersecting chunks instead.
      struct range_reader
      {
         template <class S = std::string, class Self, std::integral Offset>
            requires std::constructible_from<S, const char*, size_t> &&
                     requires(Self& reader) {
                        { reader.get(std::string_view{}, value_consumer{}) } -> std::same_as<bool>;
                     }
         std::optional<S> get(this Self&& self, std::string_view key, Offset offset,
                              uint64_t length = read_all)
         {
            std::optional<S> result;
            self.get(key, [&](std::string_view value) {
               const auto part = value_slice(value, uint64_t(offset), length);
               result.emplace(part.data(), part.size());
            });
            return result;
         }

         template <class Self, class F>
            requires value_callback<F> && requires(Self& reader) {
               { reader.get(std::string_view{}, value_consumer{}) } -> std::same_as<bool>;
            }
         bool get(this Self&& self, std::string_view key, F&& callback, uint64_t offset,
                  uint64_t length = read_all)
         {
            return self.get(key, [&](std::string_view value) {
               callback(value_slice(value, offset, length));
            });
         }
      };

      struct range_cursor : range_reader
      {
         template <class S = std::string, class Self>
            requires std::constructible_from<S, const char*, size_t> &&
                     requires(Self& reader) {
                        { reader.get_value(value_consumer{}) } -> std::same_as<bool>;
                     }
         std::optional<S> value(this Self&& self, uint64_t offset, uint64_t length = read_all)
         {
            std::optional<S> result;
            self.get_value([&](std::string_view value) {
               const auto part = value_slice(value, offset, length);
               result.emplace(part.data(), part.size());
            });
            return result;
         }

         template <class Self, class F>
            requires value_callback<F> && requires(Self& reader) {
               { reader.get_value(value_consumer{}) } -> std::same_as<bool>;
            }
         bool get_value(this Self&& self, F&& callback, uint64_t offset,
                        uint64_t length = read_all)
         {
            return self.get_value([&](std::string_view value) {
               callback(value_slice(value, offset, length));
            });
         }
      };
   }  // namespace detail
}  // namespace adb
