#pragma once
// Shared by the writer and reader cases, which run on several sets of engines.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace
{
   struct temporary_directory
   {
      std::filesystem::path path;
      temporary_directory()
      {
         std::string pattern =
             (std::filesystem::temp_directory_path() / "psitri-writers-XXXXXX").string();
         auto* result = ::mkdtemp(pattern.data());
         if (!result)
            throw std::runtime_error("mkdtemp failed");
         path = result;
      }
      ~temporary_directory()
      {
         std::error_code ignored;
         std::filesystem::remove_all(path, ignored);
      }
   };

   std::string key(std::uint32_t root, std::uint32_t round, std::uint32_t i)
   {
      char buffer[40];
      std::snprintf(buffer, sizeof(buffer), "r%02u/%05u/%05u", root, round, i);
      return buffer;
   }

   // Number of keys under `root`, checking that every key belongs to it.
   template <typename Cursor>
   std::size_t count_own_keys(Cursor& cursor, std::uint32_t root, bool& foreign)
   {
      char prefix[8];
      std::snprintf(prefix, sizeof(prefix), "r%02u/", root);
      std::size_t n = 0;
      for (cursor.seek_begin(); !cursor.is_end(); cursor.next())
      {
         if (!cursor.key().starts_with(prefix))
            foreign = true;
         ++n;
      }
      return n;
   }
}  // namespace
