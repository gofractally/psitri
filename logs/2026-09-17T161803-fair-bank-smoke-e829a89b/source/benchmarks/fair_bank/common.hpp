#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <latch>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <vector>

namespace fair_bank
{
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
struct Config
{
   fs::path path, result;
   uint64_t accounts = 10000, transfers = 10000, warmup = 1000, batch = 100;
   uint64_t readers = 2, reads_per_snapshot = 100, read_batches = 100;
   uint64_t seed = 12345, value_bytes = 8, cache_mib = 256, map_gib = 64;
   bool sync = true;
};
inline void require(bool ok, const std::string& message)
{
   if (!ok) throw std::runtime_error(message);
}
inline std::string encode(uint64_t n, size_t size = 8)
{
   std::string result(size, char(0xa5));
   for (int i = 7; i >= 0; --i) { result[i] = char(n & 255); n >>= 8; }
   return result;
}
inline uint64_t decode(std::string_view value, size_t size)
{
   require(value.size() == size, "Invalid value size");
   require(std::all_of(value.begin() + 8, value.end(), [](char c) { return c == char(0xa5); }),
           "Value payload corrupted");
   uint64_t result = 0;
   for (unsigned char c : value.substr(0, 8)) result = (result << 8) | c;
   return result;
}
// Specified generator, independent of std::uniform_int_distribution implementations.
struct Random
{
   uint64_t state;
   uint64_t next()
   {
      uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
      z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
      z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
      return z ^ (z >> 31);
   }
};
inline std::string json(std::string_view s)
{
   std::string out = "\"";
   for (unsigned char c : s)
   {
      if (c == '"' || c == '\\') { out += '\\'; out += char(c); }
      else if (c < 32) { const char* hex = "0123456789abcdef"; out += "\\u00"; out += hex[c >> 4]; out += hex[c & 15]; }
      else out += char(c);
   }
   return out + '"';
}
inline double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
struct Metric { std::string name, unit, direction; double value; };
}
