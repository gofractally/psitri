#include <array>
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <psitri/api.hpp>
#include <start_ref/large_value.hpp>
#include <thread>

namespace
{
   using Small          = psitri::small_engine;
   namespace lv         = start_ref::large_value_detail;
   constexpr size_t MiB = 1024 * 1024;
   struct small_temp
   {
      std::filesystem::path path;
      small_temp()
      {
         auto name =
             (std::filesystem::temp_directory_path() / "psitri-small-large-XXXXXX").string();
         if (!::mkdtemp(name.data()))
            throw std::runtime_error("mkdtemp failed");
         path = name;
      }
      ~small_temp()
      {
         std::error_code ec;
         std::filesystem::remove_all(path, ec);
      }
   };
   Small::config small_config(bool background = true)
   {
      Small::config c;
      c.max_segments       = 8;
      c.background_release = background;
      return c;
   }
   std::string small_payload(size_t n)
   {
      std::array<char, 65536> pattern;
      for (size_t i = 0; i < pattern.size(); ++i)
         pattern[i] = char(i * 137 + (i >> 8) * 29);
      std::string result(n, '\0');
      for (size_t i = 0; i < n; i += pattern.size())
         std::memcpy(result.data() + i, pattern.data(), std::min(pattern.size(), n - i));
      return result;
   }
   bool same(std::string_view a, std::string_view b)
   {
      return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size()) == 0);
   }
   std::string_view physical_slot(Small::write_session& ws, std::string_view key)
   {
      start_ref::kv::bval out;
      if (!ws.tree_dbg().find(key, out))
         throw std::runtime_error("missing physical slot");
      return out.v;
   }
}  // namespace

TEST_CASE("Small allocates and frees values through 512 MiB among smaller records",
          "[api][small-large][512mb]")
{
   small_temp temp;
   const auto input = small_payload(512 * MiB);
   auto       db = Small::database::open(temp.path, psitri::open_mode::create_only, small_config());
   constexpr size_t contiguous = 2 * MiB - lv::leaf_header_size - 2 - 5;  // key "large"
   for (size_t size : {contiguous - 1, contiguous, contiguous + 1, 2 * MiB, 2 * MiB + 1, 3 * MiB,
                       4 * MiB, 16 * MiB, 64 * MiB + 17, 128 * MiB + 11, 512 * MiB})
   {
      INFO("payload=" << size);
      const auto expected = std::string_view(input).substr(0, size);
      auto       ws       = db->start_write_session();
      {
         auto tx = ws->start_transaction(0);
         for (int i = 0; i < 30; ++i)
            tx.upsert("small/" + std::to_string(i), std::string(17 + i * 127, char(i)));
         tx.upsert("large", expected);
         REQUIRE(same(*tx.get<std::string>("large", size - 13, 13), expected.substr(size - 13)));
         for (int i = 0; i < 30; i += 2)
            REQUIRE(tx.remove("small/" + std::to_string(i)) == 0);
         auto any = tx.get_any("large");
         REQUIRE(any.type() == start_ref::kv::kRawType);
         REQUIRE(same(any.bytes(size - 19, 19), expected.substr(size - 19)));
         if (lv::chunked(physical_slot(*ws, "large")))
         {
            const lv::head_view head(physical_slot(*ws, "large"));
            const size_t        offset    = head.first().size() - 11;
            const auto          assembled = any.bytes(offset, 29);
            auto                copied    = any;
            any.bytes(0, 1);  // the copy must keep the previous assembly alive
            auto moved = std::move(copied);
            REQUIRE(same(assembled, expected.substr(offset, 29)));
            REQUIRE(same(moved.bytes(offset, 29), expected.substr(offset, 29)));
         }
         tx.commit();
      }
      {
         auto snapshot = ws->get_root(0);
         auto copy     = snapshot;
         auto cur      = copy.snapshot_cursor();
         cur.find("large");
         REQUIRE_FALSE(cur.is_end());
         REQUIRE(cur.value_size() == int64_t(size));
         REQUIRE(cur.value_type() == start_ref::kv::kRawType);
         REQUIRE(cur.value(size)->empty());
         REQUIRE(cur.value(adb::read_all)->empty());
         REQUIRE(cur.value(0, 0)->empty());
         REQUIRE_FALSE(cur.get("missing", 0, 7));
         REQUIRE_THROWS_AS(
             cur.get_value([](std::string_view) { throw std::runtime_error("callback"); }, 0, 1),
             std::runtime_error);
         const auto slot = physical_slot(*ws, "large");
         if (size <= contiguous)
            REQUIRE_FALSE(lv::chunked(slot));
         else
         {
            REQUIRE(lv::chunked(slot));
            const lv::head_view head(slot);
            REQUIRE_THROWS_AS(cur.value<std::string_view>(head.first().size() - 1, 2),
                              std::invalid_argument);
            REQUIRE(cur.value<std::string_view>(7, 31)->data() == head.first().data() + 7);
            if (size == 512 * MiB)
               REQUIRE(head.entries().size() > 255);
            REQUIRE(cur.get_value(
                [&](std::string_view v)
                {
                   REQUIRE(v.data() == head.first().data() + 7);
                   REQUIRE(same(v, expected.substr(7, 31)));
                },
                7, 31));
            for (const auto& e : head.entries())
            {
               const size_t offset = e.start - 11;
               REQUIRE(same(*cur.value(offset, 29), expected.substr(offset, 29)));
               unsigned calls = 0;
               REQUIRE(cur.get_value(
                   [&](std::string_view v)
                   {
                      ++calls;
                      REQUIRE(same(v, expected.substr(offset, 29)));
                   },
                   offset, 29));
               REQUIRE(calls == 1);
               REQUIRE(cur.get_value(
                   [&](std::string_view v)
                   {
                      REQUIRE(v.data() == db->large_store().get(e.grain));
                      REQUIRE(same(v, expected.substr(e.start, 1)));
                   },
                   e.start, 1));
            }
         }
         {
            auto full = cur.value();
            REQUIRE(full);
            REQUIRE(same(*full, expected));
         }
         {
            auto tx = ws->start_transaction(0);
            tx.upsert("large", "replacement");
            tx.commit();
         }
         REQUIRE(same(*snapshot.get("large", size - 113, 113), expected.substr(size - 113)));
         REQUIRE(ws->get_root(0).get("large") == "replacement");
      }
      {
         auto tx = ws->start_transaction(0);
         REQUIRE(tx.remove("large") == 0);
         for (int i = 1; i < 30; i += 2)
         {
            REQUIRE(tx.get<std::string>("small/" + std::to_string(i)) ==
                    std::string(17 + i * 127, char(i)));
            REQUIRE(tx.remove("small/" + std::to_string(i)) == 0);
         }
         tx.commit();
      }
      ws.reset();  // flush allocator windows before measuring reclamation
      db->flush_releases();
      REQUIRE(db->large_used_grains() == 0);
   }
}

TEST_CASE("Small single large root snapshots and saturated continuation refcounts",
          "[api][small-large][ownership]")
{
   small_temp temp;
   auto db = Small::database::open(temp.path, psitri::open_mode::create_only, small_config(false));
   auto ws = db->start_write_session();
   const auto input = small_payload(3 * MiB);
   {
      auto tx = ws->start_transaction(0);
      tx.upsert("large", input);
      tx.commit();
   }
   REQUIRE(db->live_root_dbg().tag() == start_ref::hkv::detail::K::value_large);
   {
      auto old    = ws->get_root(0);
      auto copied = old;
      auto cursor = copied.snapshot_cursor();
      REQUIRE(same(*cursor.get("large"), input));
      const lv::head_view original(physical_slot(*ws, "large"));
      const auto          grain = original.entries()[0].grain;
      unsigned            extra = 0;
      while (db->large_store().ref(grain) < start_ref::writer_clone_at)
      {
         db->large_store().add_ref_or_throw(grain);
         ++extra;
      }
      {
         auto tx = ws->start_transaction(0);
         tx.upsert("large/child", "force leaf clone and split");
         tx.commit();
      }
      const lv::head_view current(physical_slot(*ws, "large"));
      REQUIRE(current.entries()[0].grain != grain);
      while (extra--)
         db->large_store().release(grain);
      REQUIRE(same(*cursor.get("large", 2 * MiB - 100, 200),
                   std::string_view(input).substr(2 * MiB - 100, 200)));
      REQUIRE(same(*ws->get_root(0).get("large"), input));
      {
         auto tx = ws->start_transaction(0);
         tx.upsert("large", "short");
         tx.commit();
      }
      REQUIRE(same(*old.get("large"), input));
   }
   {
      auto tx = ws->start_transaction(0);
      tx.remove("large");
      tx.remove("large/child");
      tx.commit();
   }
   ws.reset();
   db->flush_releases();
   REQUIRE(db->large_used_grains() == 0);
}

TEST_CASE("Small releases unpublished chunks after allocation failure",
          "[api][small-large][allocation-failure]")
{
   small_temp temp;
   auto       cfg           = small_config(false);
   cfg.max_segments         = 1;
   auto                  db = Small::database::open(temp.path, psitri::open_mode::create_only, cfg);
   auto                  ws = db->start_write_session();
   std::vector<uint32_t> reserved;
   size_t                value_size = 0;
   SECTION("failure while preparing continuations")
   {
      value_size = 6 * MiB;
   }
   SECTION("failure allocating the head after preparing continuations")
   {
      value_size = 3 * MiB;
   }
   const auto input = small_payload(value_size);
   {
      auto tx = ws->start_transaction(0);
      tx.upsert("keep", "original");
      // Reserve all but one freeset word without touching the reserved payload.
      // The remaining 2 MiB admits a continuation, but cannot finish either write.
      auto& arena_session = *ws->tree_dbg().allocator_mut().lses;
      for (uint32_t i = 0; i < start_ref::lines_per_segment / 64 - 1; ++i)
         reserved.push_back(arena_session.alloc_lines(64));
      REQUIRE_THROWS_AS(tx.upsert("keep", input), std::runtime_error);
      REQUIRE(tx.get("keep") == "original");
      tx.upsert("another", "still usable");
      tx.commit();
   }
   ws.reset();
   db->flush_releases();
   REQUIRE(db->large_used_grains() == reserved.size() * 64);
   for (const auto grain : reserved)
      db->large_store().release_run(grain, 64);
   REQUIRE(db->large_used_grains() == 0);
}

TEST_CASE("Small chunked values survive abort multi-root reads and durable reopen",
          "[api][small-large][persistence]")
{
   small_temp temp;
   auto       cfg          = small_config();
   cfg.sync_mode           = sal::sync_type::fsync;
   const auto        input = small_payload(64 * MiB + 17);
   const std::string key(1024, 'k');
   {
      auto db = Small::database::open(temp.path, psitri::open_mode::create_only, cfg);
      auto ws = db->start_write_session();
      {
         auto tx = ws->start_transaction(0);
         tx.upsert(key, input);
         auto second = tx.open_root(1);
         second.upsert("other", std::string_view(input).substr(0, 3 * MiB));
         REQUIRE(same(*second.get("other", 2 * MiB - 100, 200),
                      std::string_view(input).substr(2 * MiB - 100, 200)));
         tx.commit();
      }
      REQUIRE(db->large_store().stats().bytes_synced.load() >= input.size());
      {
         auto tx = ws->start_transaction(0);
         tx.upsert(key, "abort replacement");
         tx.upsert("aborted", input);
         tx.open_root(1).upsert("other", input);
         tx.abort();
      }
      REQUIRE(same(*ws->get_root(0).get(key, input.size() - 13, 13),
                   std::string_view(input).substr(input.size() - 13)));
      REQUIRE_FALSE(ws->get_root(0).get("aborted"));
      REQUIRE(ws->get_root(1).get("other")->size() == 3 * MiB);
   }
   {
      auto db = Small::database::open(temp.path, psitri::open_mode::open_existing, cfg);
      auto ws = db->start_write_session();
      REQUIRE(same(*ws->get_root(0).get(key), input));
      REQUIRE(same(*ws->get_root(1).get("other"), std::string_view(input).substr(0, 3 * MiB)));
      {
         auto tx = ws->start_transaction(0);
         REQUIRE(tx.remove(key) == 0);
         REQUIRE(tx.open_root(1).remove("other") == 0);
         tx.commit();
      }
      ws.reset();
      db->flush_releases();
      REQUIRE(db->large_used_grains() == 0);
   }
}

TEST_CASE("Small range readers retain old chunks through concurrent replacement",
          "[api][small-large][threads]")
{
   small_temp temp;
   auto       db = Small::database::open(temp.path, psitri::open_mode::create_only, small_config());
   auto       ws = db->start_write_session();
   const auto input = small_payload(16 * MiB + 7);
   {
      auto tx = ws->start_transaction(0);
      tx.upsert("large", input);
      tx.commit();
   }
   const auto            boundary = lv::head_view(physical_slot(*ws, "large")).entries()[0].start;
   std::atomic<unsigned> reads{0};
   std::atomic<bool>     failed{false};
   std::jthread          reader(
       [&](std::stop_token stop)
       {
          try
          {
             auto rs  = db->start_read_session();
             auto cur = rs->create_cursor(0);
             while (!stop.stop_requested())
             {
                auto part = cur.get("large", boundary - 13, 91);
                if (!part || !same(*part, std::string_view(input).substr(boundary - 13, 91)))
                   failed = true;
                ++reads;
             }
          }
          catch (...)
          {
             failed = true;
          }
       });
   while (!reads && !failed)
      std::this_thread::yield();
   for (unsigned i = 0; i < 12; ++i)
   {
      auto tx = ws->start_transaction(0);
      tx.upsert("large", i % 2 ? std::string_view("replacement") : std::string_view(input));
      tx.upsert("small", std::string(1031, char(i)));
      tx.commit();
   }
   reader.request_stop();
   reader.join();
   REQUIRE(reads > 0);
   REQUIRE_FALSE(failed);
   {
      auto tx = ws->start_transaction(0);
      tx.remove("large");
      tx.remove("small");
      tx.commit();
   }
   ws.reset();
   db->flush_releases();
   REQUIRE(db->large_used_grains() == 0);
}

TEST_CASE("Small plans UINT32_MAX and rejects 4 GiB without reading input",
          "[api][small-large][limit]")
{
   static_assert(Small::max_value_size == UINT32_MAX);
   const auto p = lv::plan(UINT32_MAX, 1024);
   REQUIRE(p.continuation_count > 255);
   REQUIRE(uint64_t(p.first_size) + uint64_t(p.continuation_count) * lv::chunk_capacity >=
           UINT32_MAX);
   REQUIRE(uint64_t(p.first_size) + uint64_t(p.continuation_count - 1) * lv::chunk_capacity <
           UINT32_MAX);
   small_temp temp;
   auto       db = Small::database::open(temp.path, psitri::open_mode::create_only, small_config());
   auto       ws = db->start_write_session();
   auto       tx = ws->start_transaction(0);
   tx.upsert("keep", "unchanged");
   struct inaccessible
   {
      static constexpr size_t size() { return size_t(UINT32_MAX) + 1; }
      void* data = ::mmap(nullptr, size(), PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
      ~inaccessible()
      {
         if (data != MAP_FAILED)
            ::munmap(data, size());
      }
   } input;
   REQUIRE(input.data != MAP_FAILED);
   const std::string_view invalid(static_cast<const char*>(input.data), input.size());
   REQUIRE_THROWS_AS(tx.upsert("keep", invalid), std::invalid_argument);
   REQUIRE_THROWS_AS(tx.primary().upsert("new", invalid), std::invalid_argument);
   REQUIRE(tx.get<std::string>("keep") == "unchanged");
   REQUIRE_FALSE(tx.get<std::string>("new"));
}
