#include <sys/mman.h>
#include <array>
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <psitri/api.hpp>
#include <sal_art_ref/large_value.hpp>
#include <string>
#include <thread>

namespace
{
   constexpr size_t MiB = 1024 * 1024;
   using E              = psitri::big_engine;
   using namespace sal_art_ref;
   struct temp_directory
   {
      std::filesystem::path path;
      temp_directory()
      {
         auto pattern = (std::filesystem::temp_directory_path() / "psitri-large-XXXXXX").string();
         if (!::mkdtemp(pattern.data()))
            throw std::runtime_error("mkdtemp failed");
         path = pattern;
      }
      ~temp_directory()
      {
         std::error_code ec;
         std::filesystem::remove_all(path, ec);
      }
   };
   E::config config()
   {
      E::config cfg;
      cfg.max_pinned_cache_size_mb = 128;  // much smaller than the largest value
      cfg.sync_mode                = sal::sync_type::msync_sync;
      return cfg;
   }
   std::string payload(size_t size)
   {
      std::array<char, 65536> block;
      for (size_t i = 0; i < block.size(); ++i)
         block[i] = char(i * 131 + (i >> 8) * 17);
      std::string result(size, '\0');
      for (size_t offset = 0; offset < size; offset += block.size())
         std::memcpy(result.data() + offset, block.data(), std::min(block.size(), size - offset));
      return result;
   }
   bool matches(std::string_view actual, std::string_view expected)
   {
      return actual.size() == expected.size() &&
             (actual.empty() || std::memcmp(actual.data(), expected.data(), actual.size()) == 0);
   }
   struct padding : sal::alloc_header
   {
      static constexpr auto type_id = sal::header_type(100);
      padding(uint32_t n, sal::ptr_address_seq seq) : sal::alloc_header(n, type_id, seq) {}
   };
   void register_values(sal::allocator& a)
   {
      a.register_type_ops<padding>();
      a.register_type_ops<key_val_node>();
      a.register_type_ops<value_chunk>();
      a.start_background_threads();
   }
}  // namespace

TEST_CASE("Large-value placement includes its embedded directory", "[api][large][placement]")
{
   using large_value_detail::plan;
   constexpr uint32_t S = sal::allocator_session::fresh_allocation_capacity;
   constexpr uint32_t C = large_value_detail::chunk_capacity;
   constexpr uint32_t H = sizeof(key_val_node);
   constexpr uint32_t T = 2 * MiB;
   CHECK_FALSE(plan(3 * MiB, 0, 3 * MiB + H).start_fresh);
   CHECK(plan(3 * MiB, 0, 3 * MiB + H - 1).start_fresh);
   CHECK_FALSE(plan(T, 0, MiB).start_fresh);
   CHECK(plan(T + 1, 0, MiB).start_fresh);
   CHECK(plan(C + MiB, 0, 16 * MiB).start_fresh);
   // A tail that ignores the embedded metadata would introduce either an
   // extra continuation or a second chunk just over 2 MiB.
   CHECK(plan(C + MiB, 0, MiB).start_fresh);
   CHECK(plan(C + MiB, 0, C - MiB).start_fresh);
   CHECK_FALSE(plan(C + MiB, 0, MiB + H + sizeof(chunk_entry)).start_fresh);
   CHECK_FALSE(plan(C + MiB, 0, C - MiB + H + sizeof(chunk_entry)).start_fresh);
   CHECK(plan(2 * C + 1, 0, large_value_detail::minimum_tail + H).start_fresh);
   CHECK_FALSE(plan(4, 0, 128).start_fresh);

   // Independently enumerate directory sizes and physically fill the head and
   // continuations. Exercise UINT32_MAX without allocating a multi-GiB input.
   for (uint16_t key : {uint16_t(0), uint16_t(1024)})
      for (uint32_t size : {0u, 65535u, 65536u, T, T + 1, S - H, S - H + 1, C + 1, C + T, 2 * C,
                            uint32_t(512 * MiB), UINT32_MAX})
         for (uint32_t space : {0u, 128u, uint32_t(MiB), T, T + 128, S})
         {
            INFO("key=" << key << " size=" << size << " space=" << space);
            auto enumerate = [&](uint32_t available)
            {
               large_value_detail::allocation_plan result{};
               for (uint32_t n = 0; n < 256; ++n)
               {
                  const uint32_t header = H + key + n * sizeof(chunk_entry);
                  if (available <= header)
                     break;
                  const uint32_t first = std::min(size, available - header);
                  uint64_t       left  = uint64_t(size) - first;
                  uint32_t       count = 0;
                  while (left)
                  {
                     left -= std::min<uint64_t>(left, C);
                     ++count;
                  }
                  if (count == n)
                  {
                     result = {first, uint16_t(count)};
                     break;
                  }
               }
               return result;
            };
            const auto actual   = plan(size, key, space);
            const auto expected = enumerate(actual.start_fresh ? S : space);
            REQUIRE(actual.first_size == expected.first_size);
            REQUIRE(actual.continuation_count == expected.continuation_count);
            REQUIRE(uint64_t(actual.first_size) + uint64_t(actual.continuation_count) * C >= size);
            if (actual.continuation_count)
            {
               REQUIRE(actual.first_size > 0);
               REQUIRE(H + key + actual.continuation_count * sizeof(chunk_entry) +
                           actual.first_size ==
                       (actual.start_fresh ? S : space));
            }
            if (size > S - H - key && space >= MiB)
            {
               auto score = [&](auto p, uint32_t skipped)
               {
                  uint64_t left  = size - p.first_size;
                  uint32_t large = p.first_size > T;
                  while (left)
                  {
                     const auto chunk = std::min<uint64_t>(left, C);
                     large += chunk > T;
                     left -= chunk;
                  }
                  return std::tuple{large, unsigned(p.continuation_count) + 1, skipped};
               };
               REQUIRE(actual.start_fresh ==
                       (score(enumerate(S), space) < score(enumerate(space), 0)));
            }
         }
}

TEST_CASE("Large-value physical packing and skipped-tail accounting", "[api][large][placement]")
{
   temp_directory temp;
   auto           cfg                       = config();
   cfg.compact_pinned_unused_threshold_mb   = 255;
   cfg.compact_unpinned_unused_threshold_mb = 255;
   sal::allocator alloc(temp.path, cfg, false);
   register_values(alloc);
   auto ses = alloc.create_session();
   SECTION("3 MiB records share a segment until the next will not fit")
   {
      const auto                                     input = payload(3 * MiB);
      std::vector<sal::smart_ptr<sal::alloc_header>> refs;
      for (int i = 0; i < 11; ++i)
         refs.emplace_back(ses, large_value_detail::allocate(*ses, {}, radix::as_bytes(input),
                                                             inline_value::vt_bytes));
      const auto first = alloc.resolve(refs[0].address()).second.segment();
      for (size_t i = 0; i < 10; ++i)
         REQUIRE(alloc.resolve(refs[i].address()).second.segment() == first);
      REQUIRE(alloc.resolve(refs[10].address()).second.segment() != first);
      const uint32_t live =
          10 * key_val_node::alloc_size({}, radix::as_bytes(input), inline_value::vt_bytes);
      for (int i = 0; i < 3; ++i)
      {
         ses->sync(sal::sync_type::none);
         bool found = false;
         for (const auto& segment : alloc.dump().segments)
            if (segment.segment_num == first)
            {
               found = true;
               REQUIRE(segment.is_finalized);
               REQUIRE(segment.freed_bytes ==
                       sal::segment_size - sal::mapped_memory::segment_footer_size - live);
            }
         REQUIRE(found);
      }
   }
   SECTION("multi-segment value chooses full plus small and reuses the final tail")
   {
      auto       fill        = ses->smart_alloc<padding>(uint32_t(16 * MiB), sal::alloc_hint{});
      const auto old_segment = alloc.resolve(fill.address()).second.segment();
      const auto input       = payload(large_value_detail::chunk_capacity + MiB);
      sal::smart_ptr<sal::alloc_header> value(
          ses,
          large_value_detail::allocate(*ses, {}, radix::as_bytes(input), inline_value::vt_bytes));
      auto lock = ses->lock();
      auto dir  = ses->get_ref<key_val_node>(value.address());
      REQUIRE(dir->continuation_count == 1);
      REQUIRE(dir->chunks()[0].start == sal::allocator_session::fresh_allocation_capacity -
                                            sizeof(key_val_node) - sizeof(chunk_entry));
      REQUIRE(dir->value_len - dir->chunks()[0].start ==
              MiB + sizeof(key_val_node) + sizeof(chunk_entry) - sizeof(value_chunk));
      REQUIRE(alloc.resolve(value.address()).second.segment() != old_segment);
      const auto final_segment = alloc.resolve(dir->chunks()[0].child).second.segment();
      REQUIRE(dir->first_bytes().size() == dir->chunks()[0].start);
      REQUIRE(matches(
          {reinterpret_cast<const char*>(dir->first_bytes().data()), dir->first_bytes().size()},
          std::string_view(input).substr(0, dir->chunks()[0].start)));
      sal::smart_ptr<sal::alloc_header> next(
          ses, large_value_detail::allocate(*ses, {}, radix::as_bytes(input).first(3 * MiB),
                                            inline_value::vt_bytes));
      REQUIRE(alloc.resolve(next.address()).second.segment() == final_segment);
      ses->sync(sal::sync_type::none);
      for (const auto& info : alloc.dump().segments)
         if (info.segment_num == old_segment)
            REQUIRE(info.freed_bytes ==
                    sal::segment_size - sal::mapped_memory::segment_footer_size - 16 * MiB);
   }
   SECTION("small values consume a useful tail then continue")
   {
      auto fill = ses->smart_alloc<padding>(
          uint32_t(sal::allocator_session::fresh_allocation_capacity - MiB), sal::alloc_hint{});
      const auto                        input = payload(2 * MiB);
      sal::smart_ptr<sal::alloc_header> value(
          ses,
          large_value_detail::allocate(*ses, {}, radix::as_bytes(input), inline_value::vt_bytes));
      auto lock = ses->lock();
      auto dir  = ses->get_ref<key_val_node>(value.address());
      REQUIRE(dir->continuation_count == 1);
      REQUIRE(dir->chunks()[0].start == MiB - sizeof(key_val_node) - sizeof(chunk_entry));
      REQUIRE(alloc.resolve(value.address()).second.segment() ==
              alloc.resolve(fill.address()).second.segment());
   }
}

TEST_CASE("Unified heads embed the first payload and own only continuations",
          "[api][large][layout]")
{
   temp_directory temp;
   auto           cfg          = config();
   cfg.checksum_on_commit      = true;
   cfg.write_protect_on_commit = true;
   sal::allocator alloc(temp.path, cfg, false);
   register_values(alloc);
   auto                         ses    = alloc.create_session();
   const auto                   input  = payload(large_value_detail::chunk_capacity + 37);
   const auto                   before = ses->get_total_allocated_objects();
   sal::smart_ptr<key_val_node> head(
       ses, large_value_detail::allocate(*ses, {}, radix::as_bytes(input), inline_value::vt_bytes));
   REQUIRE(ses->get_total_allocated_objects() - before == 2);
   std::optional<sal::read_lock> pin(ses->lock());
   auto                          h = ses->get_ref<key_val_node>(head.address());
   REQUIRE(h->continuation_count == 1);
   REQUIRE(h->value_len == input.size());
   const auto boundary = h->chunks().front().start;
   const auto stored   = radix::value_detail::from_external(*h.obj(), head.address());
   {
      auto first = radix::value_detail::read(*ses, stored, 7, 41, pin);
      REQUIRE(first.is_borrowed());
      REQUIRE(first.data() == reinterpret_cast<const char*>(h->first_bytes().data()) + 7);
      REQUIRE(matches(first.view(), std::string_view(input).substr(7, 41)));
      auto ending = radix::value_detail::read(*ses, stored, boundary - 41, 41, pin);
      REQUIRE(ending.is_borrowed());  // ending exactly at the boundary stays in the head
      REQUIRE(matches(ending.view(), std::string_view(input).substr(boundary - 41, 41)));
   }
   {
      auto crossed = radix::value_detail::read(*ses, stored, boundary - 1, 2, pin);
      REQUIRE_FALSE(crossed.is_borrowed());
      REQUIRE(matches(crossed.view(), std::string_view(input).substr(boundary - 1, 2)));
   }
   // Construction completes before the session freezes/checksums the first
   // segment, even though child allocation already finalized that segment.
   ses->sync(sal::sync_type::msync_sync);
   pin.emplace(ses->lock());
   auto full = radix::value_detail::read(*ses, stored, 0, adb::read_all, pin);
   REQUIRE(matches(full.view(), input));
}

TEST_CASE("Big rejects a value above UINT32_MAX before touching its bytes", "[api][large][limit]")
{
   static_assert(E::max_value_size == UINT32_MAX);
   temp_directory temp;
   auto           db = E::database::open(temp.path, psitri::open_mode::create_only, config());
   auto           ws = db->start_write_session();
   auto           tx = ws->start_transaction(0);
   tx.upsert("keep", "unchanged");
   struct inaccessible_input
   {
      static constexpr size_t size() { return size_t(UINT32_MAX) + 1; }
      void* data = ::mmap(nullptr, size(), PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
      ~inaccessible_input()
      {
         if (data != MAP_FAILED)
            ::munmap(data, size());
      }
   } input;
   REQUIRE(input.data != MAP_FAILED);
   const std::string_view oversized(static_cast<const char*>(input.data), input.size());
   REQUIRE_THROWS_AS(tx.upsert("keep", oversized), std::invalid_argument);
   REQUIRE_THROWS_AS(tx.upsert("new", oversized), std::invalid_argument);
   REQUIRE(tx.get("keep") == "unchanged");
   REQUIRE_FALSE(tx.get("new"));
   tx.commit();
   REQUIRE(ws->get_root(0).get("keep") == "unchanged");
}

TEST_CASE("Big values through 512 MiB allocate and free among small records", "[api][large][512mb]")
{
   temp_directory temp;
   const auto     input = payload(512 * MiB);
   const uint64_t C     = sal::allocator_session::fresh_allocation_capacity - sizeof(key_val_node);
   const auto     cfg   = config();
   {
      auto db = E::database::open(temp.path, psitri::open_mode::create_only, cfg);
      auto ws = db->start_write_session();
      for (uint64_t size : std::initializer_list<uint64_t>{
               65536, 2 * MiB - 1, 2 * MiB, 2 * MiB + 1, 3 * MiB, 8 * MiB, 15 * MiB, 16 * MiB,
               C - 1, C, C + 1, 64 * MiB + 17, 128 * MiB + 11, 512 * MiB})
      {
         INFO("value bytes: " << size);
         const auto expected = std::string_view(input).substr(0, size);
         {
            auto tx = ws->start_transaction(0);
            for (int i = 0; i < 40; ++i)
               tx.upsert("small/" + std::to_string(i), std::string(31 + i * 127, char(i)));
            tx.upsert("large", expected);
            for (int i = 0; i < 40; i += 2)
               REQUIRE(tx.remove("small/" + std::to_string(i)) == 0);
            tx.upsert("after", "tail reuse");
            REQUIRE(matches(*tx.get("large", size - 101, 101), expected.substr(size - 101)));
            tx.commit();
         }
         {
            auto snapshot = ws->get_root(0);
            auto cur      = snapshot.snapshot_cursor();
            REQUIRE(cur.find("large"));
            REQUIRE(cur.value_size() == int64_t(size));
            REQUIRE(cur.get("large", 0, 0)->empty());
            REQUIRE(cur.value(size)->empty());
            REQUIRE(cur.value(adb::read_all)->empty());
            REQUIRE(matches(*snapshot.get("large", size - 7, adb::read_all),
                            expected.substr(size - 7)));
            REQUIRE_FALSE(cur.get("missing", 0, 9));
            // Inspect every actual chunk boundary, including odd first/final sizes.
            std::vector<uint64_t> boundaries;
            {
               auto ses  = db->engine_for_testing().session();
               auto lock = ses->lock();
               ses->for_each_live_object(
                   [&](auto, auto, const sal::alloc_header* h)
                   {
                      if (h->type() == key_val_node::type_id)
                         for (const auto& e : static_cast<const key_val_node*>(h)->chunks())
                            boundaries.push_back(e.start);
                   });
            }
            for (auto boundary : boundaries)
            {
               INFO("chunk boundary: " << boundary);
               const uint64_t offset = boundary - 11;
               REQUIRE(matches(*cur.value(offset, 29), expected.substr(offset, 29)));
               int calls = 0;
               REQUIRE(cur.get_value(
                   [&](std::string_view v)
                   {
                      ++calls;
                      REQUIRE(matches(v, expected.substr(offset, 29)));
                   },
                   offset, 29));
               REQUIRE(calls == 1);
               auto crossing = cur.pinned_value(offset, 29);
               REQUIRE_FALSE(crossing.is_borrowed());
               auto copied = crossing;
               auto moved  = std::move(crossing);
               REQUIRE(matches(copied.view(), expected.substr(offset, 29)));
               REQUIRE(matches(moved.view(), expected.substr(offset, 29)));
               auto within = cur.pinned_value(boundary, 1);
               REQUIRE(within.is_borrowed());
               REQUIRE(matches(within.view(), expected.substr(boundary, 1)));
            }
            // Full owning reads assemble the complete logical value, even at 512 MiB.
            {
               auto full = cur.value();
               REQUIRE(full.has_value());
               REQUIRE(matches(*full, expected));
            }
            REQUIRE_THROWS_AS(cur.get_value([](std::string_view)
                                            { throw std::runtime_error("callback"); }, size / 2, 1),
                              std::runtime_error);
            {
               auto tx = ws->start_transaction(0);
               tx.upsert("large", "replacement");
               tx.upsert("after", "new tail");
               tx.commit();
            }
            REQUIRE(matches(*cur.value(size - 113, 113), expected.substr(size - 113)));
            REQUIRE(ws->get_root(0).get("large") == "replacement");
         }  // release snapshot and cursor: head and all its continuations become reclaimable
         REQUIRE(db->wait_for_compactor(std::chrono::seconds(20)));
         {
            auto tx = ws->start_transaction(0);
            REQUIRE(tx.remove("large") == 0);
            REQUIRE(tx.remove("after") == 0);
            for (int i = 1; i < 40; i += 2)
            {
               REQUIRE(tx.get("small/" + std::to_string(i)) == std::string(31 + i * 127, char(i)));
               REQUIRE(tx.remove("small/" + std::to_string(i)) == 0);
            }
            tx.commit();
         }
         REQUIRE(db->wait_for_compactor(std::chrono::seconds(20)));
         REQUIRE(db->engine_for_testing().object_count() <= 1);  // at most the empty leftmost page
      }
      // Durable large value plus a normal record for reopen; abort another large lineage.
      auto tx = ws->start_transaction(0);
      tx.upsert("persist", std::string_view(input).substr(0, 64 * MiB + 7));
      tx.upsert("small", "survivor");
      tx.commit();
      {
         auto aborted = ws->start_transaction(0);
         auto frame   = aborted.sub_transaction();
         frame.upsert("persist", std::string_view(input).substr(0, 128 * MiB));
         REQUIRE(frame.get("persist", 128 * MiB - 1, 1)->size() == 1);
         frame.abort();
         REQUIRE(aborted.get("persist", 64 * MiB + 7, 1)->empty());
         aborted.upsert("unpublished", std::string_view(input).substr(0, 64 * MiB));
         aborted.abort();
      }
      REQUIRE(db->wait_for_compactor(std::chrono::seconds(20)));
   }
   for (auto mode : {psitri::open_mode::open_existing, psitri::open_mode::read_only})
   {
      auto db = E::database::open(temp.path, mode, cfg);
      if (mode == psitri::open_mode::open_existing)
      {
         const auto count = db->underlying_allocator().total_allocated_objects();
         db->underlying_allocator().reset_reference_counts();
         REQUIRE(db->underlying_allocator().total_allocated_objects() == count);
         db->underlying_allocator().recover();
         REQUIRE(db->underlying_allocator().total_allocated_objects() == count);
      }
      auto rs  = db->start_read_session();
      auto cur = rs->create_cursor(0);
      REQUIRE(cur.get("small") == "survivor");
      REQUIRE_FALSE(cur.get("unpublished"));
      REQUIRE(matches(*cur.get("persist", 31 * MiB, 3 * MiB),
                      std::string_view(input).substr(31 * MiB, 3 * MiB)));
      REQUIRE(cur.find("persist"));
      REQUIRE(cur.value_size() == 64 * MiB + 7);
   }
}

TEST_CASE("Chunked long keys and snapshots survive observed compaction", "[api][large][compaction]")
{
   bool compact_head = false;
   SECTION("relocate the directory with the first payload")
   {
      compact_head = true;
   }
   SECTION("relocate a continuation independently") {}
   temp_directory temp;
   auto           cfg                       = config();
   cfg.compact_pinned_unused_threshold_mb   = 1;
   cfg.compact_unpinned_unused_threshold_mb = 1;
   auto db = E::database::open(temp.path, psitri::open_mode::create_only, cfg);
   db->underlying_allocator().stop_compactor_thread();  // capture both original locations first
   auto              ws    = db->start_write_session();
   const auto        input = payload(large_value_detail::chunk_capacity + 3 * MiB);
   const std::string key   = "large/" + std::string(1018, 'k');
   {
      auto tx = ws->start_transaction(0);
      if (compact_head)
         tx.upsert("head-garbage", std::string(16 * MiB, 'g'));
      tx.upsert("a", "small");
      tx.upsert(key, input);
      if (compact_head)
         REQUIRE(tx.remove("head-garbage") == 0);  // reclaim space beside the live head
      tx.create_subtree("nested").upsert("blob", std::string_view(input).substr(0, 3 * MiB));
      tx.commit();
   }
   auto snapshot = ws->get_root(0);
   auto cur      = snapshot.snapshot_cursor();
   REQUIRE(cur.find(key));
   REQUIRE(matches(*snapshot.get_subtree("nested").get("blob", MiB, 7),
                   std::string_view(input).substr(MiB, 7)));
   sal::ptr_address head     = sal::null_ptr_address;
   sal::ptr_address last     = sal::null_ptr_address;
   uint32_t         boundary = 0;
   auto             ses      = db->engine_for_testing().session();
   {
      auto pin = ses->lock();
      ses->for_each_live_object(
          [&](auto, auto, const sal::alloc_header* h)
          {
             if (h->type() == key_val_node::type_id &&
                 static_cast<const key_val_node*>(h)->chunked())
             {
                head     = h->address();
                last     = static_cast<const key_val_node*>(h)->chunks().back().child;
                boundary = static_cast<const key_val_node*>(h)->chunks().front().start;
             }
          });
   }
   REQUIRE(head != sal::null_ptr_address);
   REQUIRE(last != sal::null_ptr_address);
   const auto            target       = compact_head ? head : last;
   const auto            old_location = db->underlying_allocator().resolve(target).second;
   std::atomic<unsigned> reads        = 0;
   std::atomic<bool>     failed       = false;
   std::jthread          reader(
       [&](std::stop_token stop)
       {
          try
          {
             auto rs   = db->start_read_session();
             auto view = rs->create_cursor(0);  // own session and immutable old root
             while (!stop.stop_requested())
             {
                auto part = view.get(key, boundary - 4096, 8192);
                if (!part || !matches(*part, std::string_view(input).substr(boundary - 4096, 8192)))
                   failed = true;
                ++reads;
             }
          }
          catch (...)
          {
             failed = true;
          }
       });
   while (reads.load() == 0 && !failed.load())
      std::this_thread::yield();
   REQUIRE_FALSE(failed.load());
   const auto garbage = payload(24 * MiB);
   const auto keep    = payload(3 * MiB);
   {
      auto tx = ws->start_transaction(0);
      for (unsigned i = 0; i < 8; ++i)
      {
         tx.upsert("garbage/" + std::to_string(i), garbage);
         tx.upsert("keep/" + std::to_string(i), keep);
      }
      // Force page splitting with external values and external full-key comparisons.
      for (unsigned i = 0; i < 40; ++i)
         tx.upsert(std::to_string(i) + std::string(900, 'z'),
                   std::string_view(input).substr(0, 65536));
      tx.commit();
   }
   {
      auto tx = ws->start_transaction(0);
      for (unsigned i = 0; i < 8; ++i)
         REQUIRE(tx.remove("garbage/" + std::to_string(i)) == 0);
      tx.upsert(key, "current replacement");
      tx.commit();
   }
   ses->next_allocation_segment();
   ses->sync(sal::sync_type::none);
   db->underlying_allocator().start_background_threads();
   const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
   while (db->underlying_allocator().resolve(target).second == old_location &&
          std::chrono::steady_clock::now() < deadline)
   {
      REQUIRE(matches(*cur.value(7, 13), std::string_view(input).substr(7, 13)));
      REQUIRE(matches(*cur.value(input.size() - 13, 13),
                      std::string_view(input).substr(input.size() - 13)));
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
   }
   const bool moved = db->underlying_allocator().resolve(target).second != old_location;
   reader.request_stop();
   reader.join();
   if (!moved)
   {
      // Say why the compactor left the object where it was: which segment it
      // is in and what the compactor sees for every segment.
      const auto segment_of = [](auto location) { return location.absolute_address() / sal::segment_size; };
      std::string report = "target stayed in segment " + std::to_string(segment_of(old_location)) +
                           (compact_head ? " (head)" : " (continuation)") +
                           "\n seg  alloc_pos     freed  pinned  read_only  finalized  free\n";
      for (const auto& seg : db->underlying_allocator().dump().segments)
      {
         char line[128];
         std::snprintf(line, sizeof(line), "%4u %10lld %9llu  %6d  %9d  %9d  %4d\n",
                       unsigned(*seg.segment_num), (long long)seg.alloc_pos,
                       (unsigned long long)seg.freed_bytes, int(seg.is_pinned),
                       int(seg.is_read_only), int(seg.is_finalized), int(seg.is_free));
         report += line;
      }
      UNSCOPED_INFO(report);
   }
   REQUIRE(reads.load() > 0);
   REQUIRE_FALSE(failed.load());
   REQUIRE(moved);  // exercise actual relocation, not merely an enabled compactor
   REQUIRE(matches(*cur.value(), input));
   REQUIRE(ws->get_root(0).get(key) == "current replacement");
   auto now = ws->get_root(0).snapshot_cursor();
   for (unsigned i = 0; i < 40; ++i)
      REQUIRE(matches(*now.get(std::to_string(i) + std::string(900, 'z')),
                      std::string_view(input).substr(0, 65536)));
}

TEST_CASE("Legacy direct views assemble large values without truncation", "[api][large]")
{
   temp_directory    temp;
   const auto        input = payload(large_value_detail::chunk_capacity + 37);
   radix::radix_tree tree(temp.path, true, config());
   REQUIRE(tree.upsert("a", "first") == -1);
   REQUIRE(tree.upsert("large", input) == -1);
   REQUIRE(tree.upsert("z", "last") == -1);
   std::string_view out;
   REQUIRE(tree.find("large", out));
   REQUIRE(matches(out, input));
   std::string key;
   REQUIRE(tree.lower_bound("l", key, out));
   REQUIRE(key == "large");
   REQUIRE(matches(out, input));
   REQUIRE(tree.upper_bound("m", key, out));
   REQUIRE(key == "large");
   REQUIRE(matches(out, input));
   unsigned calls = 0;
   tree.for_each_inorder(
       [&](std::string_view k, std::string_view v)
       {
          ++calls;
          if (k == "large")
             REQUIRE(matches(v, input));
       });
   REQUIRE(calls == 3);
   radix::radix_tree::cursor cursor(&tree);
   REQUIRE(cursor.find("large"));
   {
      auto pin = tree.session()->lock();
      REQUIRE(matches(cursor.value(pin), input));
   }
   std::string buffer;
   REQUIRE(cursor.get("large", &buffer) == int64_t(input.size()));
   REQUIRE(matches(buffer, input));
   REQUIRE(tree.upsert("large", "short") == int64_t(input.size()));
}
