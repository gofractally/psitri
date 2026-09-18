#include "common.hpp"
#if FAIR_dwal
#include "dwal.hpp"
#elif FAIR_small || FAIR_large
#include "psitri.hpp"
#elif FAIR_mdbx
#include "mdbx.hpp"
#elif FAIR_rocksdb
#include "rocksdb.hpp"
#elif FAIR_sqlite
#include "sqlite.hpp"
#elif FAIR_duckdb
#include "duckdb.hpp"
#else
#error Select a benchmark adapter
#endif
#include <charconv>
#include <cmath>
#include <iomanip>
#include <sys/resource.h>
#include <sys/stat.h>

using namespace fair_bank;
namespace
{
constexpr uint64_t initial_balance = 1'000'000'000'000ULL;
#ifdef FAIR_API_PAIR
constexpr auto workload = "api-bank-v1";
#else
constexpr auto workload = "fair-bank-v2";
#endif
uint64_t integer(std::string_view text)
{
   uint64_t n;
   auto [p, ec] = std::from_chars(text.data(), text.data() + text.size(), n);
   require(ec == std::errc{} && p == text.data() + text.size(), "Invalid unsigned integer");
   return n;
}
Config parse(int argc, char **argv)
{
   Config c;
   for (int i = 1; i < argc; ++i)
   {
      std::string arg = argv[i];
      require(i + 1 < argc, "Missing value for " + arg);
      std::string value = argv[++i];
      if (arg == "--path")
         c.path = value;
      else if (arg == "--result")
         c.result = value;
      else if (arg == "--durability")
      {
         require(std::find(std::begin(durability_levels), std::end(durability_levels), value) !=
                     std::end(durability_levels),
                 "Unknown durability");
         c.durability = value;
      }
      else if (arg == "--accounts")
         c.accounts = integer(value);
      else if (arg == "--transfers")
         c.transfers = integer(value);
      else if (arg == "--warmup")
         c.warmup = integer(value);
      else if (arg == "--batch")
         c.batch = integer(value);
      else if (arg == "--readers")
         c.readers = integer(value);
      else if (arg == "--reads-per-snapshot")
         c.reads_per_snapshot = integer(value);
      else if (arg == "--read-batches")
         c.read_batches = integer(value);
      else if (arg == "--seed")
         c.seed = integer(value);
      else if (arg == "--value-bytes")
         c.value_bytes = integer(value);
      else if (arg == "--cache-mib")
         c.cache_mib = integer(value);
      else if (arg == "--map-gib")
         c.map_gib = integer(value);
      else
         throw std::runtime_error("Unknown option " + arg);
   }
   require(!c.durability.empty() && Store::supports(c.durability),
           "Choose an explicitly supported --durability tier; inspect --describe");
   require(!c.path.empty() && !c.result.empty(),
           "--path NEW_DIRECTORY and --result NEW_FILE are required");
   require(!fs::exists(c.path) && !fs::exists(c.result),
           "Refusing to overwrite a database or result");
   require(c.accounts >= 2 && c.accounts <= 100'000'000 && c.transfers > 0 &&
               c.transfers <= 10'000'000'000ULL,
           "accounts must be 2..100M; transfers 1..10B");
   require(c.warmup <= 10'000'000'000ULL && c.batch > 0 && c.batch <= 1'000'000,
           "Invalid warmup/batch");
   require(c.readers > 0 && c.readers <= 64 && c.reads_per_snapshot > 0 &&
               c.reads_per_snapshot <= 1'000'000 && c.read_batches > 0 &&
               c.read_batches <= 1'000'000,
           "Invalid reader settings");
   require(c.value_bytes >= 8 && c.value_bytes <= 1'048'576 && c.cache_mib >= 16 &&
               c.cache_mib <= 1'048'576 && c.map_gib > 0 && c.map_gib <= 32768,
           "Invalid size settings");
   return c;
}
template <class F> void trace(const Config &c, uint64_t count, uint64_t seed, F &&fn)
{
   Random random{seed};
   for (uint64_t i = 0; i < count; ++i)
   {
      auto from = random.next() % c.accounts;
      auto to = random.next() % (c.accounts - 1);
      if (to >= from)
         ++to; // no self transfers or skipped work
      fn(from, to);
   }
}
void model_apply(const Config &c, std::vector<uint64_t> &model, uint64_t count, uint64_t seed)
{
   trace(c, count, seed,
         [&](auto from, auto to)
         {
            require(model[from] > 0, "Model exhausted account");
            --model[from];
            ++model[to];
         });
}
uint64_t validate(Store &store, const Config &c, const std::vector<uint64_t> &model)
{
   auto count = store.count();
   require(count == c.accounts, "Account cardinality changed: expected " +
                std::to_string(c.accounts) + ", observed " + std::to_string(count));
   Store::Reader reader(store);
   uint64_t hash = 14695981039346656037ULL;
   unsigned __int128 sum = 0;
   // Bounded snapshots also let large datasets reclaim storage between batches.
   for (uint64_t begin = 0; begin < c.accounts; begin += c.reads_per_snapshot)
      reader.read(
          [&](auto &tx)
          {
             for (auto i = begin; i < std::min(c.accounts, begin + c.reads_per_snapshot); ++i)
             {
                auto value = tx.get(encode(i));
                auto balance = decode(value, c.value_bytes);
                require(balance == model[i], "Account mismatch at " + std::to_string(i));
                sum += balance;
                for (unsigned char byte : value)
                {
                   hash ^= byte;
                   hash *= 1099511628211ULL;
                }
             }
          });
   require(sum == static_cast<unsigned __int128>(initial_balance) * c.accounts,
           "Balance conservation failed");
   return hash;
}
void contract_check(Store &store, const Config &c)
{
   const auto key = encode(0), original = encode(initial_balance, c.value_bytes);
   const auto changed = encode(initial_balance - 1, c.value_bytes);
   store.write(
       [&](auto &tx)
       {
          tx.put(key, changed);
          require(tx.get(key) == changed, "Read-your-writes failed");
          tx.put(key, original);
          require(tx.get(key) == original, "Repeated-key write failed");
          tx.put(key, changed);
       },
       false);
   {
      Store::Reader reader(store);
      reader.read([&](auto &tx) { require(tx.get(key) == original, "Rollback failed"); });
   }
   std::latch acquired(1), published(1);
   std::exception_ptr reader_error, writer_error;
   std::thread reader_thread(
       [&]
       {
          bool announced = false;
          try
          {
             Store::Reader reader(store);
             reader.read(
                 [&](auto &tx)
                 {
                    require(tx.get(key) == original, "Snapshot initial read failed");
                    acquired.count_down();
                    announced = true;
                    published.wait();
                    require(tx.get(key) == original, "Snapshot changed after commit");
                 });
          }
          catch (...)
          {
             reader_error = std::current_exception();
             if (!announced)
                acquired.count_down();
          }
       });
   acquired.wait();
   try
   {
      store.write([&](auto &tx) { tx.put(key, changed); });
   }
   catch (...)
   {
      writer_error = std::current_exception();
   }
   published.count_down();
   reader_thread.join();
   if (reader_error)
      std::rethrow_exception(reader_error);
   if (writer_error)
      std::rethrow_exception(writer_error);
   {
      Store::Reader reader(store);
      reader.read([&](auto &tx) { require(tx.get(key) == changed, "New snapshot missed commit"); });
   }
   store.write([&](auto &tx) { tx.put(key, original); });
}
struct WriteStats
{
   uint64_t transfers = 0, commits = 0;
   std::vector<double> latency;
   double elapsed = 0;
};
WriteStats write_phase(Store &store, const Config &c, uint64_t count, uint64_t seed,
                       const std::atomic<bool> *failed = nullptr)
{
   WriteStats stats;
   stats.latency.reserve(size_t((count + c.batch - 1) / c.batch));
   Random random{seed};
   auto start = Clock::now();
   while (stats.transfers < count)
   {
      if (failed)
         require(!failed->load(), "Concurrent reader failed");
      auto batch_start = Clock::now();
      auto size = std::min(c.batch, count - stats.transfers);
      store.write(
          [&](auto &tx)
          {
             for (uint64_t i = 0; i < size; ++i)
             {
                auto from = random.next() % c.accounts;
                auto to = random.next() % (c.accounts - 1);
                if (to >= from)
                   ++to;
                auto source = encode(from), destination = encode(to);
                auto a = decode(tx.get(source), c.value_bytes),
                     b = decode(tx.get(destination), c.value_bytes);
                require(a > 0 && b < UINT64_MAX,
                        "Invalid account balance; transfers are never silently skipped");
                tx.put(source, encode(a - 1, c.value_bytes));
                tx.put(destination, encode(b + 1, c.value_bytes));
             }
          });
      stats.transfers += size;
      ++stats.commits;
      stats.latency.push_back(seconds(batch_start) * 1e6);
   }
   stats.elapsed = seconds(start);
   return stats;
}
struct ReadStats
{
   uint64_t reads = 0, checksum = 0;
   double elapsed = 0;
   WriteStats writer;
};
ReadStats read_phase(Store &store, const Config &c, bool mixed)
{
   std::latch ready(c.readers), go(1);
   std::atomic<bool> stop{false}, failed{false};
   std::mutex error_mutex;
   std::exception_ptr error;
   std::vector<uint64_t> counts(c.readers), sums(c.readers);
   std::vector<std::thread> threads;
   auto save_error = [&]
   {
      std::lock_guard lock(error_mutex);
      if (!error)
         error = std::current_exception();
      failed.store(true);
      stop.store(true);
   };
   // All sessions are created and destroyed on their reader thread.
   try
   {
      for (uint64_t index = 0; index < c.readers; ++index)
         threads.emplace_back(
             [&, index]
             {
                bool announced = false;
                try
                {
                   Store::Reader reader(store);
                   Random random{c.seed ^ (0xf00dULL + index * 0x9e3779b97f4a7c15ULL)};
                   ready.count_down();
                   announced = true;
                   go.wait();
                   for (uint64_t batch = 0;
                        !failed.load() && (mixed ? !stop.load() : batch < c.read_batches); ++batch)
                      reader.read(
                          [&](auto &tx)
                          {
                             for (uint64_t i = 0; i < c.reads_per_snapshot; ++i)
                             {
                                auto balance = decode(tx.get(encode(random.next() % c.accounts)),
                                                      c.value_bytes);
                                require(balance <= initial_balance + c.warmup + 2 * c.transfers,
                                        "Impossible reader balance");
                                sums[index] += balance;
                                ++counts[index];
                             }
                          });
                }
                catch (...)
                {
                   save_error();
                   if (!announced)
                      ready.count_down();
                }
             });
   }
   catch (...)
   {
      failed.store(true);
      stop.store(true);
      go.count_down();
      for (auto &thread : threads)
         thread.join();
      throw;
   }
   ready.wait();
   ReadStats stats;
   auto start = Clock::now();
   go.count_down();
   try
   {
      if (mixed && !failed.load())
         stats.writer = write_phase(store, c, c.transfers, c.seed ^ 0x2222, &failed);
   }
   catch (...)
   {
      save_error();
   }
   if (mixed)
      stop.store(true);
   for (auto &thread : threads)
      thread.join();
   stats.elapsed = seconds(start);
   if (error)
      std::rethrow_exception(error);
   for (uint64_t i = 0; i < c.readers; ++i)
   {
      stats.reads += counts[i];
      stats.checksum += sums[i];
   }
   require(stats.reads > 0, "No reader work measured; increase transfer count");
   return stats;
}
std::pair<uint64_t, uint64_t> disk_usage(const fs::path &path)
{
   uint64_t logical = 0, allocated = 0;
   for (auto &entry : fs::recursive_directory_iterator(path))
      if (entry.is_regular_file())
      {
         struct stat st;
         require(::stat(entry.path().c_str(), &st) == 0, "Unable to stat database file");
         logical += uint64_t(st.st_size);
         allocated += uint64_t(st.st_blocks) * 512;
      }
   return {logical, allocated};
}
void write_result(const Config &c, const std::vector<Metric> &metrics, uint64_t hash,
                  const WriteStats &solo, const ReadStats &mixed, const ReadStats &read)
{
   auto temp = c.result;
   temp += ".tmp";
   std::ofstream out(temp);
   out << std::setprecision(17) << "{\n\"schema_version\":1,\"workload\":" << json(workload)
       << ",\"engine\":" << json(engine_name) << ",\"version\":" << json(Store::version())
       << ",\"settings\":" << json(Store::settings(c)) << ",\"status\":\"passed\",\n\"config\":{";
   out << "\"accounts\":" << c.accounts << ",\"transfers\":" << c.transfers
       << ",\"warmup\":" << c.warmup << ",\"batch\":" << c.batch << ",\"readers\":" << c.readers
       << ",\"reads_per_snapshot\":" << c.reads_per_snapshot
       << ",\"read_batches\":" << c.read_batches << ",\"seed\":" << c.seed
       << ",\"value_bytes\":" << c.value_bytes << ",\"cache_mib\":" << c.cache_mib
       << ",\"map_gib\":" << c.map_gib << ",\"durability\":" << json(c.durability) << "},\n";
   out << "\"validation\":{\"read_your_writes\":true,\"rollback\":true,\"snapshot\":true,\"exact_"
          "accounts\":true,"
          "\"balance\":true,\"reopen\":true,\"final_hash\":"
       << json(std::to_string(hash)) << "},\n";
   out << "\"counts\":{\"write_transfers\":" << solo.transfers
       << ",\"write_commits\":" << solo.commits << ",\"mixed_transfers\":" << mixed.writer.transfers
       << ",\"mixed_commits\":" << mixed.writer.commits << ",\"mixed_reads\":" << mixed.reads
       << ",\"read_only_reads\":" << read.reads
       << ",\"reader_checksum\":" << json(std::to_string(mixed.checksum + read.checksum))
       << "},\n\"metrics\":[";
   bool first = true;
   for (const auto &m : metrics)
   {
      require(std::isfinite(m.value) && m.value >= 0, "Invalid metric");
      if (!first)
         out << ',';
      first = false;
      out << "{\"engine\":" << json(engine_name) << ",\"name\":" << json(m.name)
          << ",\"unit\":" << json(m.unit) << ",\"direction\":" << json(m.direction)
          << ",\"value\":" << m.value << '}';
   }
   out << "]\n}\n";
   out.close();
   require(bool(out), "Writing result failed");
   fs::rename(temp, c.result);
}
} // namespace

int main(int argc, char **argv)
{
   if (argc == 2 && std::string_view(argv[1]) == "--describe")
   {
      std::cout << "{\"engine\":" << json(engine_name) << ",\"version\":" << json(Store::version())
                << ",\"workload\":" << json(workload) << ",\"durability\":[";
      bool first = true;
      for (auto level : durability_levels)
         if (Store::supports(level))
         {
            if (!first)
               std::cout << ',';
            first = false;
            std::cout << json(level);
         }
      std::cout << "]}\n";
      return 0;
   }
   try
   {
      auto c = parse(argc, argv);
      fs::create_directories(c.path);
      // psitri create_only owns creating its directory.
#if FAIR_small || FAIR_large || FAIR_dwal
      fs::remove(c.path);
#endif
      std::vector<uint64_t> model(c.accounts, initial_balance);
      std::vector<Metric> metrics;
      auto add =
          [&](std::string name, double value, std::string unit, std::string direction = "higher")
      { metrics.push_back({std::move(name), std::move(unit), std::move(direction), value}); };
      auto report_write = [&](const std::string &name, WriteStats &stats)
      {
         add(name, stats.transfers / stats.elapsed, "transfers/s");
         std::sort(stats.latency.begin(), stats.latency.end());
         auto percentile = [&](double q)
         { return stats.latency[size_t(std::ceil(q * stats.latency.size())) - 1]; };
         add(name + "_batch_p50", percentile(.50), "us", "lower");
         add(name + "_batch_p99", percentile(.99), "us", "lower");
      };
      auto store = std::make_unique<Store>(c, false);
      std::cerr << engine_name << " " << Store::version() << "\n"
                << Store::settings(c) << "\nload\n";
      auto start = Clock::now();
      for (uint64_t begin = 0; begin < c.accounts; begin += c.batch)
         store->write(
             [&](auto &tx)
             {
                for (auto i = begin; i < std::min(c.accounts, begin + c.batch); ++i)
                   tx.put(encode(i), encode(initial_balance, c.value_bytes));
             });
      add("load", c.accounts / seconds(start), "accounts/s");
      std::cerr << "contract checks\n";
      contract_check(*store, c);
      validate(*store, c, model);
      std::cerr << "warmup\n";
      write_phase(*store, c, c.warmup, c.seed ^ 0x7777);
      model_apply(c, model, c.warmup, c.seed ^ 0x7777);
      validate(*store, c, model);
      std::cerr << "write\n";
      auto solo = write_phase(*store, c, c.transfers, c.seed ^ 0x1111);
      report_write("write", solo);
      model_apply(c, model, c.transfers, c.seed ^ 0x1111);
      validate(*store, c, model);
      std::cerr << "mixed\n";
      auto mixed = read_phase(*store, c, true);
      report_write("mixed_write_active", mixed.writer);
      add("mixed_write", mixed.writer.transfers / mixed.elapsed, "transfers/s");
      add("mixed_read", mixed.reads / mixed.elapsed, "reads/s");
      model_apply(c, model, c.transfers, c.seed ^ 0x2222);
      auto hash = validate(*store, c, model);
      std::cerr << "read\n";
      auto read = read_phase(*store, c, false);
      add("read", read.reads / read.elapsed, "reads/s");
      auto [logical, allocated] = disk_usage(c.path);
      add("disk_before_close", double(logical), "bytes", "lower");
      add("allocated_before_close", double(allocated), "bytes", "lower");
      std::cerr << "close/reopen validation\n";
      start = Clock::now();
      store.reset();
      add("close", seconds(start), "seconds", "lower");
      std::tie(logical, allocated) = disk_usage(c.path);
      add("disk_after_close", double(logical), "bytes", "lower");
      add("allocated_after_close", double(allocated), "bytes", "lower");
      start = Clock::now();
      store = std::make_unique<Store>(c, true);
      add("reopen", seconds(start), "seconds", "lower");
      require(validate(*store, c, model) == hash, "Reopen hash changed");
      store.reset();
      struct rusage usage;
      require(getrusage(RUSAGE_SELF, &usage) == 0, "getrusage failed");
      add("peak_rss",
          double(usage.ru_maxrss) *
#ifdef __APPLE__
              1,
#else
              1024,
#endif
          "bytes", "lower");
#ifdef FAIR_CONTRACT_ONLY
      // Functional regression target: exercise the same contracts, but produce
      // no throughput or latency evidence from a potentially busy developer host.
      std::ofstream result(c.result);
      result << "{\"workload\":\"api-contract-v1\",\"status\":\"passed\","
             << "\"engine\":\"" << engine_name << "\",\"final_hash\":" << hash << "}\n";
#else
      write_result(c, metrics, hash, solo, mixed, read);
#endif
      std::cout << "PASS " << engine_name << " final_hash=" << hash << "\n";
      return 0;
   }
   catch (const std::exception &error)
   {
      std::cerr << "FAIL: " << error.what() << '\n';
      return 1;
   }
}
