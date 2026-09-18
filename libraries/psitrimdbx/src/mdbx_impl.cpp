/** @file mdbx_impl.cpp
 *  @brief PsiTri-backed MDBX C and C++ API implementation.
 */
// #define SAL_STALL_DEBUG 1              // [stall-debug] OFF for clean baseline (per-op fflush confound)
#include <mdbx.h>
#include <mdbx.h++>

// ════════════════════════════════════════════════════════════════════
// Storage backend selection.
//
// The MDBX mapping (dbi namespace prefixing, DUPSORT composite-key
// encoding, cursor ops, txn lifecycle) is engine-agnostic. Only the
// backend binding lives behind PSITRIMDBX_BACKEND_DELTA_ART: the new
// fast model is the delta_art tiered engine (start_ver hot delta over a
// sal_art_ref dwal cold base). It exposes ONE ordered KV space with
// bidirectional, mergeable cursors — exactly what this single-keyspace
// shim needs. The legacy psitri direct/dwal backend has been dropped.
// ════════════════════════════════════════════════════════════════════
// Backend selection (pick one; default = delta_art tiered):
//   PSITRIMDBX_BACKEND_DELTA_ART     — start_ver hot delta over sal_art_ref dwal cold base (new fast model)
//   PSITRIMDBX_BACKEND_START_VER     — start_ver ALONE (no cold base / drain) — isolation/standalone
//   PSITRIMDBX_BACKEND_PSITRI        — legacy psitri DIRECT (b+tree on SAL, old algo) via the
//                                      psitri::dwal::psitri_engine adb adapter — a known-good baseline.
//   PSITRIMDBX_BACKEND_TIERED_PSITRI — start_ver hot delta over a legacy-psitri cold base
//                                      (tiered, no 32GB cap, sidesteps the sal_art_ref cold base).
#if !defined(PSITRIMDBX_BACKEND_TIERED_PSITRI)
#define PSITRIMDBX_BACKEND_TIERED_PSITRI 0
#endif
#if !defined(PSITRIMDBX_BACKEND_PSITRI)
#define PSITRIMDBX_BACKEND_PSITRI 0
#endif
#if !defined(PSITRIMDBX_BACKEND_START_VER)
#define PSITRIMDBX_BACKEND_START_VER 0
#endif
//   PSITRIMDBX_BACKEND_SAL_ART_REF  — sal_art_ref DIRECT (no tiering, no merger): the cold-base
//                                     engine as the WHOLE DB, written by the user thread. Isolates
//                                     whether the cold-base SIGBUS is the engine itself vs the
//                                     tiered drain/merger interaction.
#ifndef PSITRIMDBX_BACKEND_DWAL
#define PSITRIMDBX_BACKEND_DWAL 0
#endif
#if !defined(PSITRIMDBX_BACKEND_SAL_ART_REF)
#define PSITRIMDBX_BACKEND_SAL_ART_REF 0
#endif
//   PSITRIMDBX_BACKEND_START_REF     — start_ref DIRECT (part-family ART, refcount liveness, in-RAM)
#if !defined(PSITRIMDBX_BACKEND_START_REF)
#define PSITRIMDBX_BACKEND_START_REF 0
#endif
//   PSITRIMDBX_BACKEND_GCART         — gcart spec engine DIRECT (sal_hybrid 3-region FIX/SAL/BIG, refcount, node-level COW)
#if !defined(PSITRIMDBX_BACKEND_GCART)
#define PSITRIMDBX_BACKEND_GCART 0
#endif
//   PSITRIMDBX_BACKEND_DELTA_GCART   — start_ver hot delta over a gcart COLD base (tiered; hot absorbs churn, gcart stays dense)
#if !defined(PSITRIMDBX_BACKEND_DELTA_GCART)
#define PSITRIMDBX_BACKEND_DELTA_GCART 0
#endif
//   PSITRIMDBX_BACKEND_RCART         — rcart refcount baseline DIRECT (sal_hybrid FIX/SAL/BIG, refcount + hazard readers, compact reclaim)
#if !defined(PSITRIMDBX_BACKEND_RCART)
#define PSITRIMDBX_BACKEND_RCART 0
#endif
#if !PSITRIMDBX_BACKEND_TIERED_PSITRI && !PSITRIMDBX_BACKEND_PSITRI && \
    !PSITRIMDBX_BACKEND_START_VER && !PSITRIMDBX_BACKEND_SAL_ART_REF && \
    !PSITRIMDBX_BACKEND_START_REF && !PSITRIMDBX_BACKEND_GCART && \
    !PSITRIMDBX_BACKEND_DELTA_GCART && !PSITRIMDBX_BACKEND_RCART && !PSITRIMDBX_BACKEND_DWAL
#define PSITRIMDBX_BACKEND_DELTA_ART 1
#else
#define PSITRIMDBX_BACKEND_DELTA_ART 0
#endif
// derived: which engine families the chosen backend needs to pull in
#define PSITRIMDBX_USES_TIERED \
   (PSITRIMDBX_BACKEND_DELTA_ART || PSITRIMDBX_BACKEND_TIERED_PSITRI || PSITRIMDBX_BACKEND_DELTA_GCART)
#define PSITRIMDBX_USES_PSITRI (PSITRIMDBX_BACKEND_PSITRI || PSITRIMDBX_BACKEND_TIERED_PSITRI)

#include <sal/config.hpp>    // sal::open_mode
#include <sal/diag.hpp>      // sal::env cached env reader + PSITRI_DEBUG gating
#include <start_ver/kv.hpp>  // start_ver::kv::engine (hot delta / standalone store)
#if PSITRIMDBX_BACKEND_START_REF
#include <start_ref/kv.hpp>  // start_ref::kv::engine (part-family ART, refcount liveness)
#endif
#if PSITRIMDBX_BACKEND_GCART || PSITRIMDBX_BACKEND_DELTA_GCART
#include <gcart/kv.hpp>  // gcart::kv::engine (spec engine: 3-region FIX/SAL/BIG)
#endif
#if PSITRIMDBX_BACKEND_RCART
#include <rcart/kv.hpp>  // rcart::kv::engine (refcount baseline: FIX/SAL/BIG + hazard readers)
#endif
#if PSITRIMDBX_USES_TIERED
#include <delta_art/tiered_kv.hpp>
#endif
#if PSITRIMDBX_BACKEND_DWAL
#include <psitri/dwal/byte_engine.hpp>
#endif
#if PSITRIMDBX_BACKEND_DWAL || PSITRIMDBX_BACKEND_DELTA_ART || PSITRIMDBX_BACKEND_SAL_ART_REF
#include <sal_art_ref/engine.hpp>  // sal_art_ref::radix::dwal_engine (cold base / direct)
#endif
#if PSITRIMDBX_USES_PSITRI
#include <psitri/database_impl.hpp>     // basic_database<LockPolicy> full def
#include <psitri/dwal/kv_engine.hpp>    // psitri::dwal::psitri_engine (legacy direct / cold base)
#include <psitri/read_session_impl.hpp>
#include <psitri/transaction.hpp>
#include <psitri/write_cursor.hpp>
#include <psitri/write_session_impl.hpp>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <unistd.h>

#ifdef SAL_STALL_DEBUG
// [qlog] forward decl — definition (with the write-txn filter) is below, after MDBX_txn is complete.
struct MDBX_txn;
static void qlog(const MDBX_txn* txn, char op, int subop, uint32_t dbi, const void* ki, size_t kil,
                 int rc, const void* ko, size_t kol, const void* vo, size_t vol);
#endif

// [RADIX_CRASHTRACE — TEMPORARY DIAGNOSTIC] symbolized SIGBUS/SIGSEGV backtrace (the cold-base
// Bug B memmove-from-garbage; lldb batch won't expand the stack). Arm with RADIX_CRASHTRACE=1.
#include <csignal>
#include <execinfo.h>
#ifdef PSITRI_DEBUG_BUILD
namespace
{
   void psitri_crash_handler(int sig)
   {
      const char* m = "\n=== PSITRI CRASHTRACE (signal) ===\n";
      (void)::write(2, m, std::strlen(m));
      void*     bt[64];
      const int n = ::backtrace(bt, 64);
      ::backtrace_symbols_fd(bt, n, 2);
      ::signal(sig, SIG_DFL);
      ::raise(sig);
   }
   struct psitri_crash_installer
   {
      psitri_crash_installer()
      {
         if (sal::env::get("RADIX_CRASHTRACE"))
         {
            ::signal(SIGBUS, psitri_crash_handler);
            ::signal(SIGSEGV, psitri_crash_handler);
         }
      }
   } g_psitri_crash_installer;
}  // namespace
#endif  // PSITRI_DEBUG (RADIX_CRASHTRACE)

namespace
{
   // Single-init snapshot of the shim's runtime env knobs (read ONCE at first use,
   // never re-read). These are consumed at mdbx_env_open; routing them through one
   // static-local struct keeps the open path free of scattered getenv calls.
   struct shim_env
   {
      const char* write_mode    = nullptr;  // PSITRI_MDBX_WRITE_MODE (raw; parsed by caller)
      bool        no_read_cache = false;    // PSITRI_NO_READ_CACHE present
      uint64_t    cache_mb      = 0;        // PSITRI_CACHE_MB (0 = unset)
      const char* sync          = nullptr;  // PSITRI_SYNC (raw; mapped by caller)
      uint64_t    hi_water_mb   = 0;        // PSITRI_MDBX_HI_WATER_MB (0 = unset)

      static const shim_env& get()
      {
         static const shim_env e = load();
         return e;
      }

     private:
      static shim_env load()
      {
         shim_env e;
         e.write_mode    = sal::env::get("PSITRI_MDBX_WRITE_MODE");
         e.no_read_cache = sal::env::get("PSITRI_NO_READ_CACHE") != nullptr;
         if (const char* m = sal::env::get("PSITRI_CACHE_MB"))
            e.cache_mb = std::strtoull(m, nullptr, 10);
         e.sync = sal::env::get("PSITRI_SYNC");
         if (const char* h = sal::env::get("PSITRI_MDBX_HI_WATER_MB"))
            e.hi_water_mb = std::strtoull(h, nullptr, 10);
         return e;
      }
   };
}  // namespace

// ── delta_art tiered backend aliases ─────────────────────────────────
// W = start_ver hot delta, B = sal_art_ref dwal cold base. The composed
// engine is one kv_engine: a single ordered KV space (root 0 = latest,
// W∪B merge) with bidirectional cursors. The shim funnels everything
// through this one physical space; the MDBX dbi number is encoded as a
// one-byte key namespace prefix (mapping logic, unchanged).
#if PSITRIMDBX_BACKEND_TIERED_PSITRI
using TK         = delta_art::tiered_kv<start_ver::kv::engine,
                                        psitri::dwal::psitri_engine<>>;  // start_ver hot -> psitri cold
#elif PSITRIMDBX_BACKEND_PSITRI
using TK         = psitri::dwal::psitri_engine<>;  // legacy psitri DIRECT (old algo)
#elif PSITRIMDBX_BACKEND_START_VER
using TK         = start_ver::kv::engine;  // start_ver alone — no cold base / drain
#elif PSITRIMDBX_BACKEND_START_REF
using TK         = start_ref::kv::engine;  // start_ref alone — part-family ART, refcount liveness
#elif PSITRIMDBX_BACKEND_GCART
using TK         = gcart::kv::engine;  // gcart spec engine DIRECT — 3-region FIX/SAL/BIG, refcount
#elif PSITRIMDBX_BACKEND_RCART
using TK         = rcart::kv::engine;  // rcart refcount baseline DIRECT — FIX/SAL/BIG + hazard readers
#elif PSITRIMDBX_BACKEND_DELTA_GCART
using TK         = delta_art::tiered_kv<start_ver::kv::engine,
                                        gcart::kv::engine>;  // start_ver hot delta -> gcart cold base
#elif PSITRIMDBX_BACKEND_DWAL
using TK = psitri::dwal::byte_engine<sal_art_ref::radix::engine>;
#elif PSITRIMDBX_BACKEND_SAL_ART_REF
using TK         = sal_art_ref::radix::dwal_engine;  // sal_art_ref DIRECT — no tiering / no merger
#else
using TK         = delta_art::tiered_kv<start_ver::kv::engine,
                                        sal_art_ref::radix::dwal_engine>;
#endif
using be_db      = TK::database;
using be_ws      = TK::write_session;
using be_tx      = TK::transaction;
using be_rs      = TK::read_session;
using be_cursor  = TK::cursor;
// The delta_art backend is a single logical root: writes always target
// root 0 (latest); reads use root 0 (W∪B merge). The shim's namespace
// prefix discriminates DBIs inside that one space.
static constexpr uint32_t be_root_latest = 0;

// ════════════════════════════════════════════════════════════════════
// Internal types (hidden behind opaque C handles)
// ════════════════════════════════════════════════════════════════════

static size_t system_page_size()
{
   long size = ::sysconf(_SC_PAGESIZE);
   return size > 0 ? static_cast<size_t>(size) : 4096;
}

static size_t requested_or_default_page_size(intptr_t pagesize)
{
   return pagesize > 0 ? static_cast<size_t>(pagesize) : system_page_size();
}

static size_t max_single_value_size(size_t page_size, size_t key_size)
{
   static constexpr size_t page_overhead_size = 32;
   if (page_size <= page_overhead_size + 2 * sizeof(uint16_t) + key_size)
      return 0;
   size_t page_room = page_size - page_overhead_size;
   size_t leaf_node_room = ((page_room / 2) & ~size_t{1}) - 2 * sizeof(uint16_t);
   return leaf_node_room > key_size ? leaf_node_room - key_size : 0;
}

static constexpr size_t   max_psitri_key_size = 1024;
static constexpr uint32_t mdbx_data_root_index = 1;
static constexpr size_t   mdbx_namespace_prefix_size = 1;

// Values up to the line arena's 8 KB leaf go in a value_ptr leaf; bigger ones
// (up to the large arena's 64 * 8192 = 0.5 MB object) go in a large_suffix_value
// node (K::value_large). The leaf must also hold the key suffix + a ~13-byte
// header (vstamp/slen/vlen + the data/type envelope). Values above the large cap
// are rejected with MDBX_BAD_VALSIZE (native MDBX's oversize error). 0.5 MB
// covers Ethereum (bytecode caps at 24 KB/EIP-170); a 16 KB grain would lift it
// to 1 MB if ever needed.
#if PSITRIMDBX_BACKEND_GCART || PSITRIMDBX_BACKEND_RCART
// gcart/rcart BIG region is an 8 MB-segment heap; a value rides one contiguous
// big_leaf, so the per-value cap is ~a segment (keep a margin for the inline
// header + key suffix). Comfortably covers Ethereum's 24 KB bytecode and the
// 1 MB conformance case.
static constexpr size_t   backend_value_leaf_budget = 7u * 1024 * 1024;
#else
static constexpr size_t   backend_value_leaf_budget = 64 * 8192;  // 524288 (large arena)
#endif
static constexpr size_t   max_backend_value_size =
   backend_value_leaf_budget - max_psitri_key_size - 16;  // conservative
static constexpr size_t   max_mdbx_logical_key_size =
   max_psitri_key_size - mdbx_namespace_prefix_size;
static constexpr uint32_t max_mdbx_namespace_id = 255;

static bool key_too_large(std::string_view key)
{
   return key.size() > max_mdbx_logical_key_size;
}

static bool namespace_too_large(uint32_t root_idx)
{
   return root_idx > max_mdbx_namespace_id;
}

struct namespaced_key_buffer
{
   std::array<char, max_psitri_key_size> bytes{};
   size_t                                size = 0;

   std::string_view view() const noexcept
   {
      return {bytes.data(), size};
   }
};

static std::optional<std::string_view>
namespaced_key_into(uint32_t root_idx, std::string_view key,
                    namespaced_key_buffer& out)
{
   if (namespace_too_large(root_idx) ||
       key.size() + mdbx_namespace_prefix_size > out.bytes.size())
      return std::nullopt;

   out.bytes[0] = static_cast<char>(root_idx);
   if (!key.empty())
      std::memcpy(out.bytes.data() + mdbx_namespace_prefix_size, key.data(),
                  key.size());
   out.size = key.size() + mdbx_namespace_prefix_size;
   return out.view();
}

static std::string_view namespace_prefix_into(uint32_t root_idx,
                                              namespaced_key_buffer& out)
{
   out.bytes[0] = static_cast<char>(root_idx);
   out.size = mdbx_namespace_prefix_size;
   return out.view();
}

static std::optional<std::string_view>
namespace_successor_into(uint32_t root_idx, namespaced_key_buffer& out)
{
   if (namespace_too_large(root_idx) || root_idx == max_mdbx_namespace_id)
      return std::nullopt;

   out.bytes[0] = static_cast<char>(root_idx + 1);
   out.size = mdbx_namespace_prefix_size;
   return out.view();
}

static bool key_has_namespace(std::string_view physical, uint32_t root_idx)
{
   return !namespace_too_large(root_idx) &&
          physical.size() >= mdbx_namespace_prefix_size &&
          static_cast<unsigned char>(physical[0]) ==
             static_cast<unsigned char>(root_idx);
}

static std::string_view strip_namespace(std::string_view physical)
{
   return physical.size() >= mdbx_namespace_prefix_size
             ? physical.substr(mdbx_namespace_prefix_size)
             : std::string_view{};
}

static std::optional<size_t> read_marker_page_size(const std::filesystem::path& marker)
{
   std::ifstream in{marker, std::ios::binary};
   if (!in)
      return std::nullopt;

   std::string text((std::istreambuf_iterator<char>(in)),
                    std::istreambuf_iterator<char>());
   static constexpr std::string_view tag = "pagesize=";
   auto pos = text.find(tag);
   if (pos == std::string::npos)
      return std::nullopt;

   pos += tag.size();
   size_t value = 0;
   while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9')
   {
      value = value * 10 + static_cast<size_t>(text[pos] - '0');
      ++pos;
   }
   return value != 0 ? std::optional<size_t>{value} : std::nullopt;
}

static void write_marker_page_size(const std::filesystem::path& marker,
                                   size_t page_size)
{
   std::ofstream out{marker, std::ios::binary | std::ios::trunc};
   out << "psitrimdbx\npagesize=" << page_size << '\n';
}

/// Per-DBI metadata stored in the environment.
struct dbi_info
{
   std::string name;           // empty for unnamed default DB
   uint32_t    root_index;     // logical DBI namespace byte in root 1
   unsigned    flags;          // MDBX_db_flags_t used at creation
   bool        is_dupsort;     // (flags & MDBX_DUPSORT) != 0
   bool        reverse_dup;     // (flags & MDBX_REVERSEDUP) != 0
};

static std::atomic<uint64_t> g_next_env_session_cache_id{1};

struct MDBX_env
{
   // ── delta_art tiered backend ──────────────────────────────────
   // ONE database, ONE shared write session (delta_art serializes
   // writers internally on its own writer mutex; the shim's
   // writer_mutex still enforces MDBX's one-writer-per-env contract
   // at the boundary so duplicate same-thread writers fail cleanly).
   std::shared_ptr<be_db>                         db;
   std::shared_ptr<be_ws>                         ws;
   std::filesystem::path                          path;
   uint64_t session_cache_id =
      g_next_env_session_cache_id.fetch_add(1, std::memory_order_relaxed);

   // ── Configuration (set before open) ───────────────────────────
   unsigned          max_dbs     = 16;
   unsigned          max_readers = 126;
   MDBX_env_flags_t  env_flags   = MDBX_ENV_DEFAULTS;
   bool              opened      = false;
   void*             userctx     = nullptr;
   size_t            page_size   = 4096;

   // Read mode for RO transactions: 0=buffered, 1=latest, 2=direct(get_latest)
   int               read_mode   = 1;  // default: latest (sees all committed data)
   int               write_mode  = PSITRI_WRITE_MODE_DIRECT;
   bool              write_mode_explicit = false;

   // Geometry (stored but ignored — psitri manages its own sizing)
   intptr_t geo_lower = -1, geo_now = -1, geo_upper = -1;
   intptr_t geo_growth = -1, geo_shrink = -1, geo_pagesize = -1;
   std::unordered_map<int, uint64_t> options;

   // ── DBI registry ─────────────────────────────────────────────
   mutable std::shared_mutex             dbi_mutex;
   std::vector<dbi_info>                 dbis;        // indexed by MDBX_dbi
   std::unordered_map<std::string, MDBX_dbi> name_to_dbi;
   // DBI 0 = reserved (metadata), DBI 1 = unnamed default.
   // Named DBIs start at 2. In the direct layout, the DBI number is also the
   // one-byte key namespace inside the single physical PsiTri root.

   // ── Sequence counter for txn IDs ──────────────────────────────
   std::atomic<uint64_t> next_txn_id{1};

   // MDBX permits only one write transaction per environment. Track that at
   // the shim boundary so same-thread duplicate writers fail instead of
   // self-deadlocking inside the underlying PsiTri write session.
   std::mutex              writer_mutex;
   std::condition_variable writer_cv;
   bool                    writer_active = false;
   std::thread::id         writer_owner;

   void init_default_dbi()
   {
      // DBI 0: metadata root (stores name→dbi mappings)
      dbis.push_back({"__meta__", 0, 0, false, false});
      // DBI 1: unnamed default database (root 1)
      dbis.push_back({"", 1, MDBX_DB_DEFAULTS, false, false});
   }

   MDBX_dbi allocate_dbi(const std::string& name, unsigned flags)
   {
      MDBX_dbi dbi = static_cast<MDBX_dbi>(dbis.size());
      uint32_t root_idx = dbi;
      dbis.push_back({name, root_idx, flags, (flags & MDBX_DUPSORT) != 0,
                      (flags & MDBX_REVERSEDUP) != 0});
      if (!name.empty())
         name_to_dbi[name] = dbi;

      // Under COWART, latest reads are lock-free via cow_coordinator.
      // No explicit locking setup needed.

      return dbi;
   }

   MDBX_dbi allocate_dbi_with_root(const std::string& name, unsigned flags,
                                   uint32_t root_idx)
   {
      MDBX_dbi dbi = static_cast<MDBX_dbi>(dbis.size());
      dbis.push_back({name, root_idx, flags, (flags & MDBX_DUPSORT) != 0,
                      (flags & MDBX_REVERSEDUP) != 0});
      if (!name.empty())
         name_to_dbi[name] = dbi;
      return dbi;
   }
};

// ── DUPSORT composite-key encoding ───────────────────────────────
//
// A DUPSORT DBI stores each duplicate as one physical PsiTri key:
//
//    escaped(mdbx_key) + "\0\0" + "\1" + escaped(sort_value)
//
// The shim also stores one hidden marker per outer key:
//
//    escaped(mdbx_key) + "\0\0" + "\0"
//
// That marker makes cursor::find(key) an exact seek followed by next(), while
// the value tag leaves an empty duplicate value unambiguous. Stored values are
// empty. This keeps duplicate writes in the top-level tree instead of editing a
// per-key subtree, so repeated inserts can use PsiTri's unique-root fast path.
// NUL bytes are escaped as "\0\xff"; "\0\0" is reserved as the field separator,
// which preserves bytewise sort order and prefix order.

static constexpr char dupsort_outer_tag = '\0';
static constexpr char dupsort_value_tag = '\1';

struct dupsort_key_buffer
{
   std::array<char, max_psitri_key_size> bytes{};
   size_t                                size = 0;

   std::string_view view() const noexcept
   {
      return {bytes.data(), size};
   }
};

static bool append_dupsort_escaped_into(char* out, size_t capacity, size_t& pos,
                                        std::string_view bytes,
                                        bool reverse_bytes = false)
{
   for (char ch : bytes)
   {
      unsigned char byte = static_cast<unsigned char>(ch);
      if (reverse_bytes)
         byte = static_cast<unsigned char>(~byte);

      if (byte == 0)
      {
         if (pos + 2 > capacity)
            return false;
         out[pos++] = '\0';
         out[pos++] = static_cast<char>(0xff);
      }
      else
      {
         if (pos + 1 > capacity)
            return false;
         out[pos++] = static_cast<char>(byte);
      }
   }
   return true;
}

static std::optional<std::string_view>
dupsort_prefix_into(uint32_t root_idx, std::string_view key,
                    dupsort_key_buffer& out)
{
   if (namespace_too_large(root_idx))
      return std::nullopt;

   size_t pos = 0;
   out.bytes[pos++] = static_cast<char>(root_idx);
   if (!append_dupsort_escaped_into(out.bytes.data(), out.bytes.size(), pos, key))
      return std::nullopt;

   if (pos + 2 > out.bytes.size())
      return std::nullopt;
   out.bytes[pos++] = '\0';
   out.bytes[pos++] = '\0';

   out.size = pos;
   return out.view();
}

static std::optional<std::string_view>
dupsort_tagged_key_into(uint32_t root_idx, std::string_view key, char tag,
                        dupsort_key_buffer& out)
{
   auto prefix = dupsort_prefix_into(root_idx, key, out);
   if (!prefix || out.size + 1 > out.bytes.size())
      return std::nullopt;

   out.bytes[out.size++] = tag;
   return out.view();
}

static std::optional<std::string_view>
dupsort_composite_key_into(uint32_t root_idx, std::string_view key,
                           std::string_view value, bool reverse_dup,
                           dupsort_key_buffer& out)
{
   if (!dupsort_tagged_key_into(root_idx, key, dupsort_value_tag, out))
      return std::nullopt;

   if (!append_dupsort_escaped_into(out.bytes.data(), out.bytes.size(), out.size,
                                    value, reverse_dup))
      return std::nullopt;
   return out.view();
}

static std::optional<std::string_view>
dupsort_outer_marker_key_into(uint32_t root_idx, std::string_view key,
                              dupsort_key_buffer& out)
{
   return dupsort_tagged_key_into(root_idx, key, dupsort_outer_tag, out);
}

static std::optional<std::string_view>
dupsort_first_value_key_into(uint32_t root_idx, std::string_view key,
                             dupsort_key_buffer& out)
{
   return dupsort_tagged_key_into(root_idx, key, dupsort_value_tag, out);
}

static bool dupsort_key_matches_prefix(std::string_view composite,
                                       std::string_view prefix)
{
   return composite.size() >= prefix.size()
          && composite.compare(0, prefix.size(), prefix) == 0;
}

static bool dupsort_key_is_outer_marker(std::string_view composite,
                                        std::string_view prefix)
{
   return composite.size() == prefix.size() + 1
          && dupsort_key_matches_prefix(composite, prefix)
          && composite[prefix.size()] == dupsort_outer_tag;
}

static bool dupsort_key_is_value_for_prefix(std::string_view composite,
                                            std::string_view prefix)
{
   return composite.size() >= prefix.size() + 1
          && dupsort_key_matches_prefix(composite, prefix)
          && composite[prefix.size()] == dupsort_value_tag;
}

static std::optional<std::string_view>
prefix_successor_into(std::string_view prefix, dupsort_key_buffer& out)
{
   if (prefix.size() > out.bytes.size())
      return std::nullopt;

   std::memcpy(out.bytes.data(), prefix.data(), prefix.size());
   for (size_t i = prefix.size(); i > 0; --i)
   {
      unsigned char byte = static_cast<unsigned char>(out.bytes[i - 1]);
      if (byte != 0xff)
      {
         out.bytes[i - 1] = static_cast<char>(byte + 1);
         out.size = i;
         return out.view();
      }
   }
   return std::nullopt;
}

static bool decode_dupsort_composite(std::string_view composite,
                                     uint32_t         root_idx,
                                     bool             reverse_dup,
                                     std::string&     key_out,
                                     std::string&     value_out,
                                     bool*            outer_marker_out = nullptr)
{
   key_out.clear();
   value_out.clear();
   if (outer_marker_out)
      *outer_marker_out = false;

   if (!key_has_namespace(composite, root_idx))
      return false;

   size_t pos = mdbx_namespace_prefix_size;
   bool   saw_separator = false;

   while (pos < composite.size())
   {
      unsigned char byte = static_cast<unsigned char>(composite[pos++]);
      if (byte != 0)
      {
         key_out.push_back(static_cast<char>(byte));
         continue;
      }

      if (pos >= composite.size())
         return false;

      unsigned char marker = static_cast<unsigned char>(composite[pos++]);
      if (marker == 0)
      {
         saw_separator = true;
         break;
      }
      if (marker != 0xff)
         return false;

      key_out.push_back('\0');
   }

   if (!saw_separator)
      return false;

   if (pos >= composite.size())
      return false;

   unsigned char tag = static_cast<unsigned char>(composite[pos++]);
   if (tag == static_cast<unsigned char>(dupsort_outer_tag))
   {
      if (pos != composite.size())
         return false;
      if (!outer_marker_out)
         return false;
      *outer_marker_out = true;
      return true;
   }

   if (tag != static_cast<unsigned char>(dupsort_value_tag))
      return false;

   while (pos < composite.size())
   {
      unsigned char byte = static_cast<unsigned char>(composite[pos++]);
      if (byte == 0)
      {
         if (pos >= composite.size())
            return false;

         unsigned char marker = static_cast<unsigned char>(composite[pos++]);
         if (marker != 0xff)
            return false;
         byte = 0;
      }

      if (reverse_dup)
         byte = static_cast<unsigned char>(~byte);
      value_out.push_back(static_cast<char>(byte));
   }

   return true;
}

/// Transaction-owned storage for non-cursor slices returned through the MDBX API.
///
/// Native MDBX usually returns pointers into mmap'd pages, so callers commonly
/// keep MDBX_val slices from mdbx_get() for the transaction lifetime. Cursor
/// results are borrowed from cursor_state instead and are valid until the next
/// cursor movement.
class returned_slice_arena
{
 public:
   MDBX_val hold(std::string_view bytes)
   {
      if (bytes.empty())
         return {&empty_, 0};

      char* dst = allocate(bytes.size());
      std::memcpy(dst, bytes.data(), bytes.size());
      return {dst, bytes.size()};
   }

   void clear() { chunks_.clear(); }

 private:
   struct chunk
   {
      std::unique_ptr<char[]> data;
      size_t                  capacity = 0;
      size_t                  used     = 0;
   };

   static constexpr size_t default_chunk_size = 1024 * 1024;

   char* allocate(size_t bytes)
   {
      if (chunks_.empty()
          || chunks_.back().capacity - chunks_.back().used < bytes)
      {
         const size_t capacity = std::max(default_chunk_size, bytes);
         chunks_.push_back({std::make_unique<char[]>(capacity), capacity, 0});
      }

      auto& chunk = chunks_.back();
      char* ptr = chunk.data.get() + chunk.used;
      chunk.used += bytes;
      return ptr;
   }

   std::vector<chunk> chunks_;
   char               empty_ = '\0';
};

// The per-write-txn read-own-writes overlay: physical key -> value, with
// nullopt meaning a same-txn delete (tombstone) that must MASK committed data.
using write_set_t = std::map<std::string, std::optional<std::string>>;

/// Cursor facade over the delta_art tiered cursor (the single backend).
/// Keeps the exact public surface the hundreds of call sites depend on
/// (seek/lower_bound/upper_bound/find/next/prev/key/read_value …) so the
/// engine-agnostic mapping above it is untouched.
///
/// When opened inside a WRITE transaction it is given that txn's write_set
/// and performs a bidirectional ordered merge (overlay shadows committed on
/// equal keys; an overlay tombstone hides both) — read-your-own-uncommitted-
/// writes for cursor iteration, which the delta_art transaction itself cannot
/// provide. A plain committed-only cursor (RO txns) has no overlay.
class mdbx_view_cursor
{
 public:
   explicit mdbx_view_cursor(be_cursor c) : cur_(std::move(c)) {}
   mdbx_view_cursor(be_cursor c, const write_set_t* ws)
       : cur_(std::move(c)), ws_(ws)
   {
   }

   mdbx_view_cursor(mdbx_view_cursor&& other) noexcept
       : cur_(std::move(other.cur_)), ws_(other.ws_), o_it_(other.o_it_),
         b_more_(other.b_more_), o_valid_(other.o_valid_), on_b_(other.on_b_),
         on_o_(other.on_o_), rend_(other.rend_)
   {
   }
   mdbx_view_cursor& operator=(mdbx_view_cursor&& other) noexcept
   {
      if (this != &other)
      {
         // be_cursor is move-constructible but not move-ASSIGNABLE (it holds a
         // copyable-but-not-assignable warm snapshot handle), so reset+emplace.
         cur_.reset();
         if (other.cur_)
            cur_.emplace(std::move(*other.cur_));
         ws_      = other.ws_;
         o_it_    = other.o_it_;
         b_more_  = other.b_more_;
         o_valid_ = other.o_valid_;
         on_b_    = other.on_b_;
         on_o_    = other.on_o_;
         rend_    = other.rend_;
      }
      return *this;
   }
   mdbx_view_cursor(const mdbx_view_cursor&)                = delete;
   mdbx_view_cursor& operator=(const mdbx_view_cursor&)     = delete;

   mdbx_view_cursor*       operator->() noexcept { return this; }
   const mdbx_view_cursor* operator->() const noexcept { return this; }

   bool seek_begin()
   {
      if (!has_overlay())
         return cur_->seek_begin();
      b_more_  = cur_->seek_begin();
      o_it_    = ws_->begin();
      o_valid_ = o_it_ != ws_->end();
      return settle_fwd_();
   }
   bool seek_last()
   {
      if (!has_overlay())
         return cur_->seek_last();
      b_more_ = cur_->seek_last();
      o_it_   = ws_->empty() ? ws_->end() : std::prev(ws_->end());
      o_valid_ = !ws_->empty();
      return settle_bwd_();
   }
   bool lower_bound(std::string_view key)
   {
      if (!has_overlay())
         return cur_->lower_bound(key);
      b_more_  = cur_->lower_bound(key);
      o_it_    = ws_->lower_bound(std::string(key));
      o_valid_ = o_it_ != ws_->end();
      return settle_fwd_();
   }
   // MDBX_SET_UPPERBOUND: position at the first key STRICTLY > key. The engine
   // cursor's upper_bound is now the mathematical floor (largest <= k), so the
   // strictly-greater seek on that leg is lower_bound + skip-an-exact-match. ws_
   // is a std::map, whose upper_bound is already first-strictly-greater (correct).
   bool cur_strict_gt_(std::string_view key)
   {
      bool ok = cur_->lower_bound(key);
      if (ok && cur_->key() == key)
         ok = cur_->next();
      return ok;
   }
   bool upper_bound(std::string_view key)
   {
      if (!has_overlay())
         return cur_strict_gt_(key);
      b_more_  = cur_strict_gt_(key);
      o_it_    = ws_->upper_bound(std::string(key));
      o_valid_ = o_it_ != ws_->end();
      return settle_fwd_();
   }
   bool seek(std::string_view key)
   {
      return find(key);
   }
   // Exact-match find: the tiered cursor has none; compose it from
   // lower_bound + an exact key compare (mapping glue). Works with overlay.
   bool find(std::string_view key)
   {
      if (!lower_bound(key))
         return false;
      return this->key() == key;
   }
   bool next()
   {
      if (!has_overlay())
         return cur_->next();
      // advance whichever side(s) we are sitting on, then re-settle
      if (rend_)
         return seek_begin();
      if (on_b_)
         b_more_ = cur_->next();
      if (on_o_)
         advance_overlay_fwd_();
      return settle_fwd_();
   }
   bool prev()
   {
      if (!has_overlay())
         return cur_->prev();
      if (is_end())
         return seek_last();
      if (rend_)
         return false;
      if (on_b_)
         b_more_ = cur_->prev();
      if (on_o_)
         retreat_overlay_bwd_();
      return settle_bwd_();
   }

   bool is_end() const
   {
      if (!has_overlay())
         return cur_->is_end();
      return !rend_ && !on_b_ && !on_o_;
   }
   bool is_rend() const
   {
      if (!has_overlay())
         return cur_->is_rend();
      return rend_;
   }
   std::string_view key() const
   {
      if (!has_overlay())
         return cur_->key();
      return on_o_ ? std::string_view(o_it_->first) : cur_->key();
   }

   void clear_borrowed_value() {}
   bool has_cursor() const noexcept { return cur_.has_value(); }
   void reset() noexcept
   {
      cur_.reset();
      ws_ = nullptr;
   }

   // Copy the current value into `out` and point `view` at it. RAW bytes
   // (the tiered cursor's value() unwraps the data/tombstone envelope;
   // overlay values are already raw). `view` is buffered (valid until the
   // next cursor op); returns false (never a borrowed/pinned slice).
   bool read_value(std::string& out, std::string_view& view)
   {
      if (has_overlay() && on_o_)
      {
         const auto& v = o_it_->second;  // settle guarantees this is data
         out.assign(v ? *v : std::string());
         view = out;
         return false;
      }
      auto v = cur_->template value<std::string>();
      if (!v)
      {
         view = {};
         return false;
      }
      out = std::move(*v);
      view = out;
      return false;  // buffered (not a borrowed/pinned slice)
   }

 private:
   bool has_overlay() const noexcept { return ws_ != nullptr; }

   // forward overlay state: o_it_ points at the candidate; o_valid_ tracks
   // validity for the backward direction (std::map has no rend sentinel).
   void advance_overlay_fwd_()
   {
      if (o_it_ != ws_->end())
         ++o_it_;
      o_valid_ = o_it_ != ws_->end();
   }
   void retreat_overlay_bwd_()
   {
      if (o_it_ == ws_->begin())
      {
         o_valid_ = false;
         o_it_    = ws_->end();
      }
      else
         --o_it_;
   }

   bool overlay_live_fwd_() const { return o_it_ != ws_->end(); }
   bool overlay_live_bwd_() const { return o_valid_; }

   // 2-way forward merge; overlay shadows committed; an overlay tombstone
   // (nullopt value) masks both sides at that key.
   bool settle_fwd_()
   {
      for (;;)
      {
         const bool be = !b_more_ || cur_->is_end();
         const bool oe = !overlay_live_fwd_();
         if (be && oe)
         {
            on_b_ = on_o_ = false;
            rend_ = false;
            return false;
         }
         int c = 0;  // <0: base smaller, >0: overlay smaller, 0: equal
         if (be)
            c = 1;
         else if (oe)
            c = -1;
         else
            c = cmp_(cur_->key(), o_it_->first);
         on_b_ = (c <= 0) && !be;
         on_o_ = (c >= 0) && !oe;
         rend_ = false;
         if (on_o_ && !o_it_->second)  // overlay tombstone: skip both
         {
            if (on_b_)
               b_more_ = cur_->next();
            advance_overlay_fwd_();
            continue;
         }
         return true;
      }
   }
   bool settle_bwd_()
   {
      for (;;)
      {
         const bool be = !b_more_ || cur_->is_end() || cur_->is_rend();
         const bool oe = !overlay_live_bwd_();
         if (be && oe)
         {
            on_b_ = on_o_ = false;
            rend_ = true;
            return false;
         }
         int c = 0;
         if (be)
            c = -1;  // overlay wins (it is the only live side)
         else if (oe)
            c = 1;
         else
            c = cmp_(cur_->key(), o_it_->first);
         // backward: the LARGER key wins; overlay shadows base on ties.
         on_b_ = (c >= 0) && !be;
         on_o_ = (c <= 0) && !oe;
         rend_ = false;
         if (on_o_ && !o_it_->second)
         {
            if (on_b_)
               b_more_ = cur_->prev();
            retreat_overlay_bwd_();
            continue;
         }
         return true;
      }
   }
   static int cmp_(std::string_view a, std::string_view b)
   {
      const int c = std::memcmp(a.data(), b.data(), std::min(a.size(), b.size()));
      if (c)
         return c < 0 ? -1 : 1;
      return a.size() < b.size() ? -1 : a.size() > b.size() ? 1 : 0;
   }

   std::optional<be_cursor>      cur_;
   const write_set_t*            ws_ = nullptr;     // write-txn overlay (or null)
   write_set_t::const_iterator   o_it_;             // overlay candidate
   bool                          b_more_  = false;  // base cursor positioned
   bool                          o_valid_ = false;  // overlay live (backward dir)
   bool                          on_b_    = false;  // current key is on base
   bool                          on_o_    = false;  // current key is on overlay
   bool                          rend_    = false;  // settled before-first
};

/// Cursor state for a single DBI.
struct cursor_state
{
   mdbx_view_cursor                      mc;
   uint32_t                              root_idx    = UINT32_MAX;
   bool                                  valid       = false;
   bool                                  touched     = false;
   bool                                  stale_after_write = false;
   bool                                  dupsort     = false;
   bool                                  reverse_dup = false;
   std::string                           key_buf;  // exposed key
   std::string                           val_buf;  // exposed value
   std::string_view                      val_view; // borrowed or buffered value
   bool                                  val_borrowed = false;
   std::string                           raw_key;  // physical PsiTri key

   explicit cursor_state(mdbx_view_cursor m, uint32_t ri,
                         bool ds = false, bool rd = false)
       : mc(std::move(m)),
         root_idx(ri),
         dupsort(ds),
         reverse_dup(rd)
   {
   }

   /// Read the raw value from the merge cursor into val_buf (non-DUPSORT).
   void read_value()
   {
      val_borrowed = mc.read_value(val_buf, val_view);
   }

   void expose_buffered_value()
   {
      mc.clear_borrowed_value();
      val_view     = val_buf;
      val_borrowed = false;
   }

   void drop_borrowed_value()
   {
      if (val_borrowed)
      {
         mc.clear_borrowed_value();
         val_view     = {};
         val_borrowed = false;
      }
   }

   void mark_stale_after_write()
   {
      drop_borrowed_value();
      mc.reset();
      stale_after_write = true;
   }

   std::string_view exposed_value() const
   {
      return val_view;
   }

   bool outer_positioned() const
   {
      return !mc->is_end() && !mc->is_rend();
   }

   bool sync_dup_key_val()
   {
      if (!outer_positioned())
      {
         valid = false;
         return false;
      }

      raw_key.assign(mc->key().data(), mc->key().size());
      bool is_outer_marker = false;
      if (!decode_dupsort_composite(raw_key, root_idx, reverse_dup, key_buf, val_buf,
                                    &is_outer_marker)
         || is_outer_marker)
      {
         valid = false;
         val_view = {};
         val_borrowed = false;
         return false;
      }

      expose_buffered_value();
      valid = true;
      touched = true;
      return true;
   }

   bool sync_forward_dup()
   {
      while (outer_positioned())
      {
         bool is_outer_marker = false;
         raw_key.assign(mc->key().data(), mc->key().size());
         if (decode_dupsort_composite(raw_key, root_idx, reverse_dup, key_buf, val_buf,
                                      &is_outer_marker)
            && !is_outer_marker)
         {
            expose_buffered_value();
            valid = true;
            touched = true;
            return true;
         }
         if (!is_outer_marker || !mc->next())
            break;
      }

      valid = false;
      return false;
   }

   bool sync_backward_dup()
   {
      while (outer_positioned())
      {
         bool is_outer_marker = false;
         raw_key.assign(mc->key().data(), mc->key().size());
         if (decode_dupsort_composite(raw_key, root_idx, reverse_dup, key_buf, val_buf,
                                      &is_outer_marker)
            && !is_outer_marker)
         {
            expose_buffered_value();
            valid = true;
            touched = true;
            return true;
         }
         if (!is_outer_marker || !mc->prev())
            break;
      }

      valid = false;
      return false;
   }

   bool sync_forward_dup_for_prefix(std::string_view prefix)
   {
      while (outer_positioned() && dupsort_key_matches_prefix(mc->key(), prefix))
      {
         if (dupsort_key_is_value_for_prefix(mc->key(), prefix))
            return sync_dup_key_val();
         if (!dupsort_key_is_outer_marker(mc->key(), prefix) || !mc->next())
            break;
      }

      valid = false;
      return false;
   }

   bool sync_backward_dup_for_prefix(std::string_view prefix)
   {
      while (outer_positioned() && dupsort_key_matches_prefix(mc->key(), prefix))
      {
         if (dupsort_key_is_value_for_prefix(mc->key(), prefix))
            return sync_dup_key_val();
         if (!dupsort_key_is_outer_marker(mc->key(), prefix) || !mc->prev())
            break;
      }

      valid = false;
      return false;
   }

   void sync_key_val()
   {
      if (mc->is_end() || mc->is_rend())
      {
         valid = false;
         val_view = {};
         val_borrowed = false;
         return;
      }

      valid = true;
      touched = true;
      if (!dupsort)
      {
         auto physical = mc->key();
         if (!key_has_namespace(physical, root_idx))
         {
            valid = false;
            val_view = {};
            val_borrowed = false;
            return;
         }
         auto logical = strip_namespace(physical);
         key_buf.assign(logical.data(), logical.size());
         read_value();
      }
      else
      {
         sync_dup_key_val();
      }
   }

   bool positioned_in_namespace() const
   {
      return outer_positioned() && key_has_namespace(mc->key(), root_idx);
   }

   bool seek_first_kv()
   {
      drop_borrowed_value();
      namespaced_key_buffer prefix_buf;
      auto prefix = namespace_prefix_into(root_idx, prefix_buf);
      return mc->lower_bound(prefix) && positioned_in_namespace();
   }

   bool seek_last_kv()
   {
      drop_borrowed_value();
      namespaced_key_buffer upper_buf;
      auto upper = namespace_successor_into(root_idx, upper_buf);
      bool positioned = false;
      if (upper)
      {
         positioned = mc->lower_bound(*upper);
         positioned = positioned ? mc->prev() : mc->seek_last();
      }
      else
      {
         positioned = mc->seek_last();
      }
      return positioned && positioned_in_namespace();
   }

   bool seek_kv_exact(std::string_view key)
   {
      drop_borrowed_value();
      namespaced_key_buffer key_buf;
      auto physical = namespaced_key_into(root_idx, key, key_buf);
      return physical && mc->find(*physical);
   }

   bool lower_bound_kv(std::string_view key)
   {
      drop_borrowed_value();
      namespaced_key_buffer key_buf;
      auto physical = namespaced_key_into(root_idx, key, key_buf);
      return physical && mc->lower_bound(*physical) && positioned_in_namespace();
   }

   bool upper_bound_kv(std::string_view key)
   {
      drop_borrowed_value();
      namespaced_key_buffer key_buf;
      auto physical = namespaced_key_into(root_idx, key, key_buf);
      return physical && mc->upper_bound(*physical) && positioned_in_namespace();
   }

   bool next_kv()
   {
      drop_borrowed_value();
      return mc->next() && positioned_in_namespace();
   }

   bool prev_kv()
   {
      drop_borrowed_value();
      return mc->prev() && positioned_in_namespace();
   }

   bool open_first_dup_current()
   {
      if (!valid && !sync_dup_key_val())
         return false;

      dupsort_key_buffer prefix_buf;
      auto prefix = dupsort_prefix_into(root_idx, key_buf, prefix_buf);
      if (!prefix || !mc->lower_bound(*prefix))
      {
         valid = false;
         return false;
      }
      return sync_forward_dup_for_prefix(*prefix);
   }

   bool open_last_dup_current()
   {
      if (!valid && !sync_dup_key_val())
         return false;

      dupsort_key_buffer prefix_buf;
      dupsort_key_buffer upper_buf;
      auto prefix = dupsort_prefix_into(root_idx, key_buf, prefix_buf);
      auto upper  = prefix ? prefix_successor_into(*prefix, upper_buf) : std::nullopt;
      if (!prefix || !upper)
      {
         valid = false;
         return false;
      }

      bool positioned = mc->lower_bound(*upper);
      if (positioned)
         positioned = mc->prev();
      else
         positioned = mc->seek_last();

      if (!positioned)
      {
         valid = false;
         return false;
      }

      return sync_backward_dup_for_prefix(*prefix);
   }

   bool seek_first_dup()
   {
      namespaced_key_buffer prefix_buf;
      auto prefix = namespace_prefix_into(root_idx, prefix_buf);
      if (!mc->lower_bound(prefix))
      {
         valid = false;
         return false;
      }
      return sync_forward_dup();
   }

   bool seek_last_dup()
   {
      namespaced_key_buffer upper_buf;
      auto upper = namespace_successor_into(root_idx, upper_buf);
      bool positioned = false;
      if (upper)
      {
         positioned = mc->lower_bound(*upper);
         positioned = positioned ? mc->prev() : mc->seek_last();
      }
      else
      {
         positioned = mc->seek_last();
      }

      if (!positioned)
      {
         valid = false;
         return false;
      }
      return sync_backward_dup();
   }

   bool seek_outer_first_dup(std::string_view key)
   {
      dupsort_key_buffer marker_buf;
      auto prefix = dupsort_prefix_into(root_idx, key, marker_buf);
      if (!prefix || marker_buf.size + 1 > marker_buf.bytes.size())
      {
         valid = false;
         return false;
      }

      marker_buf.bytes[marker_buf.size++] = dupsort_outer_tag;
      std::string_view marker(marker_buf.bytes.data(), prefix->size() + 1);
	      if (mc->find(marker))
	      {
	         if (!mc->next())
	         {
            valid = false;
            return false;
         }
         return sync_forward_dup_for_prefix(*prefix);
      }

      marker_buf.bytes[prefix->size()] = dupsort_value_tag;
      std::string_view first_value(marker_buf.bytes.data(), prefix->size() + 1);
      if (!mc->lower_bound(first_value))
      {
         valid = false;
         return false;
      }
      return sync_forward_dup_for_prefix(*prefix);
   }

   bool lower_outer_first_dup(std::string_view key)
   {
      dupsort_key_buffer prefix_buf;
      auto prefix = dupsort_prefix_into(root_idx, key, prefix_buf);
      if (!prefix || !mc->lower_bound(*prefix))
      {
         valid = false;
         return false;
      }
      return sync_forward_dup();
   }

   bool upper_outer_first_dup(std::string_view key)
   {
      dupsort_key_buffer prefix_buf;
      dupsort_key_buffer upper_buf;
      auto prefix = dupsort_prefix_into(root_idx, key, prefix_buf);
      auto upper  = prefix ? prefix_successor_into(*prefix, upper_buf) : std::nullopt;
      if (!prefix || !upper || !mc->lower_bound(*upper))
      {
         valid = false;
         return false;
      }
      return sync_forward_dup();
   }

   bool next_outer_first_dup()
   {
      if (!valid)
         return seek_first_dup();

      dupsort_key_buffer prefix_buf;
      dupsort_key_buffer upper_buf;
      auto prefix = dupsort_prefix_into(root_idx, key_buf, prefix_buf);
      auto upper  = prefix ? prefix_successor_into(*prefix, upper_buf) : std::nullopt;
      if (!prefix || !upper || !mc->lower_bound(*upper))
      {
         valid = false;
         return false;
      }
      return sync_forward_dup();
   }

   bool prev_outer_last_dup()
   {
      if (!valid)
         return seek_last_dup();

      auto saved = raw_key;
      dupsort_key_buffer prefix_buf;
      auto prefix = dupsort_prefix_into(root_idx, key_buf, prefix_buf);
      if (!prefix || !mc->lower_bound(*prefix) || !mc->prev())
      {
	         mc->find(saved);
	         sync_dup_key_val();
	         return false;
      }

      if (!sync_backward_dup())
      {
	         mc->find(saved);
	         sync_dup_key_val();
	         return false;
      }
      return true;
   }

   bool seek_dup_exact(std::string_view key, std::string_view value)
   {
      dupsort_key_buffer composite_buf;
      auto composite = dupsort_composite_key_into(root_idx, key, value,
                                                  reverse_dup, composite_buf);
	      if (!composite || !mc->find(*composite))
      {
         valid = false;
         return false;
      }
      return sync_dup_key_val();
   }

   bool seek_dup_lower(std::string_view key, std::string_view value)
   {
      dupsort_key_buffer prefix_buf;
      dupsort_key_buffer composite_buf;
      auto prefix    = dupsort_prefix_into(root_idx, key, prefix_buf);
      auto composite = dupsort_composite_key_into(root_idx, key, value,
                                                  reverse_dup, composite_buf);
      if (!prefix || !composite || !mc->lower_bound(*composite)
          || !dupsort_key_is_value_for_prefix(mc->key(), *prefix))
      {
         valid = false;
         return false;
      }
      return sync_dup_key_val();
   }

   bool next_dup_same()
   {
      if (!valid)
         return false;

      auto saved = raw_key;
      dupsort_key_buffer prefix_buf;
      auto prefix = dupsort_prefix_into(root_idx, key_buf, prefix_buf);
      if (!prefix)
         return false;

      if (mc->next() && dupsort_key_is_value_for_prefix(mc->key(), *prefix))
         return sync_dup_key_val();

	      mc->find(saved);
	      sync_dup_key_val();
	      return false;
   }

   bool prev_dup_same()
   {
      if (!valid)
         return false;

      auto saved = raw_key;
      dupsort_key_buffer prefix_buf;
      auto prefix = dupsort_prefix_into(root_idx, key_buf, prefix_buf);
      if (!prefix)
         return false;

      if (mc->prev() && dupsort_key_is_value_for_prefix(mc->key(), *prefix))
         return sync_dup_key_val();

	      mc->find(saved);
	      sync_dup_key_val();
	      return false;
   }

   MDBX_val key_val() const
   {
      return {const_cast<char*>(key_buf.data()), key_buf.size()};
   }

   MDBX_val data_val() const
   {
      return {const_cast<char*>(val_buf.data()), val_buf.size()};
   }
};

struct dbi_meta
{
   uint32_t root_index = UINT32_MAX;
   bool     is_dupsort = false;
   bool     reverse_dup = false;
};

struct MDBX_txn
{
   MDBX_env*         env       = nullptr;
   MDBX_txn_flags_t  txn_flags = MDBX_TXN_READWRITE;
   void*             context   = nullptr;
   uint64_t          id        = 0;

   // ── RW path: one delta_art transaction over the single root ────
   // `direct_tx` is the live write transaction (latest root 0). POINT
   // reads inside a write txn go through it (read-own-writes: the W tree
   // get() sees pending upserts/removes).
   std::optional<be_tx>                   direct_tx;

   // Cursor read-own-writes overlay: the delta_art transaction has point
   // get() but no iterating cursor over UNCOMMITTED writes. Cursors opened
   // inside a write txn therefore merge this in-memory write-set (physical
   // key -> value, nullopt = tombstone/deleted) over the committed read
   // cursor. Engine-agnostic mapping glue; small (test txns are tiny).
   std::map<std::string, std::optional<std::string>> write_set;

   struct dbi_registry_change
   {
      MDBX_dbi dbi      = 0;
      dbi_info old_info = {};
      bool     had_old  = false;
   };
   std::vector<dbi_registry_change> dbi_changes;

   // Cursor storage is owned by the transaction. mdbx_cursor_close() resets
   // and returns cursors to this intrusive free list instead of deleting them.
   std::vector<std::unique_ptr<MDBX_cursor>> cursor_storage;
   MDBX_cursor*                             free_cursors = nullptr;
   size_t                                   active_cursors = 0;

   // ── RO path: a delta_art read session + a latest (W∪B) cursor ──
   // captured at txn begin/renew. A read txn is a stable view; the read
   // session pins the snapshot, the cursor reads through the merge.
   std::shared_ptr<be_rs>                 read_session;
   std::optional<be_cursor>               read_cursor;
   bool                                   has_read_view = false;

   // Iteration read session for cursors opened inside a WRITE txn (the
   // dupsort/cursor-op mapping needs an iterable view). The session must
   // outlive every cursor it hands out, so it is owned by the txn. Lazily
   // created on first iterating cursor.
   std::shared_ptr<be_rs>                 iter_read_session;

   std::shared_ptr<be_rs> ensure_iter_read_session()
   {
      if (!iter_read_session)
         iter_read_session = env->db->start_read_session();
      return iter_read_session;
   }

   // Per-transaction DBI metadata cache. Named DBI metadata is environment
   // scoped and changes only through DBI open/drop, so hot get/put/del calls
   // should not take the env DBI mutex once a txn has seen a DBI.
   mutable std::unordered_map<MDBX_dbi, dbi_meta> dbi_meta_cache;

   // Returned MDBX_val key/value bytes. Native MDBX exposes mmap-backed slices
   // that remain usable for the transaction lifetime; this arena preserves that
   // contract for materialized PsiTri reads and cursor results.
   mutable returned_slice_arena returned_slices;

   bool committed = false;
   bool aborted   = false;
   bool write_slot_reserved = false;

   bool is_readonly() const { return (txn_flags & MDBX_TXN_RDONLY) != 0; }
};

struct MDBX_cursor
{
   MDBX_txn* txn  = nullptr;
   MDBX_dbi  dbi  = 0;
   uint32_t  root_idx = UINT32_MAX;
   bool      is_dupsort = false;
   bool      is_reverse_dup = false;
   bool      open = false;
   bool      heap_owned = false;
   bool      counted = false;
   void*     context = nullptr;
   MDBX_cursor* next_free = nullptr;

   std::optional<cursor_state> state;
};

// ════════════════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════════════════

static std::optional<int> parse_psitri_write_mode(const char* value)
{
   if (!value || value[0] == '\0')
      return std::nullopt;
   if (std::strcmp(value, "direct") == 0 || std::strcmp(value, "0") == 0)
      return PSITRI_WRITE_MODE_DIRECT;
   if (std::strcmp(value, "dwal") == 0 || std::strcmp(value, "1") == 0)
      return PSITRI_WRITE_MODE_DWAL;
   return std::nullopt;
}

static std::string_view to_sv(const MDBX_val* v)
{
   return v ? std::string_view(static_cast<const char*>(v->iov_base), v->iov_len)
            : std::string_view{};
}

static MDBX_val hold_txn_slice(const MDBX_txn* txn, std::string_view bytes)
{
   return const_cast<MDBX_txn*>(txn)->returned_slices.hold(bytes);
}

static void assign_cursor_result(MDBX_cursor* cursor, MDBX_val* key,
                                 MDBX_val* data)
{
   // MDBX contract: key/value slices returned by a cursor op stay valid until
   // the txn ends (real MDBX hands back mmap-stable pointers; the old shim used
   // value_pin). The cursor's own key_buf/val_buf are REUSED on the next op, so
   // borrowing them directly dangles for any LAZY reader that dereferences after
   // advancing — e.g. Silkworm's range pipeline (... | take_while | caching),
   // which caches a non-owning ByteView and reads it after the cursor moves,
   // yielding an empty/garbage range. Copy into the txn-lifetime arena (same as
   // mdbx_get) so the slices survive cursor movement for the whole txn.
   auto& st = *cursor->state;
   if (key)
      *key = hold_txn_slice(cursor->txn, st.key_buf);
   if (data)
      *data = hold_txn_slice(cursor->txn, st.exposed_value());
#ifdef SAL_STALL_DEBUG
   qlog(cursor->txn, 'Q', 0, 0, nullptr, 0, 0, key ? key->iov_base : nullptr,
        key ? key->iov_len : 0, data ? data->iov_base : nullptr, data ? data->iov_len : 0);
#endif
}

static dbi_meta dbi_metadata(MDBX_env* env, MDBX_dbi dbi)
{
   std::shared_lock lk(env->dbi_mutex);
   if (dbi >= env->dbis.size())
      return {};

   const auto& info = env->dbis[dbi];
   return {info.root_index, info.is_dupsort, info.reverse_dup};
}

static dbi_meta txn_dbi_metadata(const MDBX_txn* txn, MDBX_dbi dbi)
{
   if (!txn)
      return {};

   auto it = txn->dbi_meta_cache.find(dbi);
   if (it != txn->dbi_meta_cache.end())
      return it->second;

   auto meta = dbi_metadata(txn->env, dbi);
   if (meta.root_index != UINT32_MAX)
      txn->dbi_meta_cache.emplace(dbi, meta);
   return meta;
}

static void cache_txn_dbi_metadata(MDBX_txn* txn, MDBX_dbi dbi,
                                   const dbi_meta& meta)
{
   if (!txn)
      return;

   if (meta.root_index == UINT32_MAX)
      txn->dbi_meta_cache.erase(dbi);
   else
      txn->dbi_meta_cache[dbi] = meta;
}

static std::string encode_catalog_entry(unsigned flags, uint32_t root_idx)
{
   static constexpr char magic[] = {'P', 'T', 'X', '1'};
   std::string out(sizeof(magic) + sizeof(uint32_t) * 2, '\0');
   std::memcpy(out.data(), magic, sizeof(magic));
   auto f = static_cast<uint32_t>(flags);
   std::memcpy(out.data() + sizeof(magic), &f, sizeof(uint32_t));
   std::memcpy(out.data() + sizeof(magic) + sizeof(uint32_t), &root_idx,
               sizeof(uint32_t));
   return out;
}

static bool decode_catalog_entry(std::string_view value, unsigned& flags,
                                 uint32_t& root_idx, bool allow_legacy)
{
   static constexpr char magic[] = {'P', 'T', 'X', '1'};

   if (value.size() >= sizeof(magic) + sizeof(uint32_t) * 2 &&
       std::memcmp(value.data(), magic, sizeof(magic)) == 0)
   {
      uint32_t f = 0;
      std::memcpy(&f, value.data() + sizeof(magic), sizeof(uint32_t));
      std::memcpy(&root_idx, value.data() + sizeof(magic) + sizeof(uint32_t),
                  sizeof(uint32_t));
      flags = f;
      return !namespace_too_large(root_idx);
   }

   if (allow_legacy && value.size() == sizeof(uint32_t) * 2)
   {
      uint32_t f = 0;
      std::memcpy(&f, value.data(), sizeof(uint32_t));
      std::memcpy(&root_idx, value.data() + sizeof(uint32_t),
                  sizeof(uint32_t));
      flags = f;
      return !namespace_too_large(root_idx);
   }

   if (allow_legacy && value.size() == sizeof(uint32_t))
   {
      uint32_t f = 0;
      std::memcpy(&f, value.data(), sizeof(uint32_t));
      flags = f;
      root_idx = 0;
      return true;
   }

   return false;
}

static constexpr unsigned max_named_dbs()
{
   // DBI 0 is metadata, DBI 1 is the unnamed default DB, and named DBIs use a
   // one-byte namespace in the single physical PsiTri root.
   return max_mdbx_namespace_id - 1;
}

// Capture the RO snapshot for a read transaction: a delta_art read session
// (pins the version) plus a latest (W∪B) cursor reused for point reads.
static void capture_direct_read_roots(MDBX_txn* txn)
{
   txn->read_session = txn->env->db->start_read_session();
   txn->read_cursor.emplace(txn->read_session->create_cursor(be_root_latest));
   txn->has_read_view = true;
}

#ifdef SAL_STALL_DEBUG
static void stall_hb(const char* phase, std::string_view key);  // [stall-debug] defined below
#endif

static void start_direct_write_txn(MDBX_txn* txn)
{
   // The env owns the single shared write session (delta_art serializes
   // writers internally on its own mutex; start_transaction blocks until
   // it owns the writer lock, released at commit/abort).
#ifdef SAL_STALL_DEBUG
   stall_hb("TXbegin0", {});  // about to acquire writer lock + start txn (blocks here if slot held)
#endif
   txn->direct_tx.emplace(txn->env->ws->start_transaction(be_root_latest));
#ifdef SAL_STALL_DEBUG
   stall_hb("TXbegin1", {});  // writer lock acquired, txn started
#endif
}

static mdbx_view_cursor make_txn_cursor(const MDBX_txn* txn, uint32_t root_idx)
{
   (void)root_idx;
   auto* mtxn = const_cast<MDBX_txn*>(txn);
   if (!txn->is_readonly())
   {
      // A cursor inside a write txn iterates the latest committed (W∪B) view
      // MERGED with this txn's read-own-writes overlay (write_set). The
      // owning read session lives in the txn (it must outlive the cursor).
#if PSITRIMDBX_BACKEND_DWAL
      // The writer already owns DWAL's root lock. Capture its COW view through
      // the transaction, avoiding a recursive reader admission lock.
      return mdbx_view_cursor(mtxn->direct_tx->create_cursor(), &mtxn->write_set);
#else
      auto rs = mtxn->ensure_iter_read_session();
      return mdbx_view_cursor(rs->create_cursor(be_root_latest), &mtxn->write_set);
#endif
   }

   if (txn->has_read_view)
      return mdbx_view_cursor(
         mtxn->read_session->create_cursor(be_root_latest));

   // No captured view (e.g. reset txn): pin one in the txn so it outlives
   // the cursor, then position from it.
   auto rs = mtxn->ensure_iter_read_session();
   return mdbx_view_cursor(rs->create_cursor(be_root_latest));
}

static bool txn_get_physical_value(const MDBX_txn* txn, std::string_view key,
                                   std::string& out)
{
   if (!txn->is_readonly())
   {
      if (txn->direct_tx)
      {
         // Read-own-writes: the transaction's 3-way (hot->warm->base) get
         // sees this txn's pending upserts/removes plus committed data.
#ifdef SAL_STALL_DEBUG
         stall_hb("G-enter", key);
#endif
         bool r = txn->direct_tx->get(key, [&out](std::string_view vv) {
            out.assign(vv.data(), vv.size());
         });
#ifdef SAL_STALL_DEBUG
         stall_hb("G-exit", key);
         qlog(txn, 'G', 0, 0, key.data(), key.size(), r ? 1 : 0, nullptr, 0,
              r ? out.data() : nullptr, r ? out.size() : 0);
#endif
         return r;
      }
      return false;
   }

   if (txn->has_read_view && txn->read_cursor)
   {
      auto* mtxn = const_cast<MDBX_txn*>(txn);
      return mtxn->read_cursor->get(key, [&out](std::string_view vv) {
         out.assign(vv.data(), vv.size());
      });
   }

   auto rs  = txn->env->db->start_read_session();
   auto cur = rs->create_cursor(be_root_latest);
   return cur.get(key, [&out](std::string_view vv) {
      out.assign(vv.data(), vv.size());
   });
}

static bool txn_get_value(const MDBX_txn* txn, uint32_t root_idx,
                          std::string_view key, std::string& out)
{
   namespaced_key_buffer key_buf;
   auto physical = namespaced_key_into(root_idx, key, key_buf);
   return physical && txn_get_physical_value(txn, *physical, out);
}

#ifdef PSITRI_DEBUG_BUILD
// [oplog] Backend-agnostic capture of the engine-facing op stream (env MDBX_OPLOG=path).
// Records U(key,val)/R(key)/C exactly as the shim drives TK::transaction — independent of which
// backend TK is compiled in — so a log captured on a KNOWN-WORKING engine (e.g. start_ver) can be
// replayed standalone (no Silkworm) into ANY engine to reproduce/diff its behavior deterministically.
// Keys here are the physical (namespaced) keys the engine receives, matching /tmp/sar_replay's format.
static void oplog_record(char op, std::string_view key, std::string_view val)
{
   static std::mutex mtx;
   static std::FILE* f    = nullptr;
   static bool       init = false;
   std::lock_guard<std::mutex> lk(mtx);
   if (!init)
   {
      init = true;
      if (const char* p = sal::env::get("MDBX_OPLOG"))
      {
         f = std::fopen(p, "wb");
         if (f)
            std::setvbuf(f, nullptr, _IOFBF, 1 << 20);  // 1 MB buffer; flushed on every commit
      }
   }
   if (!f)
      return;
   std::fputc(op, f);
   if (op == 'C')
   {
      std::fflush(f);  // commit boundary: durably capture the batch so a hang keeps the tail
      return;
   }
   uint32_t kl = uint32_t(key.size());
   std::fwrite(&kl, 4, 1, f);
   if (kl)
      std::fwrite(key.data(), 1, kl, f);
   if (op == 'U')
   {
      uint32_t vl = uint32_t(val.size());
      std::fwrite(&vl, 4, 1, f);
      if (vl)
         std::fwrite(val.data(), 1, vl, f);
   }
}
#else
static inline void oplog_record(char, std::string_view, std::string_view) {}
#endif  // PSITRI_DEBUG (MDBX_OPLOG)

#ifdef SAL_STALL_DEBUG
// [stall-debug] Overwrites /tmp/sal_hb.txt with the current engine-call phase on every entry/exit,
// flushed. At a hang, `cat /tmp/sal_hb.txt` shows the LAST phase: a "*-enter" with no matching
// "*-exit" ⇒ the writer is hung INSIDE that engine call (engine bug); an "*-exit" ⇒ it returned and
// the hang is BETWEEN calls (Silkworm/coroutine side). seq pinpoints the op; ns/klen identify it.
static std::atomic<uint64_t> g_hb_seq{0};
static void stall_hb(const char* phase, std::string_view key)
{
   static std::FILE* hb = std::fopen("/tmp/sal_hb.txt", "w");
   if (!hb)
      return;
   std::fseek(hb, 0, SEEK_SET);
   std::fprintf(hb, "seq=%llu phase=%-8s klen=%zu ns=%u v+w_done                 \n",
                static_cast<unsigned long long>(g_hb_seq.load()), phase, key.size(),
                key.empty() ? 999u : unsigned(static_cast<uint8_t>(key[0])));
   std::fflush(hb);
}

// [qlog] Differential query/response log (env MDBX_QLOG=path). Records EVERY engine op handed to
// Silkworm — writes (U/R/C) AND reads (G point-get, Q cursor) WITH their responses — as compact
// diffable text. Captured at the public boundary, BACKEND-AGNOSTIC, so the SAME workload run on the
// working backend (psitri) and the hanging one (sal_art_ref) produces two logs whose FIRST DIFF is
// the exact op where the backends hand Silkworm different data. Filtered to the WRITE transaction
// (txn->direct_tx) = the single execution/staged-sync thread → deterministic order → clean diff.
// Values/keys are FNV-1a hashed + length, so a divergent response shows as a changed hash/len.
static uint64_t q_fnv(const void* p, size_t n)
{
   const auto* d = static_cast<const unsigned char*>(p);
   uint64_t    h = 1469598103934665603ull;
   for (size_t i = 0; i < n; ++i) { h ^= d[i]; h *= 1099511628211ull; }
   return h;
}
static void qlog(const MDBX_txn* txn, char op, int subop, uint32_t dbi, const void* ki, size_t kil,
                 int rc, const void* ko, size_t kol, const void* vo, size_t vol)
{
   if (!txn || !txn->direct_tx)  // only the execution write-txn stream (deterministic, single-thread)
      return;
   static std::mutex  mx;
   static std::FILE*  f    = nullptr;
   static bool        init = false;
   std::lock_guard<std::mutex> g(mx);
   if (!init)
   {
      init = true;
      if (const char* p = sal::env::get("MDBX_QLOG"))
      {
         f = std::fopen(p, "w");
         if (f)
            std::setvbuf(f, nullptr, _IOFBF, 1 << 20);
      }
   }
   if (!f)
      return;
   std::fprintf(f, "%c%d d%u ki%016llx/%zu rc%d ko%016llx/%zu v%016llx/%zu\n", op, subop, dbi,
                static_cast<unsigned long long>(q_fnv(ki, kil)), kil, rc,
                static_cast<unsigned long long>(q_fnv(ko, kol)), kol,
                static_cast<unsigned long long>(q_fnv(vo, vol)), vol);
}
#endif

static void txn_upsert_physical_value(MDBX_txn* txn, std::string_view key,
                                      std::string_view value,
                                      bool sorted_append_hint = false)
{
   (void)sorted_append_hint;  // delta_art has no sorted-append fast path
   if (!txn->direct_tx)
      throw std::out_of_range("MDBX direct transaction not started");
   oplog_record('U', key, value);
#ifdef SAL_STALL_DEBUG
   g_hb_seq.fetch_add(1, std::memory_order_relaxed);
   stall_hb("U-enter", key);
#endif
   txn->direct_tx->upsert(key, value);
#ifdef SAL_STALL_DEBUG
   stall_hb("U-exit", key);
   qlog(txn, 'U', 0, 0, key.data(), key.size(), 0, nullptr, 0, value.data(), value.size());
#endif
   // record in the read-own-writes cursor overlay
   txn->write_set[std::string(key)] = std::string(value);
}

static void txn_upsert_value(MDBX_txn* txn, uint32_t root_idx,
                             std::string_view key, std::string_view value,
                             bool sorted_append_hint = false)
{
   namespaced_key_buffer key_buf;
   auto physical = namespaced_key_into(root_idx, key, key_buf);
   if (!physical)
      throw std::out_of_range("MDBX logical key too large for PsiTri namespace");
   txn_upsert_physical_value(txn, *physical, value, sorted_append_hint);
}

static bool txn_remove_physical_key(MDBX_txn* txn, std::string_view key)
{
   if (!txn->direct_tx)
      return false;
   // The key is present iff it has a live value now (committed or same-txn).
   const bool present = txn->direct_tx->get(key, [](std::string_view) {});
   oplog_record('R', key, {});
#ifdef SAL_STALL_DEBUG
   g_hb_seq.fetch_add(1, std::memory_order_relaxed);
   stall_hb("R-enter", key);
#endif
   txn->direct_tx->remove(key);
#ifdef SAL_STALL_DEBUG
   stall_hb("R-exit", key);
   qlog(txn, 'R', 0, 0, key.data(), key.size(), present ? 1 : 0, nullptr, 0, nullptr, 0);
#endif
   // overlay: a same-txn tombstone that masks committed data
   txn->write_set[std::string(key)] = std::nullopt;
   return present;
}

static bool txn_remove_key(MDBX_txn* txn, uint32_t root_idx,
                           std::string_view key)
{
   namespaced_key_buffer key_buf;
   auto physical = namespaced_key_into(root_idx, key, key_buf);
   return physical && txn_remove_physical_key(txn, *physical);
}

static bool txn_remove_physical_range_any(MDBX_txn* txn, std::string_view low,
                                          std::string_view high)
{
   // delta_art has no range-remove. Enumerate every live key in [low, high)
   // through the write-txn cursor (which merges the same-txn overlay, so it
   // sees uncommitted inserts), collect them, then remove each. Collecting
   // first avoids invalidating the overlay iterator the cursor rides.
   std::vector<std::string> keys;
   {
      auto mc = make_txn_cursor(txn, mdbx_data_root_index);
      if (mc->lower_bound(low))
      {
         do
         {
            auto k = mc->key();
            if (!high.empty() && !(k < high))
               break;
            keys.emplace_back(k);
         } while (mc->next());
      }
   }
   for (const auto& k : keys)
      txn_remove_physical_key(txn, k);
   return !keys.empty();
}

static bool txn_remove_range_any(MDBX_txn* txn, uint32_t root_idx,
                                 std::string_view low, std::string_view high)
{
   namespaced_key_buffer low_buf;
   namespaced_key_buffer high_buf;
   auto physical_low = namespaced_key_into(root_idx, low, low_buf);
   auto physical_high = high.empty()
                           ? namespace_successor_into(root_idx, high_buf)
                           : namespaced_key_into(root_idx, high, high_buf);
   return physical_low &&
          txn_remove_physical_range_any(txn, *physical_low,
                                        physical_high.value_or(std::string_view{}));
}

static uint64_t count_dups_for_key(const MDBX_txn* txn, uint32_t root_idx,
                                   std::string_view key)
{
   dupsort_key_buffer prefix_buf;
   dupsort_key_buffer first_value_buf;
   auto prefix      = dupsort_prefix_into(root_idx, key, prefix_buf);
   auto first_value = dupsort_first_value_key_into(root_idx, key, first_value_buf);
   if (!prefix || !first_value)
      return 0;

   auto     mc = make_txn_cursor(txn, root_idx);
   uint64_t n  = 0;
   if (mc->lower_bound(*first_value))
   {
      do
      {
         if (!dupsort_key_is_value_for_prefix(mc->key(), *prefix))
            break;
         ++n;
      } while (mc->next());
   }
   return n;
}

static bool dupsort_key_exists(const MDBX_txn* txn, uint32_t root_idx,
                               std::string_view key)
{
   dupsort_key_buffer prefix_buf;
   dupsort_key_buffer marker_buf;
   dupsort_key_buffer first_value_buf;
   auto prefix      = dupsort_prefix_into(root_idx, key, prefix_buf);
   auto marker      = dupsort_outer_marker_key_into(root_idx, key, marker_buf);
   auto first_value = dupsort_first_value_key_into(root_idx, key, first_value_buf);
   if (!prefix || !first_value)
      return false;

   auto mc = make_txn_cursor(txn, root_idx);
	   if (marker && mc->find(*marker))
   {
      return mc->next() && dupsort_key_is_value_for_prefix(mc->key(), *prefix);
   }

   return mc->lower_bound(*first_value)
          && dupsort_key_is_value_for_prefix(mc->key(), *prefix);
}

static bool ensure_dupsort_outer_marker(MDBX_txn* txn, uint32_t root_idx,
                                        std::string_view key,
                                        bool sorted_append_hint)
{
   dupsort_key_buffer marker_buf;
   auto marker = dupsort_outer_marker_key_into(root_idx, key, marker_buf);
   if (!marker)
      return false;

   std::string existing;
   if (!txn_get_physical_value(txn, *marker, existing))
   {
      txn_upsert_physical_value(txn, *marker, std::string_view{},
                                sorted_append_hint);
   }
   return true;
}

static MDBX_cursor* acquire_cursor(MDBX_txn* txn)
{
   if (txn->free_cursors)
   {
      MDBX_cursor* c = txn->free_cursors;
      txn->free_cursors = c->next_free;
      c->next_free = nullptr;
      c->open = true;
      c->counted = false;
      return c;
   }

   txn->cursor_storage.push_back(std::make_unique<MDBX_cursor>());
   MDBX_cursor* c = txn->cursor_storage.back().get();
   c->open = true;
   return c;
}

static bool cursor_op_needs_current_position(MDBX_cursor_op op)
{
   switch (op)
   {
      case MDBX_NEXT:
      case MDBX_PREV:
      case MDBX_NEXT_DUP:
      case MDBX_PREV_DUP:
      case MDBX_NEXT_NODUP:
      case MDBX_PREV_NODUP:
      case MDBX_FIRST_DUP:
      case MDBX_LAST_DUP:
      case MDBX_GET_CURRENT:
         return true;
      default:
         return false;
   }
}

static void mark_txn_cursors_stale_after_write(MDBX_txn* txn, uint32_t root_idx)
{
   if (!txn)
      return;

   for (auto& slot : txn->cursor_storage)
   {
      MDBX_cursor* cursor = slot.get();
      if (!cursor || cursor->txn != txn || cursor->root_idx != root_idx ||
          !cursor->open || !cursor->state)
      {
         continue;
      }
      cursor->state->mark_stale_after_write();
   }
}

enum class stale_reopen_result
{
   ready,
   fulfilled,
   notfound
};

static bool sync_after_position(cursor_state& st)
{
   if (st.dupsort)
      return st.sync_dup_key_val();

   st.sync_key_val();
   return st.valid;
}

static bool move_from_deleted_current(cursor_state& st, MDBX_cursor_op op,
                                      std::string_view saved_key,
                                      std::string_view saved_raw)
{
   if (st.dupsort)
   {
      dupsort_key_buffer prefix_buf;
      auto prefix = dupsort_prefix_into(st.root_idx, saved_key, prefix_buf);
      if (!prefix)
         return false;

      bool positioned = false;
      switch (op)
      {
         case MDBX_NEXT:
            positioned = !saved_raw.empty() && st.mc->lower_bound(saved_raw);
            return positioned && st.sync_forward_dup();
         case MDBX_PREV:
            positioned = !saved_raw.empty() && st.mc->lower_bound(saved_raw);
            positioned = positioned ? st.mc->prev() : st.mc->seek_last();
            return positioned && st.sync_backward_dup();
         case MDBX_NEXT_DUP:
            positioned = !saved_raw.empty() && st.mc->lower_bound(saved_raw);
            return positioned && st.sync_forward_dup_for_prefix(*prefix);
         case MDBX_PREV_DUP:
            positioned = !saved_raw.empty() && st.mc->lower_bound(saved_raw);
            positioned = positioned ? st.mc->prev() : st.mc->seek_last();
            return positioned
                   && dupsort_key_is_value_for_prefix(st.mc->key(), *prefix)
                   && st.sync_dup_key_val();
         case MDBX_NEXT_NODUP:
         {
            dupsort_key_buffer upper_buf;
            auto upper = prefix_successor_into(*prefix, upper_buf);
            positioned = upper && st.mc->lower_bound(*upper);
            return positioned && st.sync_forward_dup();
         }
         case MDBX_PREV_NODUP:
            positioned = st.mc->lower_bound(*prefix) && st.mc->prev();
            return positioned && st.sync_backward_dup();
         case MDBX_FIRST_DUP:
            positioned = st.seek_outer_first_dup(saved_key);
            return positioned;
         case MDBX_LAST_DUP:
         {
            dupsort_key_buffer upper_buf;
            auto upper = prefix_successor_into(*prefix, upper_buf);
            if (!upper)
               return false;
            positioned = st.mc->lower_bound(*upper);
            positioned = positioned ? st.mc->prev() : st.mc->seek_last();
            return positioned && st.sync_backward_dup_for_prefix(*prefix);
         }
         default:
            return false;
      }
   }

   bool positioned = false;
   switch (op)
   {
      case MDBX_NEXT:
         positioned = st.lower_bound_kv(saved_key);
         break;
      case MDBX_PREV:
         positioned = st.lower_bound_kv(saved_key);
         positioned = positioned ? st.prev_kv() : st.seek_last_kv();
         break;
      default:
         return false;
   }

   return positioned && sync_after_position(st);
}

static stale_reopen_result reopen_stale_cursor_after_write(MDBX_cursor* cursor,
                                                           MDBX_cursor_op op)
{
   auto& st = *cursor->state;
   if (!st.stale_after_write)
      return stale_reopen_result::ready;

   auto saved_key = st.key_buf;
   auto saved_val = st.val_buf;
   auto saved_raw = st.raw_key;

   st.mc = make_txn_cursor(cursor->txn, cursor->root_idx);
   st.stale_after_write = false;

   if (!st.valid || !cursor_op_needs_current_position(op))
      return stale_reopen_result::ready;

   if (st.dupsort)
   {
      if (st.seek_dup_exact(saved_key, saved_val))
         return stale_reopen_result::ready;
   }
   else if (st.seek_kv_exact(saved_key))
   {
      st.sync_key_val();
      return stale_reopen_result::ready;
   }

   st.key_buf = std::move(saved_key);
   st.val_buf = std::move(saved_val);
   st.raw_key = std::move(saved_raw);
   st.valid   = true;

   if (!move_from_deleted_current(st, op, st.key_buf, st.raw_key))
   {
      st.valid = false;
      return stale_reopen_result::notfound;
   }

   return stale_reopen_result::fulfilled;
}

static void mark_cursor_stale_after_write(MDBX_cursor* cursor,
                                          const MDBX_val* key,
                                          const MDBX_val* data,
                                          MDBX_put_flags_t flags)
{
   if (!cursor || !cursor->state)
      return;

   (void)key;
   (void)data;
   (void)flags;

   auto& st = *cursor->state;
   st.mark_stale_after_write();
}

static bool txn_finalized(const MDBX_txn* txn)
{
   return txn && (txn->committed || txn->aborted);
}

static int reserve_write_slot(MDBX_env* env, MDBX_txn_flags_t flags)
{
   std::unique_lock lock(env->writer_mutex);
   const auto owner = std::this_thread::get_id();

   if (env->writer_active && env->writer_owner == owner)
      return MDBX_BUSY;

   if (flags & MDBX_TXN_TRY)
   {
      if (env->writer_active)
         return MDBX_BUSY;
   }
   else
   {
      env->writer_cv.wait(lock, [&] { return !env->writer_active; });
   }

   env->writer_active = true;
   env->writer_owner = owner;
   return MDBX_SUCCESS;
}

static void release_write_slot(MDBX_txn* txn)
{
   if (!txn || !txn->write_slot_reserved)
      return;

   {
      std::lock_guard lock(txn->env->writer_mutex);
      txn->env->writer_active = false;
      txn->env->writer_owner = {};
      txn->write_slot_reserved = false;
   }
   txn->env->writer_cv.notify_one();
}

static void delete_or_defer_finalized_txn(MDBX_txn* txn)
{
   if (txn && txn_finalized(txn) && txn->active_cursors == 0)
      delete txn;
}

static void release_cursor(MDBX_cursor* cursor)
{
   if (!cursor)
      return;
   if (!cursor->open && !cursor->heap_owned && !cursor->counted)
      return;

   MDBX_txn* txn = cursor->txn;
   bool finalized = txn_finalized(txn);
   bool heap_owned = cursor->heap_owned;
   cursor->state.reset();
   cursor->dbi = 0;
   cursor->root_idx = UINT32_MAX;
   cursor->is_dupsort = false;
   cursor->is_reverse_dup = false;
   cursor->open = false;
   cursor->txn = nullptr;

   if (cursor->counted)
   {
      cursor->counted = false;
      if (txn && txn->active_cursors > 0)
         --txn->active_cursors;
   }

   if (heap_owned)
   {
      delete cursor;
      delete_or_defer_finalized_txn(txn);
      return;
   }

   if (!txn)
      return;

   if (!finalized)
   {
      cursor->next_free = txn->free_cursors;
      txn->free_cursors = cursor;
   }

   delete_or_defer_finalized_txn(txn);
}

static void rollback_dbi_registry_changes(MDBX_txn* txn)
{
   if (!txn || txn->dbi_changes.empty())
      return;

   std::unique_lock lk(txn->env->dbi_mutex);
   for (auto it = txn->dbi_changes.rbegin(); it != txn->dbi_changes.rend(); ++it)
   {
      if (it->dbi >= txn->env->dbis.size())
         continue;

      auto& slot = txn->env->dbis[it->dbi];
      if (it->had_old)
      {
         if (!slot.name.empty())
            txn->env->name_to_dbi.erase(slot.name);
         slot = it->old_info;
         if (!slot.name.empty())
            txn->env->name_to_dbi[slot.name] = it->dbi;
      }
      else
      {
         if (!slot.name.empty())
            txn->env->name_to_dbi.erase(slot.name);
         slot.name.clear();
         slot.root_index = UINT32_MAX;
         slot.flags = 0;
         slot.is_dupsort = false;
         slot.reverse_dup = false;
      }
   }
   txn->dbi_changes.clear();
}

// ════════════════════════════════════════════════════════════════════
// C API implementation
// ════════════════════════════════════════════════════════════════════

// ── Version ──────────────────────────────────────────────────────

const MDBX_version_info mdbx_version = {
   0, 13, 11, 0,
   {"psitri-compat", "", "", "psitrimdbx-0.1"},
   "psitrimdbx"
};

const MDBX_build_info mdbx_build = {
   __DATE__ " " __TIME__,
#if defined(__aarch64__) || defined(_M_ARM64)
   "aarch64",
#elif defined(__x86_64__) || defined(_M_X64)
   "x86_64",
#else
   "unknown",
#endif
   "psitrimdbx",
#if defined(__clang__)
   "clang",
#elif defined(__GNUC__)
   "gcc",
#else
   "unknown",
#endif
   ""
};

// ── Error handling ───────────────────────────────────────────────

const char* mdbx_strerror(int errnum)
{
   switch (errnum)
   {
      case MDBX_SUCCESS:        return "MDBX_SUCCESS: Successful";
      case MDBX_RESULT_TRUE:    return "MDBX_RESULT_TRUE";
      case MDBX_KEYEXIST:       return "MDBX_KEYEXIST: Key already exists";
      case MDBX_NOTFOUND:       return "MDBX_NOTFOUND: No matching key/data pair found";
      case MDBX_CORRUPTED:      return "MDBX_CORRUPTED: Database is corrupted";
      case MDBX_PANIC:          return "MDBX_PANIC: Environment had fatal error";
      case MDBX_VERSION_MISMATCH: return "MDBX_VERSION_MISMATCH: Library version mismatch";
      case MDBX_INVALID:        return "MDBX_INVALID: Invalid parameter";
      case MDBX_MAP_FULL:       return "MDBX_MAP_FULL: Database map full";
      case MDBX_DBS_FULL:       return "MDBX_DBS_FULL: Maximum databases reached";
      case MDBX_READERS_FULL:   return "MDBX_READERS_FULL: Maximum readers reached";
      case MDBX_TXN_FULL:       return "MDBX_TXN_FULL: Transaction has too many dirty pages";
      case MDBX_BAD_TXN:        return "MDBX_BAD_TXN: Transaction is invalid";
      case MDBX_BAD_VALSIZE:    return "MDBX_BAD_VALSIZE: Invalid value size";
      case MDBX_BAD_DBI:        return "MDBX_BAD_DBI: Invalid database handle";
      case MDBX_PROBLEM:        return "MDBX_PROBLEM: Unexpected internal error";
      case MDBX_BUSY:           return "MDBX_BUSY: Resource is busy";
      case MDBX_EMULTIVAL:      return "MDBX_EMULTIVAL: Multiple values for a key";
      case MDBX_ENODATA:        return "MDBX_ENODATA: No data available";
      case MDBX_ENOSYS:         return "MDBX_ENOSYS: Feature not implemented";
      case MDBX_EINVAL:         return "mdbx: MDBX_EINVAL: Invalid argument";
      case MDBX_ENOMEM:         return "MDBX_ENOMEM: Out of memory";
      case MDBX_ENOFILE:        return "MDBX_ENOFILE: No such file or directory";
      case MDBX_PAGE_NOTFOUND:  return "MDBX_PAGE_NOTFOUND: Page not found";
      case MDBX_INCOMPATIBLE:   return "MDBX_INCOMPATIBLE: Incompatible operation";
      case MDBX_UNABLE_EXTEND_MAPSIZE: return "MDBX_UNABLE_EXTEND_MAPSIZE: Unable to extend map size";
      case MDBX_EKEYMISMATCH:   return "MDBX_EKEYMISMATCH: Key mismatch";
      default:                  return "Unknown MDBX error";
   }
}

const char* mdbx_strerror_r(int errnum, char* buf, size_t buflen)
{
   const char* msg = mdbx_strerror(errnum);
   if (buf && buflen > 0)
   {
      std::strncpy(buf, msg, buflen - 1);
      buf[buflen - 1] = '\0';
   }
   return buf ? buf : msg;
}

// ── Environment ──────────────────────────────────────────────────

int mdbx_env_create(MDBX_env** penv)
{
   if (!penv)
      return MDBX_EINVAL;
   *penv = new MDBX_env();
   (*penv)->page_size = system_page_size();
   return MDBX_SUCCESS;
}

// Map MDBX env durability flags to the adb commit-sync level (the commit config that
// flows to BOTH tiers). Matches MDBX semantics: a synced commit is the boundary that
// writes the seal record so the cold base compacts/recycles — without it the file grows
// by the full COW-garbage rate. Default (SYNC_DURABLE, flags==0) is a true fsync.
static sal::sync_type mdbx_flags_to_sync(MDBX_env_flags_t flags) noexcept
{
   if ((flags & MDBX_UTTERLY_NOSYNC) == MDBX_UTTERLY_NOSYNC)
      return sal::sync_type::none;
   if (flags & MDBX_SAFE_NOSYNC)
      return sal::sync_type::msync_async;
   if (flags & MDBX_NOMETASYNC)
      return sal::sync_type::msync_sync;
   return sal::sync_type::fsync;  // MDBX_SYNC_DURABLE (MDBX's default)
}

int mdbx_env_open(MDBX_env* env, const char* pathname,
                  MDBX_env_flags_t flags, mdbx_mode_t /*mode*/)
{
   if (!env || !pathname)
      return MDBX_EINVAL;
   if (env->opened)
      return MDBX_EINVAL;

   try
   {
      env->path      = pathname;
      env->env_flags = flags;
      if (!env->write_mode_explicit)
      {
         if (auto mode = parse_psitri_write_mode(shim_env::get().write_mode))
            env->write_mode = *mode;
      }

      // Determine open mode. MDBX_RDONLY is advisory: the shim enforces
      // read-only at the transaction layer (is_readonly()). The start_ver hot
      // delta has no true read-only open yet, so open create_or_open and rely
      // on the txn-layer guard — opening an existing DB never creates data.
      auto open_mode = sal::open_mode::create_or_open;

#if PSITRIMDBX_BACKEND_START_VER
      // start_ver standalone: no tiered knobs (no hot/cold flush threshold).
      env->db = be_db::open(env->path, open_mode);
#elif PSITRIMDBX_BACKEND_START_REF
      // start_ref standalone: part-family ART, refcount liveness; same open shape as start_ver.
      env->db = be_db::open(env->path, open_mode);
#elif PSITRIMDBX_BACKEND_GCART
      // gcart spec engine standalone: node-level COW + refcount; same 2-arg open shape.
      // Segment caps come from the GCART_*_SEGS env knobs (Silkworm-scale defaults).
      env->db = be_db::open(env->path, open_mode);
#elif PSITRIMDBX_BACKEND_RCART
      // rcart refcount-baseline standalone: node-level COW + refcount + hazard readers;
      // same 2-arg open shape. Segment caps from the RCART_*_SEGS env knobs.
      env->db = be_db::open(env->path, open_mode);
#elif PSITRIMDBX_BACKEND_PSITRI
      // legacy psitri direct: config is sal::runtime_config (no tiered knobs).
      psitri::runtime_config cfg;
      env->db = be_db::open(env->path, open_mode, cfg);
#elif PSITRIMDBX_BACKEND_DWAL
      if (mdbx_flags_to_sync(flags) != sal::sync_type::none)
         throw std::runtime_error("DWAL byte adapter currently supports no-sync commits only");
      TK::config cfg;
      cfg.base.enable_read_cache = !shim_env::get().no_read_cache;
      if (const auto mb = shim_env::get().cache_mb) cfg.base.max_pinned_cache_size_mb = mb;
      cfg.base.sync_mode = sal::sync_type::none;
      env->db = be_db::open(env->path, open_mode, cfg);
#elif PSITRIMDBX_BACKEND_SAL_ART_REF
      // sal_art_ref DIRECT: config is sal::runtime_config (no tiered knobs, no merger).
      sal::runtime_config cfg;
      // [stall-bisect] read-cache knobs to isolate the compactor/pinned-cache stall.
      //   PSITRI_NO_READ_CACHE=1     → disable the read cache entirely (no pinned-segment churn)
      //   PSITRI_CACHE_MB=<n>        → cap the pinned cache at n MB (multiple of 32)
      if (shim_env::get().no_read_cache)
         cfg.enable_read_cache = false;
      if (const uint64_t mb = shim_env::get().cache_mb)
         cfg.max_pinned_cache_size_mb = mb;
      env->db = be_db::open(env->path, open_mode, cfg);
#else
      TK::config cfg;
      // Durability: derive the adb commit-sync from the MDBX env flags (the commit config),
      // fanned out to both tiers at open. PSITRI_SYNC overrides for A/B benchmarking.
      cfg.commit_sync = mdbx_flags_to_sync(flags);
      if (const char* s = shim_env::get().sync)
      {
         const std::string_view v{s};
         if (v == "none")                              cfg.commit_sync = sal::sync_type::none;
         else if (v == "async" || v == "msync_async")  cfg.commit_sync = sal::sync_type::msync_async;
         else if (v == "msync" || v == "msync_sync")   cfg.commit_sync = sal::sync_type::msync_sync;
         else if (v == "fsync")                        cfg.commit_sync = sal::sync_type::fsync;
         else if (v == "full")                         cfg.commit_sync = sal::sync_type::full;
      }
      // Optional delta_art knobs from the psitri cache options (best-effort
      // mapping; cache_size -> arena budget, in MB).
      if (auto it = env->options.find(MDBX_opt_psitri_cache_size_mb);
          it != env->options.end() && it->second)
         cfg.knobs.arena_budget_bytes = it->second << 20;
      // Flush-threshold override (testing/tuning): force hot->cold flushes at a
      // chosen hot-delta size so large-value flush paths get exercised.
      if (const uint64_t hw = shim_env::get().hi_water_mb)
         cfg.knobs.hi_water_bytes = hw << 20;

      env->db = be_db::open(env->path, open_mode, cfg);
#endif
      env->ws = env->db->start_write_session();

      if (!(flags & MDBX_RDONLY))
      {
         std::filesystem::create_directories(env->path);
         auto marker = env->path / "mdbx.dat";
         env->page_size = read_marker_page_size(marker).value_or(
            requested_or_default_page_size(env->geo_pagesize));
         if (!std::filesystem::exists(marker) ||
             std::filesystem::file_size(marker) == 0 ||
             !read_marker_page_size(marker))
            write_marker_page_size(marker, env->page_size);
      }
      else
      {
         auto marker = env->path / "mdbx.dat";
         env->page_size = read_marker_page_size(marker).value_or(
            requested_or_default_page_size(env->geo_pagesize));
      }

      env->init_default_dbi();

      // Restore named DBIs from the catalog (DBI 1 namespace inside the single
      // delta_art root). Each catalog entry is name -> {flags, root_idx}; the
      // mapping/encoding is unchanged, only the backend cursor differs.
      try
      {
         auto rs  = env->db->start_read_session();
         auto cur = rs->create_cursor(be_root_latest);
         namespaced_key_buffer prefix_buf;
         auto scan_prefix = namespace_prefix_into(1, prefix_buf);
         if (cur.lower_bound(scan_prefix))
         {
            do
            {
               auto raw_key = cur.key();
               if (!key_has_namespace(raw_key, 1))
                  break;
               raw_key = strip_namespace(raw_key);

               auto name = std::string(raw_key.data(), raw_key.size());
               if (env->name_to_dbi.contains(name))
                  continue;

               auto val = cur.value<std::string>();
               unsigned f = 0;
               uint32_t r = 0;
               if (!val || !decode_catalog_entry(*val, f, r, false))
                  continue;

               if (r != 0)
                  env->allocate_dbi_with_root(name, f, r);
               else
                  env->allocate_dbi(name, f);
            } while (cur.next());
         }
      }
      catch (...)
      {
         // If catalog read fails on first open, no named DBIs to restore.
      }

      env->opened = true;

      return MDBX_SUCCESS;
   }
   catch (const std::exception&)
   {
      return MDBX_PANIC;
   }
}

int mdbx_env_close(MDBX_env* env)
{
   return mdbx_env_close_ex(env, 0);
}

int mdbx_env_close_ex(MDBX_env* env, int dont_sync)
{
   if (!env)
      return MDBX_EINVAL;

   // Teardown: drop the shared write session, then the database. delta_art's
   // database destructor joins the background merger; the hot delta's committed
   // root is the durable state (no WAL), so recovery on reopen re-flushes
   // whatever the warm slot still holds.
   (void)dont_sync;

   env->ws.reset();
   env->db.reset();
   delete env;
   return MDBX_SUCCESS;
}

int mdbx_env_set_geometry(MDBX_env* env,
                          intptr_t size_lower, intptr_t size_now,
                          intptr_t size_upper, intptr_t growth_step,
                          intptr_t shrink_threshold, intptr_t pagesize)
{
   if (!env)
      return MDBX_EINVAL;
   // Store but don't enforce — psitri manages its own storage
   env->geo_lower    = size_lower;
   env->geo_now      = size_now;
   env->geo_upper    = size_upper;
   env->geo_growth   = growth_step;
   env->geo_shrink   = shrink_threshold;
   env->geo_pagesize = pagesize;
   return MDBX_SUCCESS;
}

int mdbx_env_set_maxdbs(MDBX_env* env, MDBX_dbi dbs)
{
   if (!env || env->opened)
      return MDBX_EINVAL;
   env->max_dbs = std::min<MDBX_dbi>(dbs, max_named_dbs());
   return MDBX_SUCCESS;
}

int mdbx_env_set_maxreaders(MDBX_env* env, unsigned readers)
{
   if (!env || env->opened)
      return MDBX_EINVAL;
   env->max_readers = readers;
   return MDBX_SUCCESS;
}

int mdbx_env_sync_ex(MDBX_env* env, int /*force*/, int /*nonblock*/)
{
   if (!env || !env->opened)
      return MDBX_EINVAL;
   // delta_art has no WAL: a committed transaction's hot-delta root is already
   // the durable state. Nothing to flush here.
   return MDBX_SUCCESS;
}

int mdbx_env_get_flags(MDBX_env* env, unsigned* flags)
{
   if (!env || !flags)
      return MDBX_EINVAL;
   *flags = env->env_flags;
   return MDBX_SUCCESS;
}

int mdbx_env_stat_ex(const MDBX_env* env, const MDBX_txn* /*txn*/,
                     MDBX_stat* stat, size_t bytes)
{
   if (!env || !stat || bytes < sizeof(MDBX_stat))
      return MDBX_EINVAL;
   std::memset(stat, 0, sizeof(MDBX_stat));
   stat->ms_psize = env->page_size;
   return MDBX_SUCCESS;
}

int mdbx_env_info_ex(const MDBX_env* env, const MDBX_txn* /*txn*/,
                     MDBX_envinfo* info, size_t bytes)
{
   if (!env || !info || bytes < sizeof(MDBX_envinfo))
      return MDBX_EINVAL;
   std::memset(info, 0, sizeof(MDBX_envinfo));
   info->mi_dxb_pagesize = env->page_size;
   info->mi_sys_pagesize = system_page_size();
   return MDBX_SUCCESS;
}

void* mdbx_env_get_userctx(const MDBX_env* env)
{
   return env ? env->userctx : nullptr;
}

int mdbx_env_set_userctx(MDBX_env* env, void* ctx)
{
   if (!env)
      return MDBX_EINVAL;
   env->userctx = ctx;
   return MDBX_SUCCESS;
}

int mdbx_env_set_option(MDBX_env* env, MDBX_option_t option, uint64_t value)
{
   if (!env)
      return MDBX_EINVAL;

   switch (option)
   {
      case MDBX_opt_max_db:
         return mdbx_env_set_maxdbs(env, static_cast<MDBX_dbi>(value));
      case MDBX_opt_max_readers:
         return mdbx_env_set_maxreaders(env, static_cast<unsigned>(value));
      case MDBX_opt_psitri_write_mode:
         return mdbx_env_set_write_mode(env, static_cast<int>(value));
      default:
         env->options[static_cast<int>(option)] = value;
         return MDBX_SUCCESS;
   }
}

int mdbx_env_get_option(const MDBX_env* env, MDBX_option_t option,
                        uint64_t* value)
{
   if (!env || !value)
      return MDBX_EINVAL;

   switch (option)
   {
      case MDBX_opt_max_db:
         *value = env->max_dbs;
         return MDBX_SUCCESS;
      case MDBX_opt_max_readers:
         *value = env->max_readers;
         return MDBX_SUCCESS;
      case MDBX_opt_psitri_write_mode:
         *value = static_cast<uint64_t>(env->write_mode);
         return MDBX_SUCCESS;
      default:
      {
         auto it = env->options.find(static_cast<int>(option));
         *value = it == env->options.end() ? 0 : it->second;
         return MDBX_SUCCESS;
      }
   }
}

int mdbx_env_copy(MDBX_env* env, const char* dest, unsigned /*flags*/)
{
   if (!env || !dest || !env->opened)
      return MDBX_EINVAL;

   try
   {
      std::filesystem::copy(
         env->path, dest,
         std::filesystem::copy_options::recursive |
            std::filesystem::copy_options::overwrite_existing);
      return MDBX_SUCCESS;
   }
   catch (...)
   {
      return MDBX_PANIC;
   }
}

int mdbx_env_set_write_mode(MDBX_env* env, int mode)
{
   if (!env || (mode != PSITRI_WRITE_MODE_DIRECT &&
                mode != PSITRI_WRITE_MODE_DWAL))
      return MDBX_EINVAL;
   if (env->opened)
      return MDBX_EINVAL;
   env->write_mode = mode;
   env->write_mode_explicit = true;
   return MDBX_SUCCESS;
}

int mdbx_env_set_read_mode(MDBX_env* env, int mode)
{
   if (!env || mode < 0 || mode > 2)
      return MDBX_EINVAL;
   env->read_mode = mode;
   return MDBX_SUCCESS;
}

// ── Transactions ─────────────────────────────────────────────────

int mdbx_txn_begin_ex(MDBX_env* env, MDBX_txn* parent,
                      MDBX_txn_flags_t flags, MDBX_txn** txn,
                      void* context)
{
   if (!env || !env->opened || !txn)
      return MDBX_EINVAL;
   if (parent)
      return MDBX_ENOSYS; // Nested transactions not yet supported

   try
   {
      auto t       = std::make_unique<MDBX_txn>();
      t->env       = env;
      t->txn_flags = flags;
      t->context   = context;
      t->id        = env->next_txn_id.fetch_add(1);

      if (flags & MDBX_TXN_RDONLY)
      {
         capture_direct_read_roots(t.get());
      }
      else
      {
         const int rc = reserve_write_slot(env, flags);
         if (rc != MDBX_SUCCESS)
            return rc;
         t->write_slot_reserved = true;

         try
         {
            start_direct_write_txn(t.get());
         }
         catch (...)
         {
            release_write_slot(t.get());
            throw;
         }
      }

      *txn = t.release();
      return MDBX_SUCCESS;
   }
   catch (...)
   {
      return MDBX_ENOMEM;
   }
}

int mdbx_txn_commit_ex(MDBX_txn* txn, MDBX_commit_latency* latency)
{
   if (!txn)
      return MDBX_EINVAL;
   if (txn->committed || txn->aborted)
      return MDBX_BAD_TXN;

   if (latency)
      std::memset(latency, 0, sizeof(MDBX_commit_latency));

   try
   {
      if (txn->direct_tx)
      {
         oplog_record('C', {}, {});
#ifdef SAL_STALL_DEBUG
         stall_hb("C-enter", {});
#endif
         txn->direct_tx->commit();
#ifdef SAL_STALL_DEBUG
         stall_hb("C-exit", {});
         qlog(txn, 'C', 0, 0, nullptr, 0, 0, nullptr, 0, nullptr, 0);
#endif
         txn->direct_tx.reset();
#ifdef SAL_STALL_DEBUG
         stall_hb("C-reset", {});
#endif
         txn->write_set.clear();
         release_write_slot(txn);
#ifdef SAL_STALL_DEBUG
         stall_hb("C-done", {});  // shim commit fully returned; any further hang is Silkworm-side
#endif
      }

      txn->dbi_changes.clear();
      txn->committed = true;
      delete_or_defer_finalized_txn(txn);
      return MDBX_SUCCESS;
   }
   catch (...)
   {
      release_write_slot(txn);
      rollback_dbi_registry_changes(txn);
      txn->aborted = true;
      delete_or_defer_finalized_txn(txn);
      return MDBX_PANIC;
   }
}

int mdbx_txn_abort(MDBX_txn* txn)
{
   if (!txn)
      return MDBX_EINVAL;
   if (txn->committed || txn->aborted)
   {
      delete_or_defer_finalized_txn(txn);
      return MDBX_SUCCESS;
   }

   try
   {
      if (txn->direct_tx)
      {
         txn->direct_tx->abort();
         txn->direct_tx.reset();
         txn->write_set.clear();
         release_write_slot(txn);
      }

      rollback_dbi_registry_changes(txn);
      txn->aborted = true;
      delete_or_defer_finalized_txn(txn);
      return MDBX_SUCCESS;
   }
   catch (...)
   {
      release_write_slot(txn);
      rollback_dbi_registry_changes(txn);
      txn->aborted = true;
      delete_or_defer_finalized_txn(txn);
      return MDBX_PANIC;
   }
}

int mdbx_txn_reset(MDBX_txn* txn)
{
   if (!txn || !txn->is_readonly())
      return MDBX_BAD_TXN;
   txn->read_cursor.reset();
   txn->read_session.reset();
   txn->has_read_view = false;
   txn->returned_slices.clear();
   return MDBX_SUCCESS;
}

int mdbx_txn_renew(MDBX_txn* txn)
{
   if (!txn || !txn->is_readonly())
      return MDBX_BAD_TXN;
   try
   {
      txn->returned_slices.clear();
      txn->read_cursor.reset();
      txn->read_session.reset();
      txn->has_read_view = false;
      capture_direct_read_roots(txn);
      return MDBX_SUCCESS;
   }
   catch (...)
   {
      return MDBX_PANIC;
   }
}

MDBX_env* mdbx_txn_env(const MDBX_txn* txn)
{
   return txn ? txn->env : nullptr;
}

uint64_t mdbx_txn_id(const MDBX_txn* txn)
{
   return txn ? txn->id : 0;
}

int mdbx_txn_flags(const MDBX_txn* txn)
{
   return txn ? static_cast<int>(txn->txn_flags) : MDBX_EINVAL;
}

static int ensure_rw_root(MDBX_txn* txn, uint32_t root_idx);

// ── DBI operations ───────────────────────────────────────────────

int mdbx_dbi_open(MDBX_txn* txn, const char* name,
                  MDBX_db_flags_t flags, MDBX_dbi* dbi)
{
   if (!txn || !dbi)
      return MDBX_EINVAL;

   MDBX_env* env = txn->env;

   // Unnamed default database
   if (!name || name[0] == '\0')
   {
      *dbi = 1; // DBI 1 = default unnamed DB
      cache_txn_dbi_metadata(txn, *dbi, dbi_metadata(env, *dbi));
      return MDBX_SUCCESS;
   }

   std::string sname(name);
   {
      std::shared_lock lk(env->dbi_mutex);
      auto it = env->name_to_dbi.find(sname);
      if (it != env->name_to_dbi.end())
      {
         *dbi = it->second;
         const auto& info = env->dbis[*dbi];
         cache_txn_dbi_metadata(
            txn, *dbi,
            {info.root_index, info.is_dupsort, info.reverse_dup});
         return MDBX_SUCCESS;
      }
   }

   // Not found — create if requested
   if (!(flags & MDBX_CREATE))
      return MDBX_NOTFOUND;

   if (txn->is_readonly())
      return MDBX_EACCESS;

   {
      std::unique_lock lk(env->dbi_mutex);
      // Double-check
      auto it = env->name_to_dbi.find(sname);
      if (it != env->name_to_dbi.end())
      {
         *dbi = it->second;
         return MDBX_SUCCESS;
      }

      if (env->dbis.size() >= env->max_dbs + 2)
         return MDBX_DBS_FULL;

      *dbi = env->allocate_dbi(sname, flags);
      const auto& info = env->dbis[*dbi];
      cache_txn_dbi_metadata(
         txn, *dbi, {info.root_index, info.is_dupsort, info.reverse_dup});
      txn->dbi_changes.push_back({*dbi, {}, false});
   }

   // Persist the DBI catalog record through the active transaction. MDBX
   // exposes named DB records through the unnamed main DBI, so root 1 carries
   // the public catalog. The value stores both flags and the PsiTri root id so
   // reopening in sorted catalog order cannot remap table data to new roots.
   try
   {
      int rc = ensure_rw_root(txn, 1);
      if (rc != MDBX_SUCCESS)
      {
         rollback_dbi_registry_changes(txn);
         return rc;
      }
      uint32_t root_idx = txn_dbi_metadata(txn, *dbi).root_index;
      auto catalog_value = encode_catalog_entry(flags, root_idx);
      txn_upsert_value(txn, 1, sname, std::string_view(catalog_value));
   }
   catch (...)
   {
      rollback_dbi_registry_changes(txn);
      return MDBX_PANIC;
   }

#ifdef PSITRI_DEBUG_BUILD
   if (sal::env::get("PSITRI_TRACE"))
      std::fprintf(stderr, "[TRACE] dbi_open name=%s dbi=%u root=%u flags=0x%x\n", sname.c_str(),
                   unsigned(*dbi), txn_dbi_metadata(txn, *dbi).root_index, unsigned(flags));
#endif
   return MDBX_SUCCESS;
}

int mdbx_dbi_close(MDBX_env* /*env*/, MDBX_dbi /*dbi*/)
{
   // No-op: psitri roots are persistent
   return MDBX_SUCCESS;
}

int mdbx_dbi_stat(const MDBX_txn* txn, MDBX_dbi dbi,
                  MDBX_stat* stat, size_t bytes)
{
   if (!txn || !stat || bytes < sizeof(MDBX_stat))
      return MDBX_EINVAL;

   auto meta = txn_dbi_metadata(txn, dbi);
   if (meta.root_index == UINT32_MAX)
      return MDBX_BAD_DBI;

   std::memset(stat, 0, sizeof(MDBX_stat));
   stat->ms_psize = txn->env->page_size;

   try
   {
      uint32_t root_idx = meta.root_index;
      bool is_ds  = meta.is_dupsort;
      bool rev_ds = meta.reverse_dup;
      uint64_t count = 0;
      uint64_t overflow_pages = 0;

      if (is_ds)
      {
         cursor_state st(make_txn_cursor(txn, root_idx), root_idx, true, rev_ds);
         if (st.seek_first_dup())
         {
            do
            {
               if (st.valid)
                  ++count;
            } while (st.next_dup_same() || st.next_outer_first_dup());
         }
      }
      else
      {
         cursor_state st(make_txn_cursor(txn, root_idx), root_idx, false, false);
         if (st.seek_first_kv())
         {
            do
            {
               st.sync_key_val();
               if (!st.valid)
                  continue;

               ++count;
               size_t max_inline =
                  max_single_value_size(txn->env->page_size, st.key_buf.size());
               if (st.val_buf.size() > max_inline)
                  ++overflow_pages;
            } while (st.next_kv());
         }
      }

      stat->ms_entries = count;
      if (count)
      {
         stat->ms_depth = 1;
         stat->ms_leaf_pages = 1;
      }
      stat->ms_overflow_pages = overflow_pages;
   }
   catch (...)
   {
      // If counting fails, return zeroed stats rather than failing.
   }

   return MDBX_SUCCESS;
}

int mdbx_drop(MDBX_txn* txn, MDBX_dbi dbi, int del)
{
   if (!txn || txn->is_readonly())
      return MDBX_BAD_TXN;

   auto meta = txn_dbi_metadata(txn, dbi);
   if (meta.root_index == UINT32_MAX)
      return MDBX_BAD_DBI;
   uint32_t root_idx = meta.root_index;

   {
      int rc = ensure_rw_root(txn, root_idx);
      if (rc != MDBX_SUCCESS)
         return rc;
   }

   try
   {
      // Clear the whole logical DBI namespace in the single physical root.
      namespaced_key_buffer low_buf;
      namespaced_key_buffer high_buf;
      auto low  = namespace_prefix_into(root_idx, low_buf);
      auto high = namespace_successor_into(root_idx, high_buf);
      txn_remove_physical_range_any(txn, low, high.value_or(std::string_view{}));

      if (del)
      {
         // Remove the DBI entry through the active transaction so abort can
         // restore both the catalog and the in-memory DBI registry.
         std::unique_lock lk(txn->env->dbi_mutex);
         if (dbi < txn->env->dbis.size() && !txn->env->dbis[dbi].name.empty())
         {
            auto old = txn->env->dbis[dbi];
            auto name = old.name;
            txn->dbi_changes.push_back({dbi, old, true});
            txn->env->name_to_dbi.erase(name);
            txn->env->dbis[dbi].name.clear();
            txn->env->dbis[dbi].root_index = UINT32_MAX;
            txn->env->dbis[dbi].flags = 0;
            txn->env->dbis[dbi].is_dupsort = false;
            txn->env->dbis[dbi].reverse_dup = false;
            cache_txn_dbi_metadata(txn, dbi, {});

            lk.unlock();

            int rc = ensure_rw_root(txn, 1);
            if (rc != MDBX_SUCCESS)
            {
               rollback_dbi_registry_changes(txn);
               return rc;
            }
            txn_remove_key(txn, 1, name);
         }
      }

      return MDBX_SUCCESS;
   }
   catch (...)
   {
      rollback_dbi_registry_changes(txn);
      return MDBX_PANIC;
   }
}

int mdbx_dbi_flags_ex(const MDBX_txn* txn, MDBX_dbi dbi,
                      unsigned* flags, unsigned* state)
{
   if (!txn || !flags)
      return MDBX_EINVAL;
   std::shared_lock lk(txn->env->dbi_mutex);
   if (dbi >= txn->env->dbis.size())
      return MDBX_BAD_DBI;
   *flags = txn->env->dbis[dbi].flags;
   if (state)
      *state = 0;
   return MDBX_SUCCESS;
}

// ── Key-value operations ─────────────────────────────────────────

#ifdef PSITRI_DEBUG_BUILD
// [PSITRI_NF_LOG] Debug: keep ONLY the last NOTFOUND read (near-zero cost — no per-call I/O,
// since Silkworm legitimately probes-via-cursor millions of times) and dump it at process exit.
// The fatal MDBX_NOTFOUND is the very last read before exit, so the last-recorded entry is it.
static const bool      g_nf_on = sal::env::get("PSITRI_NF_LOG") != nullptr;
static char            g_lnf_key[80];
static char            g_lnf_where[40];
static std::atomic<uint32_t> g_lnf_klen{0}, g_lnf_dbi{0}, g_lnf_root{0};
static std::atomic<uint64_t> g_lnf_count{0};
static std::atomic<bool>     g_lnf_registered{false};

static void dbg_nf_dump_atexit()
{
   const char* path = sal::env::get("PSITRI_NF_LOG");
   if (!path)
      return;
   FILE* f = std::fopen(path, "w");
   if (!f)
      return;
   uint32_t kl = g_lnf_klen.load();
   if (kl > sizeof g_lnf_key)
      kl = sizeof g_lnf_key;
   std::fprintf(f, "LAST_NF where=%s dbi=%u root=%u klen=%u total=%llu key=", g_lnf_where,
                g_lnf_dbi.load(), g_lnf_root.load(), kl,
                static_cast<unsigned long long>(g_lnf_count.load()));
   for (uint32_t i = 0; i < kl; ++i)
      std::fprintf(f, "%02x", static_cast<unsigned char>(g_lnf_key[i]));
   std::fprintf(f, "\n");
   std::fclose(f);
}

static inline void dbg_log_notfound(MDBX_dbi dbi, uint32_t root_idx, std::string_view key,
                                    const char* where)
{
   if (!g_nf_on)
      return;
   bool exp = false;
   if (g_lnf_registered.compare_exchange_strong(exp, true))
      std::atexit(dbg_nf_dump_atexit);
   uint32_t kl = key.size() < sizeof g_lnf_key ? uint32_t(key.size()) : uint32_t(sizeof g_lnf_key);
   std::memcpy(g_lnf_key, key.data(), kl);
   std::snprintf(g_lnf_where, sizeof g_lnf_where, "%s", where);
   g_lnf_klen.store(kl);
   g_lnf_dbi.store(unsigned(dbi));
   g_lnf_root.store(root_idx);
   g_lnf_count.fetch_add(1, std::memory_order_relaxed);
}
#else
static inline void dbg_log_notfound(MDBX_dbi, uint32_t, std::string_view, const char*) {}
#endif  // PSITRI_DEBUG (PSITRI_NF_LOG)

int mdbx_get(const MDBX_txn* txn, MDBX_dbi dbi,
             const MDBX_val* key, MDBX_val* data)
{
   if (!txn || !key || !data)
      return MDBX_EINVAL;

   auto meta = txn_dbi_metadata(txn, dbi);
   if (meta.root_index == UINT32_MAX)
      return MDBX_BAD_DBI;

   auto    key_sv = to_sv(key);
   auto*   mtxn   = const_cast<MDBX_txn*>(txn);
   uint32_t root_idx = meta.root_index;
   bool    is_ds  = meta.is_dupsort;
   bool    rev_ds = meta.reverse_dup;

   if (key_too_large(key_sv))
      return MDBX_NOTFOUND;

   try
   {
      if (is_ds)
      {
         cursor_state st(make_txn_cursor(txn, root_idx), root_idx, true, rev_ds);
         if (!st.seek_outer_first_dup(key_sv))
         {
            dbg_log_notfound(dbi, root_idx, key_sv, "get_ds");
            return MDBX_NOTFOUND;
         }

         *data = hold_txn_slice(mtxn, st.val_buf);
         return MDBX_SUCCESS;
      }

      // Non-DUPSORT path
      std::string value;
      if (!txn_get_value(txn, root_idx, key_sv, value))
      {
         dbg_log_notfound(dbi, root_idx, key_sv, "get_plain");
         return MDBX_NOTFOUND;
      }
      *data = hold_txn_slice(mtxn, value);
      return MDBX_SUCCESS;
   }
   catch (...)
   {
      return MDBX_PANIC;
   }
}

int mdbx_get_ex(const MDBX_txn* txn, MDBX_dbi dbi,
                MDBX_val* key, MDBX_val* data, size_t* values_count)
{
   int rc = mdbx_get(txn, dbi, key, data);
   if (values_count)
   {
      *values_count = 1;
      auto meta = txn_dbi_metadata(txn, dbi);
      if (rc == MDBX_SUCCESS && meta.is_dupsort)
      {
         auto key_sv = to_sv(key);
         *values_count = static_cast<size_t>(
            count_dups_for_key(txn, meta.root_index, key_sv));
      }
   }
   return rc;
}

/// Ensure the write transaction is live. The shim is single-keyspace: the
/// dbi number is a key namespace prefix into the one delta_art root, so every
/// dbi is "covered" once the transaction exists.
static int ensure_rw_root(MDBX_txn* txn, uint32_t root_idx)
{
   if (!txn || txn->is_readonly())
      return MDBX_BAD_TXN;
   if (namespace_too_large(root_idx))
      return MDBX_BAD_DBI;

   return txn->direct_tx ? MDBX_SUCCESS : MDBX_BAD_TXN;
}

static int mdbx_put_impl(MDBX_txn* txn, uint32_t root_idx, bool is_ds,
                         bool rev_ds, const MDBX_val* key, MDBX_val* data,
                         MDBX_put_flags_t flags)
{
   if (!txn || !key || !data || txn->is_readonly())
      return MDBX_EINVAL;

   if (root_idx == UINT32_MAX)
      return MDBX_BAD_DBI;

   int rc = ensure_rw_root(txn, root_idx);
   if (rc != MDBX_SUCCESS)
      return rc;

   auto key_sv = to_sv(key);
   auto val_sv = to_sv(data);

   if (key_too_large(key_sv))
      return MDBX_BAD_VALSIZE;

   try
   {
      if (is_ds)
      {
         dupsort_key_buffer composite_buf;
         auto composite = dupsort_composite_key_into(root_idx, key_sv, val_sv,
                                                     rev_ds, composite_buf);
         if (!composite)
            return MDBX_BAD_VALSIZE;

         if (flags & MDBX_NODUPDATA)
         {
            auto mc = make_txn_cursor(txn, root_idx);
            if (mc->find(*composite))
               return MDBX_KEYEXIST;
         }
         if (flags & MDBX_NOOVERWRITE)
         {
            if (dupsort_key_exists(txn, root_idx, key_sv))
               return MDBX_KEYEXIST;
         }

         bool append_hint = (flags & (MDBX_APPEND | MDBX_APPENDDUP)) != 0;
         if (!ensure_dupsort_outer_marker(txn, root_idx, key_sv, append_hint))
            return MDBX_BAD_VALSIZE;

         txn_upsert_physical_value(txn, *composite, std::string_view{},
                                   append_hint);
         mark_txn_cursors_stale_after_write(txn, root_idx);
         return MDBX_SUCCESS;
      }

      // Non-DUPSORT path
      // The tiered hot delta stores a value in one word-confined line run; a
      // value too large to fit a leaf is rejected up front (native MDBX returns
      // MDBX_BAD_VALSIZE for oversize values) rather than panicking deep in the
      // backend allocator.
      if (val_sv.size() > max_backend_value_size)
         return MDBX_BAD_VALSIZE;

      if (flags & MDBX_NOOVERWRITE)
      {
         std::string existing;
         if (txn_get_value(txn, root_idx, key_sv, existing))
         {
            *data = hold_txn_slice(txn, existing);
            return MDBX_KEYEXIST;
         }
      }

      txn_upsert_value(txn, root_idx, key_sv, val_sv,
                       (flags & (MDBX_APPEND | MDBX_APPENDDUP)) != 0);
      mark_txn_cursors_stale_after_write(txn, root_idx);
      return MDBX_SUCCESS;
   }
   catch (const std::exception& e)
   {
      std::fprintf(stderr, "MDBX_PANIC put_impl: keylen=%zu vallen=%zu what=%s\n",
                   key->iov_len, data->iov_len, e.what());
      std::fflush(stderr);
      return MDBX_PANIC;
   }
   catch (...)
   {
      std::fprintf(stderr, "MDBX_PANIC put_impl: non-std exception\n");
      std::fflush(stderr);
      return MDBX_PANIC;
   }
}

int mdbx_put(MDBX_txn* txn, MDBX_dbi dbi,
             const MDBX_val* key, MDBX_val* data,
             MDBX_put_flags_t flags)
{
   if (!txn || !key || !data || txn->is_readonly())
      return MDBX_EINVAL;

   auto meta = txn_dbi_metadata(txn, dbi);
#ifdef PSITRI_DEBUG_BUILD
   if (sal::env::get("PSITRI_TRACE"))
      std::fprintf(stderr, "[TRACE] put dbi=%u root=%u klen=%zu vlen=%zu\n", unsigned(dbi),
                   meta.root_index, key->iov_len, data->iov_len);
#endif
   return mdbx_put_impl(txn, meta.root_index, meta.is_dupsort,
                        meta.reverse_dup, key, data, flags);
}

int mdbx_del(MDBX_txn* txn, MDBX_dbi dbi,
             const MDBX_val* key, const MDBX_val* data)
{
   if (!txn || !key || txn->is_readonly())
      return MDBX_EINVAL;

   auto meta = txn_dbi_metadata(txn, dbi);
   if (meta.root_index == UINT32_MAX)
      return MDBX_BAD_DBI;
   uint32_t root_idx = meta.root_index;

   int rc = ensure_rw_root(txn, root_idx);
   if (rc != MDBX_SUCCESS)
      return rc;

   auto key_sv = to_sv(key);
   bool is_ds  = meta.is_dupsort;
   bool rev_ds = meta.reverse_dup;

   if (key_too_large(key_sv))
      return MDBX_BAD_VALSIZE;

   try
   {
      if (is_ds)
      {
         if (data)
         {
            auto val_sv = to_sv(data);
            dupsort_key_buffer composite_buf;
            auto composite = dupsort_composite_key_into(root_idx, key_sv, val_sv,
                                                        rev_ds, composite_buf);
            if (!composite)
               return MDBX_BAD_VALSIZE;

            bool removed = txn_remove_physical_key(txn, *composite);
            if (removed)
            {
               if (!dupsort_key_exists(txn, root_idx, key_sv))
               {
                  dupsort_key_buffer marker_buf;
                  if (auto marker = dupsort_outer_marker_key_into(root_idx, key_sv,
                                                                  marker_buf))
                     txn_remove_physical_key(txn, *marker);
               }
               mark_txn_cursors_stale_after_write(txn, root_idx);
            }
            return removed ? MDBX_SUCCESS : MDBX_NOTFOUND;
         }
         else
         {
            // Delete ALL duplicate values for this key.
            dupsort_key_buffer prefix_buf;
            dupsort_key_buffer upper_buf;
            auto prefix = dupsort_prefix_into(root_idx, key_sv, prefix_buf);
            auto upper  = prefix ? prefix_successor_into(*prefix, upper_buf)
                                 : std::nullopt;
            bool removed = false;
            if (prefix && upper)
            {
               if (!dupsort_key_exists(txn, root_idx, key_sv))
                  return MDBX_NOTFOUND;
               removed = txn_remove_physical_range_any(txn, *prefix, *upper);
               if (removed)
                  mark_txn_cursors_stale_after_write(txn, root_idx);
            }
            return removed ? MDBX_SUCCESS : MDBX_NOTFOUND;
         }
      }

      // Non-DUPSORT
      bool removed = txn_remove_key(txn, root_idx, key_sv);
      if (removed)
         mark_txn_cursors_stale_after_write(txn, root_idx);
      return removed ? MDBX_SUCCESS : MDBX_NOTFOUND;
   }
   catch (...)
   {
      return MDBX_PANIC;
   }
}

int mdbx_replace(MDBX_txn* txn, MDBX_dbi dbi,
                 const MDBX_val* key, MDBX_val* new_data,
                 MDBX_val* old_data, MDBX_put_flags_t flags)
{
   if (!txn || !key)
      return MDBX_EINVAL;

   // Get old value first
   if (old_data)
   {
      int rc = mdbx_get(txn, dbi, key, old_data);
      if (rc != MDBX_SUCCESS && rc != MDBX_NOTFOUND)
         return rc;
   }

   // Put new value
   if (new_data)
      return mdbx_put(txn, dbi, key, new_data, flags);

   // No new data = delete
   return mdbx_del(txn, dbi, key, nullptr);
}

// ── Cursor operations ────────────────────────────────────────────

int mdbx_cursor_open(MDBX_txn* txn, MDBX_dbi dbi, MDBX_cursor** cursor)
{
   if (!txn || !cursor)
      return MDBX_EINVAL;
   if (txn_finalized(txn))
      return MDBX_BAD_TXN;

   auto meta = txn_dbi_metadata(txn, dbi);
   if (meta.root_index == UINT32_MAX)
      return MDBX_BAD_DBI;

   MDBX_cursor* c = nullptr;
   try
   {
      c = acquire_cursor(txn);
      c->txn  = txn;
      c->dbi  = dbi;
      c->root_idx = meta.root_index;
      c->is_dupsort = meta.is_dupsort;
      c->is_reverse_dup = meta.reverse_dup;

      // Ensure root is writable if RW
      if (!txn->is_readonly())
      {
         int rc = ensure_rw_root(txn, c->root_idx);
         if (rc != MDBX_SUCCESS)
         {
            release_cursor(c);
            return rc;
         }
      }

      auto mc = make_txn_cursor(txn, c->root_idx);
      c->state.emplace(std::move(mc), c->root_idx, c->is_dupsort,
                       c->is_reverse_dup);
      c->counted = true;
      ++txn->active_cursors;

      *cursor = c;
      return MDBX_SUCCESS;
   }
   catch (const std::runtime_error&)
   {
      release_cursor(c);
      return MDBX_CORRUPTED;
   }
   catch (...)
   {
      release_cursor(c);
      return MDBX_PANIC;
   }
}

void mdbx_cursor_close(MDBX_cursor* cursor)
{
   release_cursor(cursor);
}

static int mdbx_cursor_get_impl(MDBX_cursor* cursor, MDBX_val* key,
                                MDBX_val* data, MDBX_cursor_op op)
{
   if (!cursor || !cursor->state)
      return MDBX_EINVAL;
   if (!cursor->txn || txn_finalized(cursor->txn))
      return MDBX_BAD_TXN;

   auto& st = *cursor->state;
   try
   {
      if (st.stale_after_write)
      {
         auto reopen = reopen_stale_cursor_after_write(cursor, op);
         if (reopen == stale_reopen_result::notfound)
            return MDBX_NOTFOUND;
         if (reopen == stale_reopen_result::fulfilled)
         {
            assign_cursor_result(cursor, key, data);
            return MDBX_SUCCESS;
         }
      }
   }
   catch (...)
   {
      return MDBX_PANIC;
   }

   auto& mc = st.mc;

   if (st.dupsort)
   {
      try
      {
         bool ok = false;
         switch (op)
         {
            case MDBX_FIRST:
               ok = st.seek_first_dup();
               break;
            case MDBX_LAST:
               ok = st.seek_last_dup();
               break;
            case MDBX_NEXT:
               ok = st.valid ? (st.next_dup_same() || st.next_outer_first_dup())
                             : st.seek_first_dup();
               break;
            case MDBX_PREV:
               ok = st.valid ? (st.prev_dup_same() || st.prev_outer_last_dup())
                             : st.seek_last_dup();
               break;
            case MDBX_GET_CURRENT:
               if (!st.valid)
                  return MDBX_ENODATA;
               ok = true;
               break;
            case MDBX_SET:
            case MDBX_SET_KEY:
            {
               if (!key)
                  return MDBX_EINVAL;
               auto key_sv = to_sv(key);
               if (key_too_large(key_sv))
                  return MDBX_NOTFOUND;
               ok = st.seek_outer_first_dup(key_sv);
               break;
            }
            case MDBX_SET_RANGE:
            case MDBX_SET_LOWERBOUND:
            {
               if (!key)
                  return MDBX_EINVAL;
               auto key_sv = to_sv(key);
               if (key_too_large(key_sv))
                  return MDBX_NOTFOUND;
               ok = st.lower_outer_first_dup(key_sv);
               break;
            }
            case MDBX_SET_UPPERBOUND:
            {
               if (!key)
                  return MDBX_EINVAL;
               auto key_sv = to_sv(key);
               if (key_too_large(key_sv))
                  return MDBX_NOTFOUND;
               ok = st.upper_outer_first_dup(key_sv);
               break;
            }
            case MDBX_NEXT_NODUP:
               ok = st.valid ? st.next_outer_first_dup() : st.seek_first_dup();
               break;
            case MDBX_PREV_NODUP:
               ok = st.valid ? st.prev_outer_last_dup() : st.seek_last_dup();
               break;
            case MDBX_FIRST_DUP:
               if (!st.valid)
                  return MDBX_EINVAL;
               ok = st.open_first_dup_current();
               break;
            case MDBX_LAST_DUP:
               if (!st.valid)
                  return MDBX_EINVAL;
               ok = st.open_last_dup_current();
               break;
            case MDBX_NEXT_DUP:
               ok = st.valid ? st.next_dup_same() : st.seek_first_dup();
               break;
            case MDBX_PREV_DUP:
               ok = st.valid ? st.prev_dup_same() : st.seek_last_dup();
               break;
            case MDBX_GET_BOTH:
            {
               if (!key || !data)
                  return MDBX_EINVAL;
               auto key_sv = to_sv(key);
               auto val_sv = to_sv(data);
               if (key_too_large(key_sv) || key_too_large(val_sv))
                  return MDBX_NOTFOUND;
               ok = st.seek_dup_exact(key_sv, val_sv);
               break;
            }
            case MDBX_GET_BOTH_RANGE:
            {
               if (!key || !data)
                  return MDBX_EINVAL;
               auto key_sv = to_sv(key);
               auto val_sv = to_sv(data);
               if (key_too_large(key_sv) || key_too_large(val_sv))
                  return MDBX_NOTFOUND;
               ok = st.seek_dup_lower(key_sv, val_sv);
               break;
            }
            default:
               return MDBX_EINVAL;
         }

         if (!ok || !st.valid)
            return MDBX_NOTFOUND;

         assign_cursor_result(cursor, key, data);
         return MDBX_SUCCESS;
      }
      catch (...)
      {
         return MDBX_PANIC;
      }
   }

   try
   {
      bool ok = false;
      switch (op)
      {
         case MDBX_FIRST:
            st.touched = true;
            ok = st.seek_first_kv();
            break;
         case MDBX_LAST:
            st.touched = true;
            ok = st.seek_last_kv();
            break;
         case MDBX_NEXT:
            if (!st.touched)
               ok = st.seek_first_kv(); // First call after open
            else if (mc->is_end())
               ok = false;
            else
               ok = st.next_kv();
            st.touched = true;
            break;
         case MDBX_PREV:
            if (!st.touched)
               ok = st.seek_last_kv(); // First call after open
            else if (mc->is_rend())
               ok = false;
            else
               ok = st.prev_kv();
            st.touched = true;
            break;
         case MDBX_GET_CURRENT:
            if (!st.valid)
               return MDBX_ENODATA;
            assign_cursor_result(cursor, key, data);
            return MDBX_SUCCESS;
         case MDBX_SET:
         case MDBX_SET_KEY:
         {
            if (!key)
               return MDBX_EINVAL;
            auto key_sv = to_sv(key);
            if (key_too_large(key_sv))
               return MDBX_NOTFOUND;
            st.touched = true;
            ok = st.seek_kv_exact(key_sv);
            break;
         }
         case MDBX_SET_RANGE:
         case MDBX_SET_LOWERBOUND:
         {
            if (!key)
               return MDBX_EINVAL;
            auto key_sv = to_sv(key);
            if (key_too_large(key_sv))
               return MDBX_NOTFOUND;
            st.touched = true;
            ok = st.lower_bound_kv(key_sv);
            break;
         }
         case MDBX_SET_UPPERBOUND:
         {
            if (!key)
               return MDBX_EINVAL;
            auto key_sv = to_sv(key);
            if (key_too_large(key_sv))
               return MDBX_NOTFOUND;
            st.touched = true;
            ok = st.upper_bound_kv(key_sv);
            break;
         }

         case MDBX_NEXT_NODUP:
            if (!st.touched)
               ok = st.seek_first_kv();
            else if (mc->is_end())
               ok = false;
            else
               ok = st.next_kv();
            st.touched = true;
            break;

         case MDBX_PREV_NODUP:
            if (!st.touched)
               ok = st.seek_last_kv();
            else if (mc->is_rend())
               ok = false;
            else
               ok = st.prev_kv();
            st.touched = true;
            break;

         case MDBX_LAST_DUP:
         case MDBX_FIRST_DUP:
            return MDBX_INCOMPATIBLE;
         case MDBX_NEXT_DUP:
            if (!st.touched)
               ok = st.seek_first_kv();
            else if (mc->is_end())
               ok = false;
            else
               ok = st.next_kv();
            st.touched = true;
            break;
         case MDBX_PREV_DUP:
            if (!st.touched)
               ok = st.seek_last_kv();
            else if (mc->is_rend())
               ok = false;
            else
               ok = st.prev_kv();
            st.touched = true;
            break;
         case MDBX_GET_BOTH:
         case MDBX_GET_BOTH_RANGE:
            return MDBX_INCOMPATIBLE;

         default:
            return MDBX_EINVAL;
      }

      if (!ok || mc->is_end() || mc->is_rend())
      {
         st.valid = false;
         return MDBX_NOTFOUND;
      }

      st.sync_key_val();
      if (!st.valid)
         return MDBX_NOTFOUND;

      assign_cursor_result(cursor, key, data);

      return MDBX_SUCCESS;
   }
   catch (...)
   {
      return MDBX_PANIC;
   }
}

int mdbx_cursor_get(MDBX_cursor* cursor, MDBX_val* key,
                    MDBX_val* data, MDBX_cursor_op op)
{
#ifdef PSITRI_DEBUG_BUILD
   // [PSITRI_NF_LOG] capture the incoming key (lookup key for SET-family ops) and the op,
   // then log on a NOTFOUND result. The last entry before a fatal MDBX_NOTFOUND is the culprit.
   static const bool nf_log = sal::env::get("PSITRI_NF_LOG") != nullptr;
   std::string in_key;
   if (nf_log && key && key->iov_base)
      in_key.assign(static_cast<const char*>(key->iov_base), key->iov_len);
   int rc = mdbx_cursor_get_impl(cursor, key, data, op);
   if (rc == MDBX_NOTFOUND && nf_log)
   {
      char where[32];
      std::snprintf(where, sizeof where, "cursor_get op=%d", int(op));
      const uint32_t root = (cursor && cursor->state) ? cursor->root_idx : 0u;
      dbg_log_notfound(cursor ? cursor->dbi : MDBX_dbi(0), root, in_key, where);
   }
   if (sal::env::get("PSITRI_TRACE") && (op == MDBX_FIRST || op == MDBX_LAST))
      std::fprintf(stderr, "[TRACE] cursor_get op=%d dbi=%u root=%u rc=%d\n", int(op),
                   cursor ? unsigned(cursor->dbi) : 0u,
                   (cursor && cursor->state) ? cursor->root_idx : 0u, rc);
   return rc;
#else
   return mdbx_cursor_get_impl(cursor, key, data, op);
#endif
}

int mdbx_cursor_put(MDBX_cursor* cursor, const MDBX_val* key,
                    MDBX_val* data, MDBX_put_flags_t flags)
{
   if (!cursor || !cursor->txn || txn_finalized(cursor->txn) ||
       cursor->txn->is_readonly())
      return MDBX_BAD_TXN;

   int rc = mdbx_put_impl(cursor->txn, cursor->root_idx, cursor->is_dupsort,
                          cursor->is_reverse_dup, key, data, flags);
   if (rc != MDBX_SUCCESS)
      return rc;

   try
   {
      mark_cursor_stale_after_write(cursor, key, data, flags);
      return MDBX_SUCCESS;
   }
   catch (...)
   {
      return MDBX_PANIC;
   }
}

int mdbx_cursor_del(MDBX_cursor* cursor, MDBX_put_flags_t flags)
{
   if (!cursor || !cursor->state || !cursor->state->valid)
      return MDBX_EINVAL;
   if (!cursor->txn || txn_finalized(cursor->txn) || cursor->txn->is_readonly())
      return MDBX_BAD_TXN;

   auto& st = *cursor->state;

   if (st.dupsort)
   {
      if (flags & MDBX_ALLDUPS)
      {
         // Delete all duplicates for the current key
         MDBX_val key = st.key_val();
         return mdbx_del(cursor->txn, cursor->dbi, &key, nullptr);
      }
      else
      {
         MDBX_val key  = st.key_val();
         MDBX_val data = st.data_val();
         return mdbx_del(cursor->txn, cursor->dbi, &key, &data);
      }
   }

   MDBX_val key = st.key_val();
   return mdbx_del(cursor->txn, cursor->dbi, &key, nullptr);
}

int mdbx_cursor_count(const MDBX_cursor* cursor, size_t* count)
{
   if (!cursor || !count || !cursor->state)
      return MDBX_EINVAL;
   if (!cursor->txn || txn_finalized(cursor->txn))
      return MDBX_BAD_TXN;

   auto& st = *cursor->state;
   if (!st.valid)
      return MDBX_EINVAL;

   if (!st.dupsort)
   {
      *count = 1;
      return MDBX_SUCCESS;
   }

   auto root_idx = cursor->root_idx;
   if (root_idx == UINT32_MAX)
      return MDBX_BAD_DBI;

   *count = static_cast<size_t>(
      count_dups_for_key(cursor->txn, root_idx, st.key_buf));
   return MDBX_SUCCESS;
}

int mdbx_cursor_on_first(const MDBX_cursor* cursor)
{
   if (!cursor || !cursor->state || !cursor->state->valid)
      return MDBX_NOTFOUND;
   if (!cursor->txn || txn_finalized(cursor->txn))
      return MDBX_BAD_TXN;

   if (cursor->state->dupsort)
   {
      auto root_idx = cursor->root_idx;
      cursor_state tmp(make_txn_cursor(cursor->txn, root_idx), root_idx, true,
                       cursor->state->reverse_dup);
      if (!tmp.seek_first_dup())
         return MDBX_NOTFOUND;
      return (tmp.raw_key == cursor->state->raw_key)
                ? MDBX_RESULT_TRUE
                : MDBX_RESULT_FALSE;
   }

   if (cursor->state->mc.has_cursor() && cursor->state->mc->is_end())
      return MDBX_NOTFOUND;

   auto root_idx    = cursor->root_idx;
   cursor_state tmp(make_txn_cursor(cursor->txn, root_idx), root_idx, false,
                    false);
   if (tmp.seek_first_kv())
   {
      tmp.sync_key_val();
      if (tmp.valid && tmp.key_buf == cursor->state->key_buf)
         return MDBX_RESULT_TRUE;
   }
   return MDBX_RESULT_FALSE;
}

int mdbx_cursor_on_last(const MDBX_cursor* cursor)
{
   if (!cursor || !cursor->state || !cursor->state->valid)
      return MDBX_NOTFOUND;
   if (!cursor->txn || txn_finalized(cursor->txn))
      return MDBX_BAD_TXN;

   if (cursor->state->dupsort)
   {
      auto root_idx = cursor->root_idx;
      cursor_state tmp(make_txn_cursor(cursor->txn, root_idx), root_idx, true,
                       cursor->state->reverse_dup);
      if (!tmp.seek_last_dup())
         return MDBX_NOTFOUND;
      return (tmp.raw_key == cursor->state->raw_key)
                ? MDBX_RESULT_TRUE
                : MDBX_RESULT_FALSE;
   }

   if (cursor->state->mc.has_cursor() && cursor->state->mc->is_end())
      return MDBX_NOTFOUND;

   auto root_idx    = cursor->root_idx;
   cursor_state tmp(make_txn_cursor(cursor->txn, root_idx), root_idx, false,
                    false);
   if (tmp.seek_last_kv())
   {
      tmp.sync_key_val();
      if (tmp.valid && tmp.key_buf == cursor->state->key_buf)
         return MDBX_RESULT_TRUE;
   }
   return MDBX_RESULT_FALSE;
}

int mdbx_cursor_renew(MDBX_txn* txn, MDBX_cursor* cursor)
{
   if (!txn || !cursor)
      return MDBX_EINVAL;
   if (txn_finalized(txn))
      return MDBX_BAD_TXN;

   auto meta = txn_dbi_metadata(txn, cursor->dbi);
   if (meta.root_index == UINT32_MAX)
      return MDBX_BAD_DBI;

   try
   {
      if (!cursor->counted || cursor->txn != txn)
      {
         MDBX_txn* old_txn = cursor->txn;
         if (cursor->counted && cursor->txn && cursor->txn->active_cursors > 0)
            --cursor->txn->active_cursors;
         cursor->counted = true;
         ++txn->active_cursors;
         delete_or_defer_finalized_txn(old_txn);
      }
      cursor->txn = txn;
      cursor->root_idx = meta.root_index;
      cursor->is_dupsort = meta.is_dupsort;
      cursor->is_reverse_dup = meta.reverse_dup;
      auto mc = make_txn_cursor(txn, cursor->root_idx);
      cursor->state.reset();
      cursor->state.emplace(std::move(mc), cursor->root_idx, cursor->is_dupsort,
                            cursor->is_reverse_dup);
      return MDBX_SUCCESS;
   }
   catch (...)
   {
      return MDBX_PANIC;
   }
}

MDBX_dbi mdbx_cursor_dbi(const MDBX_cursor* cursor)
{
   return cursor ? cursor->dbi : 0;
}

MDBX_txn* mdbx_cursor_txn(const MDBX_cursor* cursor)
{
   return cursor ? cursor->txn : nullptr;
}

MDBX_cursor* mdbx_cursor_create(void* context)
{
   auto* cursor = new MDBX_cursor();
   cursor->heap_owned = true;
   cursor->open = true;
   cursor->context = context;
   return cursor;
}

int mdbx_cursor_copy(const MDBX_cursor* src, MDBX_cursor* dest)
{
   if (!src || !dest || !src->txn || !src->state)
      return MDBX_EINVAL;

   dest->txn = src->txn;
   dest->dbi = src->dbi;
   dest->is_dupsort = src->is_dupsort;
   dest->is_reverse_dup = src->is_reverse_dup;
   dest->open = true;

   int rc = mdbx_cursor_renew(src->txn, dest);
   if (rc != MDBX_SUCCESS)
      return rc;

   if (src->state->valid)
   {
      auto& src_state = *src->state;
      auto& dst_state = *dest->state;
      if (src->is_dupsort)
      {
	         if (dst_state.mc->find(src_state.raw_key))
	            dst_state.sync_dup_key_val();
      }
      else
      {
         bool ok = dst_state.seek_kv_exact(src_state.key_buf);
         if (ok)
            dst_state.sync_key_val();
      }
   }

   return MDBX_SUCCESS;
}

// ════════════════════════════════════════════════════════════════════
// C++ API implementation
// ════════════════════════════════════════════════════════════════════

namespace mdbx
{
   // ── env methods ───────────────────────────────────────────────────

   MDBX_env_flags_t env::operate_parameters::make_flags() const noexcept
   {
      unsigned f = MDBX_ENV_DEFAULTS;
      if (mode == env::readonly)
         f |= MDBX_RDONLY;
      if (mode == env::write_mapped_io)
         f |= MDBX_WRITEMAP;
      switch (durability)
      {
         case env::robust_synchronous:       break;
         case env::half_synchronous_weak_last: f |= MDBX_NOMETASYNC; break;
         case env::lazy_weak_tail:           f |= MDBX_SAFE_NOSYNC; break;
         case env::whole_fragile:            f |= MDBX_UTTERLY_NOSYNC; break;
      }
      return static_cast<MDBX_env_flags_t>(f);
   }

   env& env::set_geometry(const geometry& geo)
   {
      error::success_or_throw(
         mdbx_env_set_geometry(handle_, geo.size_lower, geo.size_now,
                               geo.size_upper, geo.growth_step,
                               geo.shrink_threshold, geo.pagesize));
      return *this;
   }

   unsigned env::max_maps() const
   {
      return handle_ ? static_cast<const MDBX_env*>(handle_)->max_dbs : 0;
   }

   unsigned env::max_readers() const
   {
      return handle_ ? static_cast<const MDBX_env*>(handle_)->max_readers : 0;
   }

   env& env::set_context(void* ctx)
   {
      error::success_or_throw(mdbx_env_set_userctx(handle_, ctx));
      return *this;
   }

   bool env::sync_to_disk(bool force, bool nonblock)
   {
      return mdbx_env_sync_ex(handle_, force ? 1 : 0, nonblock ? 1 : 0) == MDBX_SUCCESS;
   }

   MDBX_stat env::get_stat() const
   {
      MDBX_stat stat{};
      error::success_or_throw(
         mdbx_env_stat_ex(handle_, nullptr, &stat, sizeof(stat)));
      return stat;
   }

   MDBX_envinfo env::get_info() const
   {
      MDBX_envinfo info{};
      error::success_or_throw(
         mdbx_env_info_ex(handle_, nullptr, &info, sizeof(info)));
      return info;
   }

   size_t env::get_pagesize() const
   {
      if (!handle_)
         error::success_or_throw(MDBX_EINVAL);
      return static_cast<const MDBX_env*>(handle_)->page_size;
   }

   void env::copy(const char* dest, bool compactify, bool /*force_dynamic*/)
   {
      error::success_or_throw(
         mdbx_env_copy(handle_, dest, compactify ? 1u : 0u));
   }

   std::filesystem::path env::get_path() const
   {
      if (!handle_)
         error::success_or_throw(MDBX_EINVAL);
      return static_cast<const MDBX_env*>(handle_)->path;
   }

   void env::close_map(const map_handle& map)
   {
      mdbx_dbi_close(handle_, map.dbi);
   }

   txn_managed env::start_read() const
   {
      MDBX_txn* t = nullptr;
      error::success_or_throw(
         mdbx_txn_begin(const_cast<MDBX_env*>(handle_), nullptr, MDBX_TXN_RDONLY, &t));
      return txn_managed(t);
   }

   txn_managed env::start_write(bool dont_wait)
   {
      MDBX_txn* t     = nullptr;
      auto      flags = dont_wait
                           ? static_cast<MDBX_txn_flags_t>(MDBX_TXN_READWRITE | MDBX_TXN_TRY)
                           : MDBX_TXN_READWRITE;
      error::success_or_throw(mdbx_txn_begin(handle_, nullptr, flags, &t));
      return txn_managed(t);
   }

   // ── env_managed ───────────────────────────────────────────────────

   env_managed::env_managed(const char* pathname,
                            const create_parameters& cp,
                            const operate_parameters& op,
                            bool /*accede*/)
   {
      MDBX_env* e = nullptr;
      error::success_or_throw(mdbx_env_create(&e));
      handle_ = e;

      if (op.max_maps)
         error::success_or_throw(mdbx_env_set_maxdbs(e, op.max_maps));
      if (op.max_readers)
         error::success_or_throw(mdbx_env_set_maxreaders(e, op.max_readers));
      if (op.psitri_cache_size_mb)
         error::success_or_throw(mdbx_env_set_option(
            e, MDBX_opt_psitri_cache_size_mb, *op.psitri_cache_size_mb));
      if (op.psitri_cache_window_sec)
         error::success_or_throw(mdbx_env_set_option(
            e, MDBX_opt_psitri_cache_window_sec, *op.psitri_cache_window_sec));
      if (op.psitri_write_mode)
         error::success_or_throw(mdbx_env_set_write_mode(e, *op.psitri_write_mode));

      error::success_or_throw(mdbx_env_set_geometry(
         e, cp.geometry.size_lower, cp.geometry.size_now,
         cp.geometry.size_upper, cp.geometry.growth_step,
         cp.geometry.shrink_threshold, cp.geometry.pagesize));

      auto flags = op.make_flags();
      if (!cp.use_subdirectory)
         flags = static_cast<MDBX_env_flags_t>(flags | MDBX_NOSUBDIR);

      error::success_or_throw(mdbx_env_open(e, pathname, flags, cp.file_mode_bits));
   }

   env_managed::~env_managed() noexcept
   {
      if (handle_)
      {
         mdbx_env_close(handle_);
         handle_ = nullptr;
      }
   }

   void env_managed::close(bool dont_sync)
   {
      if (handle_)
      {
         mdbx_env_close_ex(handle_, dont_sync ? 1 : 0);
         handle_ = nullptr;
      }
   }

   env_managed& env_managed::operator=(env_managed&& o) noexcept
   {
      if (handle_)
         mdbx_env_close(handle_);
      handle_   = o.handle_;
      o.handle_ = nullptr;
      return *this;
   }

   // ── txn methods ───────────────────────────────────────────────────

   ::mdbx::env txn::env() const noexcept
   {
      auto* e = mdbx_txn_env(handle_);
      return ::mdbx::env(reinterpret_cast<MDBX_env*>(e));
   }

   void* txn::get_context() const noexcept
   {
      return handle_ ? handle_->context : nullptr;
   }

   txn& txn::set_context(void* ctx)
   {
      if (handle_)
         handle_->context = ctx;
      return *this;
   }

   map_handle txn::open_map(const char* name, key_mode /*km*/, value_mode vm) const
   {
      MDBX_dbi   dbi   = 0;
      unsigned   flags = 0;
      if (vm != value_mode::single)
         flags |= static_cast<unsigned>(vm);
      int rc = mdbx_dbi_open(const_cast<MDBX_txn*>(handle_), name,
                             static_cast<MDBX_db_flags_t>(flags), &dbi);
      if (rc == MDBX_NOTFOUND)
         throw not_found();
      error::success_or_throw(rc);
      return map_handle(dbi);
   }

   map_handle txn::create_map(const char* name, key_mode /*km*/, value_mode vm)
   {
      MDBX_dbi   dbi   = 0;
      unsigned   flags = MDBX_CREATE;
      if (vm != value_mode::single)
         flags |= static_cast<unsigned>(vm);
      error::success_or_throw(
         mdbx_dbi_open(handle_, name, static_cast<MDBX_db_flags_t>(flags), &dbi));
      return map_handle(dbi);
   }

   void txn::drop_map(map_handle map)
   {
      error::success_or_throw(mdbx_drop(handle_, map.dbi, 1));
   }

   void txn::clear_map(map_handle map)
   {
      error::success_or_throw(mdbx_drop(handle_, map.dbi, 0));
   }

   bool txn::drop_map(const std::string& name, bool throw_if_absent)
   {
      MDBX_dbi dbi = 0;
      int rc = mdbx_dbi_open(handle_, name.c_str(), MDBX_DB_DEFAULTS, &dbi);
      if (rc == MDBX_NOTFOUND)
      {
         if (throw_if_absent)
            throw not_found();
         return false;
      }
      error::success_or_throw(rc);
      error::success_or_throw(mdbx_drop(handle_, dbi, 1));
      return true;
   }

   bool txn::clear_map(const std::string& name, bool throw_if_absent)
   {
      MDBX_dbi dbi = 0;
      int rc = mdbx_dbi_open(handle_, name.c_str(), MDBX_DB_DEFAULTS, &dbi);
      if (rc == MDBX_NOTFOUND)
      {
         if (throw_if_absent)
            throw not_found();
         return false;
      }
      error::success_or_throw(rc);
      error::success_or_throw(mdbx_drop(handle_, dbi, 0));
      return true;
   }

   cursor_managed txn::open_cursor(map_handle map) const
   {
      MDBX_cursor* c = nullptr;
      error::success_or_throw(
         mdbx_cursor_open(const_cast<MDBX_txn*>(handle_), map.dbi, &c));
      return cursor_managed(c);
   }

   slice txn::get(map_handle map, const slice& key) const
   {
      MDBX_val k = key;
      MDBX_val v;
      int rc = mdbx_get(handle_, map.dbi, &k, &v);
      if (rc == MDBX_NOTFOUND)
         throw not_found();
      error::success_or_throw(rc);
      return slice(v);
   }

   slice txn::get(map_handle map, const slice& key,
                  const slice& value_at_absence) const
   {
      MDBX_val k = key;
      MDBX_val v;
      int rc = mdbx_get(handle_, map.dbi, &k, &v);
      if (rc == MDBX_NOTFOUND)
         return value_at_absence;
      error::success_or_throw(rc);
      return slice(v);
   }

   void txn::upsert(map_handle map, const slice& key, const slice& value)
   {
      MDBX_val k = key;
      MDBX_val v = value;
      error::success_or_throw(mdbx_put(handle_, map.dbi, &k, &v, MDBX_UPSERT));
   }

   void txn::insert(map_handle map, const slice& key, slice value)
   {
      MDBX_val k = key;
      MDBX_val v = value;
      error::success_or_throw(mdbx_put(handle_, map.dbi, &k, &v, MDBX_NOOVERWRITE));
   }

   value_result txn::try_insert(map_handle map, const slice& key, slice value)
   {
      MDBX_val k = key;
      MDBX_val v = value;
      int rc = mdbx_put(handle_, map.dbi, &k, &v, MDBX_NOOVERWRITE);
      if (rc == MDBX_KEYEXIST)
         return {slice(v), false};
      error::success_or_throw(rc);
      return {value, true};
   }

   void txn::update(map_handle map, const slice& key, const slice& value)
   {
      MDBX_val k = key;
      MDBX_val v = value;
      error::success_or_throw(mdbx_put(handle_, map.dbi, &k, &v, MDBX_CURRENT));
   }

   bool txn::try_update(map_handle map, const slice& key, const slice& value)
   {
      MDBX_val k = key;
      MDBX_val v = value;
      int rc = mdbx_put(handle_, map.dbi, &k, &v, MDBX_CURRENT);
      if (rc == MDBX_NOTFOUND)
         return false;
      error::success_or_throw(rc);
      return true;
   }

   MDBX_error_t txn::put(map_handle map, const slice& key, slice* value,
                          MDBX_put_flags_t flags) noexcept
   {
      MDBX_val k = key;
      MDBX_val v = *value;
      int rc = mdbx_put(handle_, map.dbi, &k, &v, flags);
      if (rc == MDBX_SUCCESS || rc == MDBX_KEYEXIST)
         *value = slice(v);
      return static_cast<MDBX_error_t>(rc);
   }

   bool txn::erase(map_handle map, const slice& key)
   {
      MDBX_val k = key;
      int rc = mdbx_del(handle_, map.dbi, &k, nullptr);
      if (rc == MDBX_NOTFOUND)
         return false;
      error::success_or_throw(rc);
      return true;
   }

   bool txn::erase(map_handle map, const slice& key, const slice& value)
   {
      MDBX_val k = key;
      MDBX_val v = value;
      int rc = mdbx_del(handle_, map.dbi, &k, &v);
      if (rc == MDBX_NOTFOUND)
         return false;
      error::success_or_throw(rc);
      return true;
   }

   MDBX_stat txn::get_map_stat(map_handle map) const
   {
      MDBX_stat stat{};
      error::success_or_throw(
         mdbx_dbi_stat(handle_, map.dbi, &stat, sizeof(stat)));
      return stat;
   }

   map_handle::info txn::get_handle_info(map_handle map) const
   {
      map_handle::info info{};
      info.dbi = map.dbi;
      unsigned raw_flags = 0;
      error::success_or_throw(
         mdbx_dbi_flags_ex(handle_, map.dbi, &raw_flags, &info.state));
      info.flags = static_cast<MDBX_db_flags_t>(raw_flags);
      return info;
   }

   // ── txn_managed ───────────────────────────────────────────────────

   txn_managed::~txn_managed() noexcept
   {
      if (handle_)
      {
         mdbx_txn_abort(handle_);
         handle_ = nullptr;
      }
   }

   void txn_managed::abort()
   {
      if (handle_)
      {
         error::success_or_throw(mdbx_txn_abort(handle_));
         handle_ = nullptr;
      }
   }

   void txn_managed::commit()
   {
      if (handle_)
      {
         error::success_or_throw(mdbx_txn_commit(handle_));
         handle_ = nullptr;
      }
   }

   txn_managed& txn_managed::operator=(txn_managed&& o) noexcept
   {
      if (handle_)
         mdbx_txn_abort(handle_);
      handle_   = o.handle_;
      o.handle_ = nullptr;
      return *this;
   }

   // ── cursor methods ────────────────────────────────────────────────

   cursor::move_result cursor::do_get(MDBX_cursor_op op, MDBX_val* key,
                                      MDBX_val* data, bool throw_notfound) const
   {
      MDBX_val k{}, d{};
      if (key)
         k = *key;
      if (data)
         d = *data;
      int rc = mdbx_cursor_get(const_cast<MDBX_cursor*>(handle_), &k, &d, op);
      if (rc == MDBX_NOTFOUND)
      {
         if (throw_notfound)
            throw not_found();
         return move_result({}, {}, false);
      }
      error::success_or_throw(rc);
      return move_result(slice(k), slice(d), true);
   }

   cursor::move_result cursor::to_first(bool throw_notfound)
   {
      return do_get(MDBX_FIRST, nullptr, nullptr, throw_notfound);
   }

   cursor::move_result cursor::to_last(bool throw_notfound)
   {
      return do_get(MDBX_LAST, nullptr, nullptr, throw_notfound);
   }

   cursor::move_result cursor::to_next(bool throw_notfound)
   {
      return do_get(MDBX_NEXT, nullptr, nullptr, throw_notfound);
   }

   cursor::move_result cursor::to_previous(bool throw_notfound)
   {
      return do_get(MDBX_PREV, nullptr, nullptr, throw_notfound);
   }

   cursor::move_result cursor::current(bool throw_notfound) const
   {
      return do_get(MDBX_GET_CURRENT, nullptr, nullptr, throw_notfound);
   }

   cursor::move_result cursor::to_key_equal(const slice& key, bool throw_notfound)
   {
      MDBX_val k = key;
      return do_get(MDBX_SET_KEY, &k, nullptr, throw_notfound);
   }

   bool cursor::seek(const slice& key)
   {
      MDBX_val k = key;
      MDBX_val d;
      int rc = mdbx_cursor_get(handle_, &k, &d, MDBX_SET_KEY);
      return rc == MDBX_SUCCESS;
   }

   cursor::move_result cursor::find(const slice& key, bool throw_notfound)
   {
      MDBX_val k = key;
      return do_get(MDBX_SET_KEY, &k, nullptr, throw_notfound);
   }

   cursor::move_result cursor::lower_bound(const slice& key, bool throw_notfound)
   {
      MDBX_val k = key;
      return do_get(MDBX_SET_RANGE, &k, nullptr, throw_notfound);
   }

   cursor::move_result cursor::upper_bound(const slice& key, bool throw_notfound)
   {
      MDBX_val k = key;
      return do_get(MDBX_SET_UPPERBOUND, &k, nullptr, throw_notfound);
   }

   cursor::move_result cursor::move(move_operation op, MDBX_val* key,
                                    MDBX_val* value, bool throw_notfound)
   {
      return do_get(op, key, value, throw_notfound);
   }

   cursor::move_result cursor::move(move_operation op, bool throw_notfound)
   {
      return do_get(op, nullptr, nullptr, throw_notfound);
   }

   cursor::move_result cursor::move(move_operation op, const slice& key,
                                    bool throw_notfound)
   {
      MDBX_val k = key;
      return do_get(op, &k, nullptr, throw_notfound);
   }

   cursor::move_result cursor::move(move_operation op, const slice& key,
                                    const slice& value, bool throw_notfound)
   {
      MDBX_val k = key;
      MDBX_val v = value;
      return do_get(op, &k, &v, throw_notfound);
   }

   // DUPSORT multi-value stubs
   cursor::move_result cursor::to_current_first_multi(bool throw_notfound)
   {
      return do_get(MDBX_FIRST_DUP, nullptr, nullptr, throw_notfound);
   }

   cursor::move_result cursor::to_current_last_multi(bool throw_notfound)
   {
      return do_get(MDBX_LAST_DUP, nullptr, nullptr, throw_notfound);
   }

   cursor::move_result cursor::to_current_prev_multi(bool throw_notfound)
   {
      if (!handle_ || !handle_->is_dupsort)
         return do_get(MDBX_PREV, nullptr, nullptr, throw_notfound);
      return do_get(MDBX_PREV_DUP, nullptr, nullptr, throw_notfound);
   }

   cursor::move_result cursor::to_current_next_multi(bool throw_notfound)
   {
      if (!handle_ || !handle_->is_dupsort)
         return do_get(MDBX_NEXT, nullptr, nullptr, throw_notfound);
      return do_get(MDBX_NEXT_DUP, nullptr, nullptr, throw_notfound);
   }

   cursor::move_result cursor::to_next_dup(bool throw_notfound)
   {
      return do_get(MDBX_NEXT_DUP, nullptr, nullptr, throw_notfound);
   }

   cursor::move_result cursor::to_prev_dup(bool throw_notfound)
   {
      return do_get(MDBX_PREV_DUP, nullptr, nullptr, throw_notfound);
   }

   cursor::move_result cursor::to_next_nodup(bool throw_notfound)
   {
      return do_get(MDBX_NEXT_NODUP, nullptr, nullptr, throw_notfound);
   }

   cursor::move_result cursor::to_prev_nodup(bool throw_notfound)
   {
      return do_get(MDBX_PREV_NODUP, nullptr, nullptr, throw_notfound);
   }

   cursor::move_result cursor::to_previous_last_multi(bool throw_notfound)
   {
      return do_get(MDBX_PREV_NODUP, nullptr, nullptr, throw_notfound);
   }

   cursor::move_result cursor::to_next_first_multi(bool throw_notfound)
   {
      return do_get(MDBX_NEXT_NODUP, nullptr, nullptr, throw_notfound);
   }

   cursor::move_result cursor::find_multivalue(const slice& key, const slice& value,
                                                bool throw_notfound)
   {
      MDBX_val k = key;
      MDBX_val v = value;
      return do_get(MDBX_GET_BOTH, &k, &v, throw_notfound);
   }

   cursor::move_result cursor::lower_bound_multivalue(const slice& key, const slice& value,
                                                       bool throw_notfound)
   {
      MDBX_val k = key;
      MDBX_val v = value;
      return do_get(MDBX_GET_BOTH_RANGE, &k, &v, throw_notfound);
   }

   size_t cursor::count_multivalue() const
   {
      size_t cnt = 0;
      mdbx_cursor_count(handle_, &cnt);
      return cnt;
   }

   bool cursor::eof() const
   {
      if (!handle_ || !handle_->state)
         return true;
      if (!handle_->state->valid)
         return true;
      if (handle_->state->stale_after_write || !handle_->state->mc.has_cursor())
         return false;
      return handle_->state->mc->is_end() || handle_->state->mc->is_rend();
   }

   bool cursor::on_first() const
   {
      return mdbx_cursor_on_first(handle_) == MDBX_RESULT_TRUE;
   }

   bool cursor::on_last() const
   {
      return mdbx_cursor_on_last(handle_) == MDBX_RESULT_TRUE;
   }

   void cursor::upsert(const slice& key, const slice& value)
   {
      MDBX_val k = key;
      MDBX_val v = value;
      error::success_or_throw(mdbx_cursor_put(handle_, &k, &v, MDBX_UPSERT));
   }

   void cursor::append(const slice& key, const slice& value)
   {
      MDBX_val k = key;
      MDBX_val v = value;
      error::success_or_throw(mdbx_cursor_put(handle_, &k, &v, MDBX_APPEND));
   }

   MDBX_error_t cursor::put(const slice& key, slice* value,
                            MDBX_put_flags_t flags) noexcept
   {
      MDBX_val k = key;
      MDBX_val v = value ? static_cast<MDBX_val>(*value) : MDBX_val{nullptr, 0};
      int rc = mdbx_cursor_put(handle_, &k, &v, flags);
      if (value && (rc == MDBX_SUCCESS || rc == MDBX_KEYEXIST))
         *value = slice(v);
      return static_cast<MDBX_error_t>(rc);
   }

   void cursor::insert(const slice& key, slice value)
   {
      MDBX_val k = key;
      MDBX_val v = value;
      error::success_or_throw(mdbx_cursor_put(handle_, &k, &v, MDBX_NOOVERWRITE));
   }

   value_result cursor::try_insert(const slice& key, slice value)
   {
      MDBX_val k = key;
      MDBX_val v = value;
      int rc = mdbx_cursor_put(handle_, &k, &v, MDBX_NOOVERWRITE);
      if (rc == MDBX_KEYEXIST)
         return {slice(v), false};
      error::success_or_throw(rc);
      return {value, true};
   }

   void cursor::update(const slice& key, const slice& value)
   {
      MDBX_val k = key;
      MDBX_val v = value;
      error::success_or_throw(mdbx_cursor_put(handle_, &k, &v, MDBX_CURRENT));
   }

   bool cursor::try_update(const slice& key, const slice& value)
   {
      MDBX_val k = key;
      MDBX_val v = value;
      int rc = mdbx_cursor_put(handle_, &k, &v, MDBX_CURRENT);
      if (rc == MDBX_NOTFOUND)
         return false;
      error::success_or_throw(rc);
      return true;
   }

   bool cursor::erase(bool whole_multivalue)
   {
      auto flags = whole_multivalue ? MDBX_ALLDUPS : MDBX_UPSERT;
      int rc = mdbx_cursor_del(handle_, static_cast<MDBX_put_flags_t>(flags));
      if (rc == MDBX_NOTFOUND)
         return false;
      error::success_or_throw(rc);
      return true;
   }

   bool cursor::erase(const slice& key, bool whole_multivalue)
   {
      MDBX_val k = key;
      if (whole_multivalue)
      {
         int rc = mdbx_del(mdbx_cursor_txn(handle_), mdbx_cursor_dbi(handle_),
                           &k, nullptr);
         if (rc == MDBX_NOTFOUND)
            return false;
         error::success_or_throw(rc);
         return true;
      }

      MDBX_val v{};
      int rc = mdbx_cursor_get(handle_, &k, &v, MDBX_SET);
      if (rc == MDBX_NOTFOUND)
         return false;
      error::success_or_throw(rc);
      rc = mdbx_del(mdbx_cursor_txn(handle_), mdbx_cursor_dbi(handle_), &k, &v);
      if (rc == MDBX_NOTFOUND)
         return false;
      error::success_or_throw(rc);
      return true;
   }

   bool cursor::erase(const slice& key, const slice& value)
   {
      MDBX_val k = key;
      MDBX_val v = value;
      int rc = mdbx_del(mdbx_cursor_txn(handle_), mdbx_cursor_dbi(handle_), &k, &v);
      if (rc == MDBX_NOTFOUND)
         return false;
      error::success_or_throw(rc);
      return true;
   }

   void cursor::renew(::mdbx::txn& t)
   {
      error::success_or_throw(mdbx_cursor_renew(t, handle_));
   }

   void cursor::bind(::mdbx::txn& t, map_handle map)
   {
      // Rebind = close + reopen
      mdbx_cursor_close(handle_);
      handle_ = nullptr;
      MDBX_cursor* c = nullptr;
      error::success_or_throw(mdbx_cursor_open(t, map.dbi, &c));
      handle_ = c;
   }

   ::mdbx::txn cursor::txn() const
   {
      return ::mdbx::txn(mdbx_cursor_txn(handle_));
   }

   map_handle cursor::map() const
   {
      return map_handle(mdbx_cursor_dbi(handle_));
   }

   // ── cursor_managed ────────────────────────────────────────────────

   cursor_managed::~cursor_managed() noexcept
   {
      if (handle_)
      {
         mdbx_cursor_close(handle_);
         handle_ = nullptr;
      }
   }

   void cursor_managed::close()
   {
      if (handle_)
      {
         mdbx_cursor_close(handle_);
         handle_ = nullptr;
      }
   }

   cursor_managed& cursor_managed::operator=(cursor_managed&& o) noexcept
   {
      if (handle_)
         mdbx_cursor_close(handle_);
      handle_   = o.handle_;
      o.handle_ = nullptr;
      return *this;
   }

}  // namespace mdbx
