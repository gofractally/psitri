// 48-bit handle impact analysis: instantiate part's hkv tree with a 6-byte
// indexed line handle (4-bit kind tag + 44-bit line index = 2 PB @ 128B lines)
// next to the production 4-byte one (28-bit index = 32 GB), and drive BOTH
// through the same load/read/iterate harness. hkv is templated on the handle
// via Allocator::hkv_handle, so this needs zero changes to part itself.
//
//   part-ptr48-bench [keys=20e6] [chunk=1e6]
//
// Reports per width: insert/find/iter ns, arena footprint, and a node-kind
// census (count/lines/bytes) so the router-density cost of +2B handles is
// measured, not estimated.
//
// PREREQUISITE for the 48-bit side: cb_hdr exactly fills its 128-byte line at
// 4-byte handles (no sizeof in the alloc, no static_assert) — at 6 bytes it is
// 164 B and every header write tramples the neighboring line. The 6-bit side
// only runs against a tree.hpp whose cbucket CL0 is width-adaptive:
//   page_line[sizeof(Ptr) > 4 ? 8 : 12], computed prefix cap,
//   static_assert(sizeof(cb_hdr) == 128), CB_PAGES to match, and the two
//   hardcoded 36-byte prefix caps (cb_make2_ memcpy + the c <= 36 formation
//   gate) switched to sizeof(prefix). Byte-identical at 4 B.
#include <part/kv.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>

using Clock = std::chrono::steady_clock;
using part::kv::pval;
using K = hkv::detail::K;

static uint64_t splitmix64(uint64_t x)
{
   x += 0x9E3779B97F4A7C15ull;
   x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
   x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
   return x ^ (x >> 31);
}
static std::string_view k8(const uint64_t& v)
{
   return {reinterpret_cast<const char*>(&v), 8};
}
static double ns_per(Clock::time_point a, Clock::time_point b, uint64_t n)
{
   return double(std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count()) /
          double(n ? n : 1);
}
static const char* kname(K k)
{
   switch (k)
   {
      case K::value_ptr: return "leaf";
      case K::value_inline: return "inline";
      case K::prefix_node: return "prefix";
      case K::setlist_u8: return "setlist8";
      case K::setlist_u16: return "setlist16";
      case K::node_full: return "full";
      case K::full_x: return "full_x";
      case K::c2: return "c2";
      case K::c4: return "c4";
      case K::c8: return "c8";
      case K::cx: return "cx";
      case K::bucket: return "bucket";
      case K::cbucket: return "cbucket";
      default: return "?";
   }
}

// ── 6-byte indexed handle: same shape contract as part::detail::line_ptr ────
struct line_ptr6
{
   static constexpr bool indexed = true;
   uint8_t               b[6];

   static line_ptr6 null() noexcept
   {
      line_ptr6 p;
      std::memset(p.b, 0xFF, 6);  // tag nibble becomes 0xF == K::null
      return p;
   }
   static line_ptr6 from_off(size_t byte_off, K k) noexcept
   {
      assert((byte_off & 0x7f) == 0 && "part: node offsets are line-aligned");
      assert(byte_off >> 51 == 0 && "part: offset exceeds the 44-bit line index");
      line_ptr6      p;
      const uint64_t raw = (uint64_t(byte_off) >> 3) | uint8_t(k);
      std::memcpy(p.b, &raw, 6);
      return p;
   }
   uint64_t raw() const noexcept
   {
      uint64_t v = 0;
      std::memcpy(&v, b, 6);
      return v;
   }
   K              tag() const noexcept { return K(b[0] & 0xF); }
   bool           is_null() const noexcept { return tag() == K::null; }
   part::line_idx index() const noexcept { return part::line_idx(raw() >> 4); }
};
static_assert(sizeof(line_ptr6) == 6);

// part_alloc with the wide handle (verbatim copy otherwise)
template <class T>
struct part_alloc6
{
   using value_type = T;
   using hkv_handle = line_ptr6;

   part::arena*   a   = nullptr;
   part::session* ses = nullptr;

   part_alloc6() = default;
   explicit part_alloc6(part::session* s) noexcept : a(&s->store()), ses(s) {}
   explicit part_alloc6(part::arena* ar) noexcept : a(ar) {}
   template <class U>
   part_alloc6(const part_alloc6<U>& o) noexcept : a(o.a), ses(o.ses)
   {
   }

   const std::byte* hkv_base() const noexcept { return a ? a->base() : nullptr; }

   static constexpr size_t lines_(size_t bytes) noexcept
   {
      return (bytes + part::line_bytes - 1) / part::line_bytes;
   }
   T* allocate(size_t n)
   {
      assert(ses && "read-only view allocators cannot allocate");
      const part::line_idx i = ses->alloc_lines(uint32_t(lines_(n * sizeof(T))));
      return static_cast<T*>(a->get(i));
   }
   void deallocate(T* p, size_t n) noexcept
   {
      const part::line_idx i =
          part::line_idx((reinterpret_cast<const std::byte*>(p) - a->base()) / part::line_bytes);
      if (ses)
         ses->release_run(i, uint32_t(lines_(n * sizeof(T))));
      else
         a->release_run(i, uint32_t(lines_(n * sizeof(T))));
   }
   template <class U>
   bool operator==(const part_alloc6<U>& o) const noexcept
   {
      return a == o.a && ses == o.ses;
   }

   static constexpr bool hkv_cow = true;
   uint8_t hkv_ref(uint32_t line) const noexcept { return a->ref(line); }
   bool    hkv_retain(uint32_t line) noexcept { return a->retain(line); }
   template <class D>
   bool hkv_release(uint32_t line, D&& d) noexcept(noexcept(d(line)))
   {
      return a->release(line, static_cast<D&&>(d));
   }
   void hkv_free_run(uint32_t line, uint32_t n) noexcept { a->release_run(line, n); }
};

// ── one full measurement pass, templated over the allocator (= handle width) ──
template <template <class> class Alloc>
static void run_side(const char* name, const char* dirs, uint64_t total, uint64_t chunk)
{
   using engine = hkv::kv_tree<std::string_view, pval, hkv::mode::none, Alloc<pval>>;
   using root_t = std::decay_t<decltype(std::declval<engine&>().root_handle())>;

   std::filesystem::path dir(dirs);
   std::filesystem::remove_all(dir);
   part::arena_config ac;
   ac.dir             = dir;
   ac.sync_mode       = sal::sync_type::none;
   ac.reuse_threshold = 0.125;  // match part::kv
   part::arena ar(ac);
   auto        ses = ar.start_session();
   engine      t{Alloc<pval>(&ses)};
   root_t      live = root_t::null();

   printf("\n== %s (%zu-byte handle) ==\n", name, sizeof(root_t));
   printf("%10s %9s %9s %7s %9s %9s %8s\n", "keys", "ins ns", "rd ns", "hit%", "iter ns",
          "file GB", "free%");

   std::mt19937_64 rng(0xC0FFEE);
   uint64_t        done = 0;
   while (done < total)
   {
      const uint64_t end = std::min(total, done + chunk);
      auto           t0  = Clock::now();
      {
         auto ptx = ses.start_transaction(0);
         for (uint64_t i = done; i < end; ++i)
         {
            const uint64_t k = splitmix64(i);
            const uint64_t v = splitmix64(i ^ 0xDEADBEEFCAFEBABEull);
            t.cow_insert(k8(k), pval::from(std::string_view(k8(v))));
         }
         ptx.commit();
      }
      // publish the new root, retire the old (the kv commit essentials)
      const root_t nr = t.root_handle();
      if (nr.raw() != live.raw())
      {
         if (!nr.is_null() && !t.cow_retain(nr))
            throw std::runtime_error("root retain refused");
         if (!live.is_null())
            t.cow_release(live);
         live = nr;
      }
      auto           t1    = Clock::now();
      const uint64_t n_ins = end - done;
      done                 = end;

      // reads + iteration against a snapshot view (the kv cursor shape)
      uint64_t found = 0, stepped = 0, ksum = 0;
      auto     t2 = Clock::now(), t3 = t2, t4 = t2, t5 = t2;
      {
         engine view{Alloc<pval>(&ar)};
         view.adopt_root(live);
         t2 = Clock::now();
         for (uint64_t r = 0; r < 100'000; ++r)
         {
            const uint64_t k = splitmix64(rng() % done);
            pval           out;
            if (view.find(k8(k), out))
               ++found;
         }
         t3                = Clock::now();
         const uint64_t k0 = splitmix64(rng() % done);
         auto           it = view.lower_bound(k8(k0));
         t4                = Clock::now();
         for (uint64_t s = 0; s < 100'000; ++s)
         {
            if (it == view.end())
               it = view.begin();
            ksum += uint8_t((*it).first[0]);
            ++it;
            ++stepped;
         }
         t5 = Clock::now();
         view.adopt_root(root_t::null());  // the view owns nothing
      }
      (void)ksum;

      uint64_t free_ln = 0;
      for (uint32_t sg = 0; sg < ar.num_segments(); ++sg)
         free_ln += ar.freeset_of(sg).free_lines();
      const double file_gb =
          double(ar.num_segments()) * part::segment_bytes / (1024.0 * 1024 * 1024);
      const double freep = 100.0 * double(free_ln) /
                           (double(ar.num_segments()) * part::lines_per_segment);
      printf("%9.0fM %9.1f %9.1f %6.1f%% %9.1f %9.2f %7.1f%%\n", double(done) / 1e6,
             ns_per(t0, t1, n_ins), ns_per(t2, t3, 100'000),
             100.0 * double(found) / 100'000.0, ns_per(t4, t5, stepped), file_gb, freep);
      fflush(stdout);
   }

   // node-kind census on the final tree
   struct agg
   {
      uint64_t n = 0, bytes = 0, nbr = 0, lu = 0, la = 0;
   };
   std::map<K, agg> by;
   {
      engine view{Alloc<pval>(&ar)};
      view.adopt_root(live);
      view.census(
          [&](typename engine::census_row r)
          {
             auto& a = by[r.kind];
             ++a.n;
             a.bytes += r.bytes;
             a.nbr += r.nbranch;
             a.lu += r.lines_used;
             a.la += r.lines_alloc;
          });
      view.adopt_root(root_t::null());
   }
   printf("  %-10s %10s %12s %9s %8s %10s\n", "kind", "count", "MB", "avg nbr", "lines u",
          "lines a");
   uint64_t tot_bytes = 0;
   for (auto& [k, a] : by)
   {
      printf("  %-10s %10llu %12.1f %9.1f %8llu %10llu\n", kname(k), (unsigned long long)a.n,
             double(a.bytes) / 1048576.0, a.n ? double(a.nbr) / double(a.n) : 0.0,
             (unsigned long long)a.lu, (unsigned long long)a.la);
      tot_bytes += a.bytes;
   }
   printf("  tree total: %.1f MB = %.1f B/key\n", double(tot_bytes) / 1048576.0,
          double(tot_bytes) / double(total));

   t.cow_release(t.root_handle());  // ~write_session teardown dance
   t.adopt_root(root_t::null());
}

int main(int argc, char** argv)
{
   const uint64_t total = argc > 1 ? strtoull(argv[1], nullptr, 10) : 20'000'000ull;
   const uint64_t chunk = argc > 2 ? strtoull(argv[2], nullptr, 10) : 1'000'000ull;
   printf("part 48-bit handle A/B: %llu keys, %llu/commit\n", (unsigned long long)total,
          (unsigned long long)chunk);
   run_side<part::part_alloc>("line_ptr 32-bit (28-bit index, 32 GB ceiling)",
                              "/tmp/part_ab_4", total, chunk);
   run_side<part_alloc6>("line_ptr6 48-bit (44-bit index, 2 PB ceiling)", "/tmp/part_ab_6",
                         total, chunk);
   return 0;
}
