// nvbench.cpp - read-only microbench of the file-tier expert reads (GSQ-217 GGUF, Q3_K experts, 3 spans of
// 5,068,800 B each = one 14.5 MiB blob).  Modes:
//   pool  : persistent T threads, each blob split into P parts (the engine's read_part), O_DIRECT pread in chunks
//           of <= C bytes into a 4 KiB-aligned bounce, then memcpy into the destination (the arena slot)
//   spawn : the same but T std::threads created per batch (gpu_run's arena_adapt path)
//   uring : one thread, raw io_uring, every C-byte chunk of the batch in flight at once (QD up to -q)
// Options: -m mode -t threads -p parts -b blobs/batch -n batches -g gap_us -c chunk -d direct(1/0) -k copy(1/0)
//          -D heap|reg (destination: malloc or hipHostRegister'ed like the arena) -s same(1: always the same blobs)
//          -q uring depth -r seed -T (print per-batch times)
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <linux/io_uring.h>
#include <mutex>
#include <random>
#include <string>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include <dlfcn.h>

static const uint64_t SPAN = 5068800, BLOB = 3 * SPAN;
struct Ex { int l, e; uint64_t off[3]; };
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static int g_fd = -1;
static uint64_t g_chunk = 1 << 20;
static bool g_copy = true, g_direct = true;
static int g_smask = 7;
static bool g_contig = false;
static int g_ws = 0;

// one aligned read of [off, off+n) into dst via the bounce (direct) or straight (buffered)
static void read_range(uint64_t off, uint64_t n, uint8_t* dst, uint8_t* bounce) {
    if (!g_direct) {
        uint64_t got = 0;
        while (got < n) {
            const uint64_t want = std::min<uint64_t>(g_chunk, n - got);
            const ssize_t k = ::pread(g_fd, dst + got, want, (off_t) (off + got));
            if (k <= 0) { perror("pread"); std::abort(); }
            got += (uint64_t) k;
        }
        return;
    }
    const uint64_t a0 = off & ~4095ull, a1 = (off + n + 4095) & ~4095ull;
    uint64_t got = 0;
    while (got < a1 - a0) {
        const uint64_t want = std::min<uint64_t>(g_chunk, a1 - a0 - got);
        const ssize_t k = ::pread(g_fd, bounce + got, want, (off_t) (a0 + got));
        if (k <= 0) { perror("pread direct"); std::abort(); }
        got += (uint64_t) k;
    }
    if (g_copy) std::memcpy(dst, bounce + (off - a0), n);
}

// part `part` of `nparts` of blob x into dst (engine's Ds41FileBlobs::read_part cut)
static void read_part(const Ex& x, int part, int nparts, uint8_t* dst, uint8_t* bounce) {
    uint64_t lo = BLOB * part / nparts, hi = BLOB * (part + 1) / nparts;
    lo &= ~4095ull;
    if (part + 1 < nparts) hi &= ~4095ull; else hi = BLOB;
    uint64_t at = 0;
    for (int s = 0; s < 3; ++s) {
        const uint64_t a = std::max(lo, at), b = std::min(hi, at + SPAN);
        if (a < b) read_range(x.off[s] + (a - at), b - a, dst + a, bounce);
        at += SPAN;
    }
}

// ---- raw io_uring ----
struct Ring {
    int fd = -1;
    unsigned *sq_head, *sq_tail, *sq_mask, *sq_array, *cq_head, *cq_tail, *cq_mask;
    io_uring_sqe* sqes;
    io_uring_cqe* cqes;
    unsigned entries;
    bool init(unsigned n) {
        io_uring_params p{};
        fd = (int) syscall(__NR_io_uring_setup, n, &p);
        if (fd < 0) { perror("io_uring_setup"); return false; }
        entries = p.sq_entries;
        size_t sz = std::max<size_t>(p.sq_off.array + p.sq_entries * sizeof(unsigned), p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe));
        uint8_t* q = (uint8_t*) mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQ_RING);
        if (q == MAP_FAILED) { perror("mmap sq"); return false; }
        sq_head = (unsigned*) (q + p.sq_off.head); sq_tail = (unsigned*) (q + p.sq_off.tail);
        sq_mask = (unsigned*) (q + p.sq_off.ring_mask); sq_array = (unsigned*) (q + p.sq_off.array);
        cq_head = (unsigned*) (q + p.cq_off.head); cq_tail = (unsigned*) (q + p.cq_off.tail);
        cq_mask = (unsigned*) (q + p.cq_off.ring_mask); cqes = (io_uring_cqe*) (q + p.cq_off.cqes);
        sqes = (io_uring_sqe*) mmap(nullptr, p.sq_entries * sizeof(io_uring_sqe), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQES);
        if (sqes == MAP_FAILED) { perror("mmap sqes"); return false; }
        return true;
    }
    void push_read(int file, void* buf, unsigned len, uint64_t off, uint64_t ud) {
        const unsigned t = *sq_tail, i = t & *sq_mask;
        io_uring_sqe* s = &sqes[i];
        std::memset(s, 0, sizeof *s);
        s->opcode = IORING_OP_READ; s->fd = file; s->addr = (uint64_t) buf; s->len = len; s->off = off; s->user_data = ud;
        sq_array[i] = i;
        __atomic_store_n(sq_tail, t + 1, __ATOMIC_RELEASE);
    }
    int enter(unsigned sub, unsigned minc) {
        return (int) syscall(__NR_io_uring_enter, fd, sub, minc, minc ? IORING_ENTER_GETEVENTS : 0, nullptr, 0);
    }
    int reap() {   // returns completions reaped
        unsigned h = *cq_head; int n = 0;
        while (h != __atomic_load_n(cq_tail, __ATOMIC_ACQUIRE)) {
            const io_uring_cqe* c = &cqes[h & *cq_mask];
            if (c->res < 0) { fprintf(stderr, "uring read error %d\n", c->res); std::abort(); }
            ++h; ++n;
        }
        __atomic_store_n(cq_head, h, __ATOMIC_RELEASE);
        return n;
    }
};

int main(int argc, char** argv) {
    const std::string home = std::getenv("HOME") ? std::getenv("HOME") : ".";
    std::string mode = "pool", dstm = "heap", path = home + "/models/xolotl/V41-1017-GSQ/V41-1017-GSQ-217.gguf",
                list = home + "/.cache/xolotl/nvme/experts.txt";
    int T = 8, P = 8, NB = 1, N = 200, Q = 256, seed = 1;
    long gap = 0; bool same = false, trace = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto v = [&]() { return std::string(argv[++i]); };
        if (a == "-m") mode = v(); else if (a == "-t") T = std::stoi(v()); else if (a == "-p") P = std::stoi(v());
        else if (a == "-b") NB = std::stoi(v()); else if (a == "-n") N = std::stoi(v()); else if (a == "-g") gap = std::stol(v());
        else if (a == "-c") g_chunk = std::stoull(v()); else if (a == "-d") g_direct = std::stoi(v()); else if (a == "-k") g_copy = std::stoi(v());
        else if (a == "-D") dstm = v(); else if (a == "-s") same = std::stoi(v()); else if (a == "-q") Q = std::stoi(v());
        else if (a == "-W") g_ws = std::stoi(v()); else if (a == "-C") g_contig = std::stoi(v()); else if (a == "-S") g_smask = std::stoi(v()); else if (a == "-r") seed = std::stoi(v()); else if (a == "-T") trace = true; else if (a == "-f") path = v();
        else { fprintf(stderr, "bad arg %s\n", a.c_str()); return 1; }
    }
    std::vector<Ex> ex;
    { FILE* f = fopen(list.c_str(), "r"); Ex x; unsigned long long a, b, c;
      while (fscanf(f, "%d %d %llu %llu %llu", &x.l, &x.e, &a, &b, &c) == 5) { x.off[0] = a; x.off[1] = b; x.off[2] = c; ex.push_back(x); }
      fclose(f); }
    if (g_contig) for (Ex& x : ex) { x.off[1] = x.off[0] + SPAN; x.off[2] = x.off[0] + 2 * SPAN; }
    g_fd = ::open(path.c_str(), O_RDONLY | (g_direct ? O_DIRECT : 0));
    if (g_fd < 0) { perror("open"); return 1; }
    { int f2 = ::open(path.c_str(), O_RDONLY); posix_fadvise(f2, 0, 0, POSIX_FADV_DONTNEED); ::close(f2); }   // no page-cache hits
    if (!g_direct) posix_fadvise(g_fd, 0, 0, POSIX_FADV_RANDOM);
    // destination: 16 blob slots packed at BLOB stride (odd ones 2 KiB off a page, like the arena)
    const int NS = 16;
    const size_t dbytes = (size_t) NS * BLOB + 4096;
    uint8_t* dst = (uint8_t*) aligned_alloc(4096, (dbytes + 4095) & ~4095ull);
    std::memset(dst, 0, dbytes);
    if (dstm == "reg") {
        void* h = dlopen("libamdhip64.so", RTLD_NOW);
        if (!h) { fprintf(stderr, "no libamdhip64\n"); return 1; }
        auto reg = (int (*)(void*, size_t, unsigned)) dlsym(h, "hipHostRegister");
        int r = reg(dst, (dbytes + 4095) & ~4095ull, 0);
        fprintf(stderr, "hipHostRegister -> %d\n", r);
    }
    std::mt19937_64 rng(seed);
    std::vector<int> pick((size_t) NB);
    auto choose = [&](int it) {
        for (int j = 0; j < NB; ++j) pick[(size_t) j] = same ? j : (int) (rng() % (g_ws > 0 ? (size_t) g_ws : ex.size()));
        (void) it;
    };
    const uint64_t bmax = SPAN + 3 * 4096 + g_chunk;
    std::vector<double> ts;
    // ---- pool ----
    std::atomic<int> gen{0}, done{0};
    std::atomic<size_t> nx{0};
    std::atomic<bool> quit{false};
    std::mutex mu; std::condition_variable cv;
    size_t tasks = 0;
    auto work = [&](uint8_t* bounce) {
        for (size_t r; (r = nx.fetch_add(1)) < tasks;) {
            const int b = (int) (r / (size_t) P);
            read_part(ex[(size_t) pick[(size_t) b]], (int) (r % (size_t) P), P, dst + (size_t) (b % NS) * BLOB, bounce);
        }
    };
    std::vector<std::thread> pool;
    if (mode == "pool")
        for (int q = 0; q < T; ++q)
            pool.emplace_back([&]() {
                uint8_t* bounce = (uint8_t*) aligned_alloc(4096, (bmax + 4095) & ~4095ull);
                std::memset(bounce, 0, bmax);
                int seen = 0;
                for (;;) {
                    { std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&] { return gen.load() != seen || quit.load(); }); }
                    if (quit.load()) break;
                    seen = gen.load();
                    work(bounce);
                    done.fetch_add(1);
                }
            });
    Ring ring;
    std::vector<uint8_t*> ub;   // uring bounces: per blob, 3 aligned span regions
    if (mode == "uring") {
        if (!ring.init((unsigned) Q)) return 1;
        for (int b = 0; b < NB; ++b) {
            uint8_t* p = (uint8_t*) aligned_alloc(4096, 3 * (SPAN + 8192));
            std::memset(p, 0, 3 * (SPAN + 8192));
            ub.push_back(p);
        }
    }
    double tot_read = 0, tot_copy = 0;
    for (int it = 0; it < N; ++it) {
        choose(it);
        if (gap > 0) usleep((useconds_t) gap);
        const double t0 = now_ms();
        if (mode == "pool") {
            tasks = (size_t) NB * (size_t) P;
            nx = 0; done = 0;
            { std::lock_guard<std::mutex> lk(mu); gen.fetch_add(1); }
            cv.notify_all();
            while (done.load() < T) { /* spin */ }
        } else if (mode == "spawn") {
            tasks = (size_t) NB * (size_t) P;
            nx = 0;
            std::vector<std::thread> th;
            const int nth = (int) std::min<size_t>((size_t) T, tasks);
            for (int q = 0; q < nth; ++q)
                th.emplace_back([&]() {
                    thread_local uint8_t* bounce = nullptr;
                    if (!bounce) { bounce = (uint8_t*) aligned_alloc(4096, (bmax + 4095) & ~4095ull); }
                    work(bounce);
                });
            for (auto& t : th) t.join();
        } else if (mode == "uring") {
            // every chunk of every span in flight (up to the ring depth), then the copies
            struct C { int fdx; void* buf; unsigned len; uint64_t off; };
            std::vector<C> cs;
            for (int b = 0; b < NB; ++b) {
                const Ex& x = ex[(size_t) pick[(size_t) b]];
                for (int s = 0; s < 3; ++s) {
                    if (!(g_smask >> s & 1)) continue;
                    const uint64_t a0 = x.off[s] & ~4095ull, a1 = (x.off[s] + SPAN + 4095) & ~4095ull;
                    uint8_t* base = ub[(size_t) b] + (size_t) s * (SPAN + 8192);
                    for (uint64_t o = a0; o < a1; o += g_chunk)
                        cs.push_back({g_fd, base + (o - a0), (unsigned) std::min<uint64_t>(g_chunk, a1 - o), o});
                }
            }
            size_t sub = 0, comp = 0, infl = 0;
            while (comp < cs.size()) {
                unsigned n = 0;
                while (sub < cs.size() && infl < (size_t) ring.entries) {
                    ring.push_read(cs[sub].fdx, cs[sub].buf, cs[sub].len, cs[sub].off, sub); ++sub; ++infl; ++n;
                }
                ring.enter(n, 1);
                const int r = ring.reap();
                comp += (size_t) r; infl -= (size_t) r;
            }
            const double tr = now_ms();
            tot_read += tr - t0;
            if (g_copy)
                for (int b = 0; b < NB; ++b) {
                    const Ex& x = ex[(size_t) pick[(size_t) b]];
                    for (int s = 0; s < 3; ++s)
                        std::memcpy(dst + (size_t) (b % NS) * BLOB + (size_t) s * SPAN,
                                    ub[(size_t) b] + (size_t) s * (SPAN + 8192) + (x.off[s] & 4095), SPAN);
                }
            tot_copy += now_ms() - tr;
        }
        const double dt = now_ms() - t0;
        ts.push_back(dt);
        if (trace) printf("%d %.3f\n", it, dt);
    }
    if (mode == "pool") { { std::lock_guard<std::mutex> lk(mu); quit = true; } cv.notify_all(); for (auto& t : pool) t.join(); }
    std::vector<double> s = ts;
    const int skip = std::min<int>(5, (int) s.size() / 10);
    s.erase(s.begin(), s.begin() + skip);
    std::sort(s.begin(), s.end());
    double m = 0; for (double v : s) m += v; m /= (double) s.size();
    auto pc = [&](double q) { return s[(size_t) std::min<double>((double) s.size() - 1, q * (double) s.size())]; };
    const double gbs = (double) NB * (double) BLOB / (m * 1e-3) / 1e9;
    printf("mode=%s T=%d P=%d NB=%d chunk=%llu direct=%d copy=%d dst=%s gap=%ldus same=%d | mean %.3f ms p10 %.3f p50 %.3f p90 %.3f max %.3f | %.2f GB/s | per blob %.3f ms",
           mode.c_str(), T, P, NB, (unsigned long long) g_chunk, (int) g_direct, (int) g_copy, dstm.c_str(), gap, (int) same,
           m, pc(0.1), pc(0.5), pc(0.9), s.back(), gbs, m / NB);
    if (mode == "uring") printf(" | read %.3f copy %.3f", tot_read / N, tot_copy / N);
    printf("\n");
    return 0;
}
