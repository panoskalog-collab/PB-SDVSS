#define _GNU_SOURCE
// main_sharma_repro_benchmark.c
// ------------------------------------------------------------
// Reproducible RELIC benchmark of Sharma et al. ID-SDVBS
// Type-III asymmetric adaptation: e: G1 x G2 -> GT.
//
// Benchmark-methodology improvements:
//   - clock_gettime(CLOCK_MONOTONIC) wall-clock timing.
//   - command-line iterations, warm-up iterations, repeated runs.
//   - optional CPU pinning with --pin-core 0.
//   - mean / median / standard deviation reporting.
//   - CSV export.
//   - compiler, CPU, system, and RELIC pairing-parameter metadata.
//   - message generation and verification dataset generation excluded from timed regions.
//   - setup/extract measured separately.
// ------------------------------------------------------------
//
// Build example:
//   gcc -O3 -Wall -Wextra -D COMPILE_FLAGS='"-O3 -Wall -Wextra"' main.c -o main -I/home/panagiotis/relic_install/include -L/home/panagiotis/relic_install/lib -Wl,-rpath,/home/panagiotis/relic_install/lib -lrelic -lgmp -lm
//
// Run example:
//   ./main --iterations 1000 --warmup 100 --repetitions 10 --pin-core 0 --csv sharma_id_sdvbs_results.csv
// ------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <errno.h>
#include <limits.h>
#include <unistd.h>
#include <sys/utsname.h>

#ifdef __linux__
#include <sched.h>
#endif

#include <relic/relic.h>
#include <relic/relic_core.h>

#ifndef COMPILE_FLAGS
#define COMPILE_FLAGS "unknown"
#endif

#define DEFAULT_ITERATIONS 1000
#define DEFAULT_WARMUP     100
#define DEFAULT_REPS       10
#define MSG_LEN            200

#define CHECK_OK(cond, msg) \
    do { if (!(cond)) { fprintf(stderr, "Error: %s\n", (msg)); exit(EXIT_FAILURE); } } while (0)

// Some RELIC builds define init/free macros as no-ops for automatic allocation.
// This small touch prevents -Wunused-parameter warnings in warning-clean builds.
#define RELIC_TOUCH(x) do { volatile const void *p_ = (const void *)&(x); (void)p_; } while (0)

// -----------------------------------------------------------------------------
// Configuration and utility code
// -----------------------------------------------------------------------------

typedef struct {
    size_t iterations;
    size_t warmup;
    int repetitions;
    int pin_requested;
    int pin_core;
    const char *csv_path;
} bench_config_t;

typedef struct {
    double mean;
    double median;
    double stddev;
} stats_t;

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void random_bytes(uint8_t *buf, size_t len) {
    rand_bytes(buf, (int)len);
}

static long parse_long_checked(const char *s, const char *name, long min_value) {
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno != 0 || !s[0] || *end != '\0' || v < min_value) {
        fprintf(stderr, "Invalid %s: %s\n", name, s);
        exit(EXIT_FAILURE);
    }
    return v;
}

static void usage(const char *prog) {
    printf("Usage:\n");
    printf("  %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  --iterations N     Timed operations per repeated run. Default: %d\n", DEFAULT_ITERATIONS);
    printf("  --warmup W         Untimed warm-up operations. Default: %d\n", DEFAULT_WARMUP);
    printf("  --repetitions R    Number of repeated timed runs. Default: %d\n", DEFAULT_REPS);
    printf("  --pin-core C       Pin process to CPU core C. Core 0 is valid.\n");
    printf("  --csv FILE         Write CSV summary to FILE.\n");
    printf("  --help             Show this help message.\n\n");
    printf("Example:\n");
    printf("  %s --iterations 1000 --warmup 100 --repetitions 10 --pin-core 0 --csv sharma_id_sdvbs_results.csv\n", prog);
}

static bench_config_t parse_args(int argc, char **argv) {
    bench_config_t cfg;
    cfg.iterations = DEFAULT_ITERATIONS;
    cfg.warmup = DEFAULT_WARMUP;
    cfg.repetitions = DEFAULT_REPS;
    cfg.pin_requested = 0;
    cfg.pin_core = -1;
    cfg.csv_path = NULL;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            exit(EXIT_SUCCESS);
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            cfg.iterations = (size_t)parse_long_checked(argv[++i], "iterations", 1);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            cfg.warmup = (size_t)parse_long_checked(argv[++i], "warmup", 0);
        } else if (strcmp(argv[i], "--repetitions") == 0 && i + 1 < argc) {
            cfg.repetitions = (int)parse_long_checked(argv[++i], "repetitions", 1);
        } else if (strcmp(argv[i], "--pin-core") == 0 && i + 1 < argc) {
            cfg.pin_core = (int)parse_long_checked(argv[++i], "pin-core", 0);
            cfg.pin_requested = 1;
        } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            cfg.csv_path = argv[++i];
        } else {
            fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
            usage(argv[0]);
            exit(EXIT_FAILURE);
        }
    }
    return cfg;
}

static int pin_process_to_core(int core_id) {
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core_id, &set);
    return sched_setaffinity(0, sizeof(set), &set);
#else
    (void)core_id;
    return -1;
#endif
}

static int count_online_cores(void) {
#ifdef _SC_NPROCESSORS_ONLN
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n > 0 && n < INT_MAX) return (int)n;
#endif
    return -1;
}

static void print_cpu_governors(int max_to_print) {
#ifdef __linux__
    int cores = count_online_cores();
    if (cores < 0) return;
    if (cores > max_to_print) cores = max_to_print;
    for (int i = 0; i < cores; ++i) {
        char path[256];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor", i);
        FILE *fp = fopen(path, "r");
        if (!fp) continue;
        char buf[128];
        if (fgets(buf, sizeof(buf), fp)) {
            buf[strcspn(buf, "\n")] = '\0';
            printf("CPU governor cpu%d: %s\n", i, buf);
        }
        fclose(fp);
    }
#else
    (void)max_to_print;
#endif
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

static stats_t compute_stats(const double *values, int n) {
    CHECK_OK(n > 0, "compute_stats with no samples");
    stats_t s;
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += values[i];
    s.mean = sum / (double)n;

    double *tmp = (double *)malloc((size_t)n * sizeof(double));
    CHECK_OK(tmp != NULL, "malloc stats tmp");
    memcpy(tmp, values, (size_t)n * sizeof(double));
    qsort(tmp, (size_t)n, sizeof(double), cmp_double);
    if (n % 2) s.median = tmp[n / 2];
    else       s.median = 0.5 * (tmp[n / 2 - 1] + tmp[n / 2]);
    free(tmp);

    double var = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = values[i] - s.mean;
        var += d * d;
    }
    s.stddev = sqrt(var / (double)n);
    return s;
}

static void print_result(const char *label, const stats_t *s, int reps) {
    double throughput = 1.0 / (s->mean > 0.0 ? s->mean : 1e-12);
    printf("  Algorithm %s, single-threaded:\n", label);
    printf("    mean:       %.3f us/op\n", s->mean * 1e6);
    printf("    median:     %.3f us/op\n", s->median * 1e6);
    printf("    std. dev.:  %.3f us/op over %d repeated runs\n", s->stddev * 1e6, reps);
    printf("    throughput: %.2f ops/sec, based on mean\n\n", throughput);
}

static void csv_write_header(FILE *csv) {
    fprintf(csv, "implementation,operation,iterations,warmup,repetitions,mean_us,median_us,stddev_us,throughput_ops_s,notes\n");
}

static void csv_write_row(FILE *csv,
                          const char *operation,
                          const bench_config_t *cfg,
                          const stats_t *s,
                          const char *notes) {
    double throughput = 1.0 / (s->mean > 0.0 ? s->mean : 1e-12);
    fprintf(csv,
            "Sharma_ID_SDVBS_RELIC,%s,%zu,%zu,%d,%.6f,%.6f,%.6f,%.6f,%s\n",
            operation,
            cfg->iterations,
            cfg->warmup,
            cfg->repetitions,
            s->mean * 1e6,
            s->median * 1e6,
            s->stddev * 1e6,
            throughput,
            notes ? notes : "");
}

// -----------------------------------------------------------------------------
// Hash helpers
// -----------------------------------------------------------------------------

static void bn_rand_nonzero_mod(bn_t out, const bn_t q) {
    do {
        bn_rand_mod(out, q);
    } while (bn_is_zero(out));
}

static void hash_to_bn_mod_q(bn_t out, const uint8_t *in, size_t in_len) {
    bn_t q, t;
    bn_null(q); bn_null(t);
    bn_new(q);  bn_new(t);

    g1_get_ord(q);
    bn_read_bin(t, in, (int)in_len);
    bn_mod(out, t, q);

    if (bn_is_zero(out)) bn_set_dig(out, 1);

    bn_free(q);
    bn_free(t);
}

static void H1_to_g1(g1_t out, const uint8_t *id, size_t idlen) {
    uint8_t h[32];
    md_map_sh256(h, id, (int)idlen);
    g1_map(out, h, 32);
}

static void H1_to_g2(g2_t out, const uint8_t *id, size_t idlen) {
    uint8_t h[32];
    md_map_sh256(h, id, (int)idlen);
    g2_map(out, h, 32);
}

static void H2_to_scalar(bn_t h_out,
                         const uint8_t *msg,
                         size_t mlen,
                         const g1_t U_prime) {
    uint8_t buf[4096];

    int u_len = g1_size_bin(U_prime, 1);
    CHECK_OK(u_len > 0, "invalid U_prime size");
    CHECK_OK(mlen + (size_t)u_len <= sizeof(buf), "H2 buffer too small");

    memcpy(buf, msg, mlen);
    g1_write_bin(buf + mlen, u_len, U_prime, 1);

    uint8_t h[32];
    md_map_sh256(h, buf, (int)(mlen + (size_t)u_len));

    hash_to_bn_mod_q(h_out, h, 32);
}

// -----------------------------------------------------------------------------
// Structures
// -----------------------------------------------------------------------------

typedef struct {
    bn_t s;          // master secret scalar
    g1_t P_IDs;      // signer public identity point in G1
    g1_t Q_IDs;      // signer private key in G1
    g2_t P_IDv;      // verifier public identity point in G2
    g2_t Q_IDv;      // verifier private key in G2
} id_sdvbs_keys_t;

typedef struct {
    g1_t U_prime;
    gt_t sigma;
} id_sdvbs_sig_t;

typedef struct {
    double commitment;
    double blinding;
    double signer_response;
    double unblinding;
    double total_issue;
} issue_phase_time_t;

typedef struct {
    uint8_t msg[MSG_LEN];
    id_sdvbs_sig_t sig;
} verify_item_t;

// -----------------------------------------------------------------------------
// Init / free helpers
// -----------------------------------------------------------------------------

static void keys_init(id_sdvbs_keys_t *K) {
    RELIC_TOUCH(K);
    bn_null(K->s);       bn_new(K->s);
    g1_null(K->P_IDs);   g1_new(K->P_IDs);
    g1_null(K->Q_IDs);   g1_new(K->Q_IDs);
    g2_null(K->P_IDv);   g2_new(K->P_IDv);
    g2_null(K->Q_IDv);   g2_new(K->Q_IDv);
}

static void keys_free(id_sdvbs_keys_t *K) {
    RELIC_TOUCH(K);
    bn_free(K->s);
    g1_free(K->P_IDs);
    g1_free(K->Q_IDs);
    g2_free(K->P_IDv);
    g2_free(K->Q_IDv);
}

static void sig_init(id_sdvbs_sig_t *sig) {
    RELIC_TOUCH(sig);
    g1_null(sig->U_prime); g1_new(sig->U_prime);
    gt_null(sig->sigma);   gt_new(sig->sigma);
}

static void sig_free(id_sdvbs_sig_t *sig) {
    RELIC_TOUCH(sig);
    g1_free(sig->U_prime);
    gt_free(sig->sigma);
}

static verify_item_t *verify_items_alloc(size_t n) {
    verify_item_t *items = (verify_item_t *)calloc(n, sizeof(*items));
    CHECK_OK(items != NULL, "calloc verify items");
    for (size_t i = 0; i < n; ++i) sig_init(&items[i].sig);
    return items;
}

static void verify_items_free(verify_item_t *items, size_t n) {
    if (!items) return;
    for (size_t i = 0; i < n; ++i) sig_free(&items[i].sig);
    free(items);
}

// -----------------------------------------------------------------------------
// Setup and Extract
// -----------------------------------------------------------------------------

static void setup_and_extract(id_sdvbs_keys_t *K,
                              const uint8_t *id_s,
                              size_t id_s_len,
                              const uint8_t *id_v,
                              size_t id_v_len) {
    keys_init(K);

    bn_t q;
    bn_null(q); bn_new(q);
    g1_get_ord(q);

    bn_rand_nonzero_mod(K->s, q);

    H1_to_g1(K->P_IDs, id_s, id_s_len);
    g1_mul(K->Q_IDs, K->P_IDs, K->s);
    g1_norm(K->Q_IDs, K->Q_IDs);

    H1_to_g2(K->P_IDv, id_v, id_v_len);
    g2_mul(K->Q_IDv, K->P_IDv, K->s);
    g2_norm(K->Q_IDv, K->Q_IDv);

    bn_free(q);
}

// -----------------------------------------------------------------------------
// ID-SDVBS Issue Protocol
// -----------------------------------------------------------------------------

static int issue_protocol_timed(const uint8_t *msg,
                                size_t mlen,
                                const id_sdvbs_keys_t *K,
                                id_sdvbs_sig_t *sig,
                                issue_phase_time_t *T) {
    double t0, t1, total0, total1;

    bn_t q;
    bn_null(q); bn_new(q);
    g1_get_ord(q);

    bn_t r, x, y, h, h1, x_inv, tmp_bn;
    bn_null(r); bn_null(x); bn_null(y); bn_null(h);
    bn_null(h1); bn_null(x_inv); bn_null(tmp_bn);

    bn_new(r); bn_new(x); bn_new(y); bn_new(h);
    bn_new(h1); bn_new(x_inv); bn_new(tmp_bn);

    g1_t U, tmp_g1, V, V_prime;
    g1_null(U); g1_null(tmp_g1); g1_null(V); g1_null(V_prime);
    g1_new(U);  g1_new(tmp_g1);  g1_new(V);  g1_new(V_prime);

    total0 = now_seconds();

    // Commitment Phase: U = r P_IDs
    t0 = now_seconds();
    bn_rand_nonzero_mod(r, q);
    g1_mul(U, K->P_IDs, r);
    g1_norm(U, U);
    t1 = now_seconds();
    if (T) T->commitment += t1 - t0;

    // Blinding Phase: U' = xU + xyP_IDs; h = H2(m,U'); h1 = x^{-1}h + y
    t0 = now_seconds();
    bn_rand_nonzero_mod(x, q);
    bn_rand_nonzero_mod(y, q);

    g1_mul(sig->U_prime, U, x);

    bn_mul(tmp_bn, x, y);
    bn_mod(tmp_bn, tmp_bn, q);

    g1_mul(tmp_g1, K->P_IDs, tmp_bn);
    g1_add(sig->U_prime, sig->U_prime, tmp_g1);
    g1_norm(sig->U_prime, sig->U_prime);

    H2_to_scalar(h, msg, mlen, sig->U_prime);

    bn_mod_inv(x_inv, x, q);

    bn_mul(tmp_bn, x_inv, h);
    bn_mod(tmp_bn, tmp_bn, q);

    bn_add(h1, tmp_bn, y);
    bn_mod(h1, h1, q);
    t1 = now_seconds();
    if (T) T->blinding += t1 - t0;

    // Signer Response Phase: V = (r + h1) Q_IDs
    t0 = now_seconds();
    bn_add(tmp_bn, r, h1);
    bn_mod(tmp_bn, tmp_bn, q);
    g1_mul(V, K->Q_IDs, tmp_bn);
    g1_norm(V, V);
    t1 = now_seconds();
    if (T) T->signer_response += t1 - t0;

    // Unblinding Phase: V' = xV; sigma = e(V', P_IDv)
    t0 = now_seconds();
    g1_mul(V_prime, V, x);
    g1_norm(V_prime, V_prime);
    pc_map(sig->sigma, V_prime, K->P_IDv);
    t1 = now_seconds();
    if (T) T->unblinding += t1 - t0;

    total1 = now_seconds();
    if (T) T->total_issue += total1 - total0;

    bn_free(q);
    bn_free(r); bn_free(x); bn_free(y); bn_free(h);
    bn_free(h1); bn_free(x_inv); bn_free(tmp_bn);

    g1_free(U); g1_free(tmp_g1); g1_free(V); g1_free(V_prime);

    return 1;
}

static int issue_protocol(const uint8_t *msg,
                          size_t mlen,
                          const id_sdvbs_keys_t *K,
                          id_sdvbs_sig_t *sig) {
    return issue_protocol_timed(msg, mlen, K, sig, NULL);
}

// -----------------------------------------------------------------------------
// Verification
// -----------------------------------------------------------------------------

static int verify_protocol(const uint8_t *msg,
                           size_t mlen,
                           const id_sdvbs_keys_t *K,
                           const id_sdvbs_sig_t *sig) {
    bn_t h;
    bn_null(h); bn_new(h);

    H2_to_scalar(h, msg, mlen, sig->U_prime);

    g1_t tmp_g1, lhs_g1;
    g1_null(tmp_g1); g1_null(lhs_g1);
    g1_new(tmp_g1);  g1_new(lhs_g1);

    g1_mul(tmp_g1, K->P_IDs, h);
    g1_add(lhs_g1, sig->U_prime, tmp_g1);
    g1_norm(lhs_g1, lhs_g1);

    gt_t check_sigma;
    gt_null(check_sigma); gt_new(check_sigma);

    pc_map(check_sigma, lhs_g1, K->Q_IDv);

    int ok = (gt_cmp(sig->sigma, check_sigma) == RLC_EQ);

    gt_free(check_sigma);
    g1_free(tmp_g1);
    g1_free(lhs_g1);
    bn_free(h);

    return ok;
}

// -----------------------------------------------------------------------------
// Optional DVBSim
// -----------------------------------------------------------------------------

static int simulate_protocol(const uint8_t *msg,
                             size_t mlen,
                             const id_sdvbs_keys_t *K,
                             id_sdvbs_sig_t *sig) {
    bn_t q;
    bn_null(q); bn_new(q);
    g1_get_ord(q);

    bn_t r, x, y, h, h1, x_inv, tmp_bn;
    bn_null(r); bn_null(x); bn_null(y); bn_null(h);
    bn_null(h1); bn_null(x_inv); bn_null(tmp_bn);

    bn_new(r); bn_new(x); bn_new(y); bn_new(h);
    bn_new(h1); bn_new(x_inv); bn_new(tmp_bn);

    g1_t U, tmp_g1, V, V_prime;
    g1_null(U); g1_null(tmp_g1); g1_null(V); g1_null(V_prime);
    g1_new(U);  g1_new(tmp_g1);  g1_new(V);  g1_new(V_prime);

    bn_rand_nonzero_mod(r, q);
    bn_rand_nonzero_mod(x, q);
    bn_rand_nonzero_mod(y, q);

    // U = r P_IDs
    g1_mul(U, K->P_IDs, r);
    g1_norm(U, U);

    // U' = xU + xyP_IDs
    g1_mul(sig->U_prime, U, x);

    bn_mul(tmp_bn, x, y);
    bn_mod(tmp_bn, tmp_bn, q);

    g1_mul(tmp_g1, K->P_IDs, tmp_bn);
    g1_add(sig->U_prime, sig->U_prime, tmp_g1);
    g1_norm(sig->U_prime, sig->U_prime);

    // h = H2(m, U')
    H2_to_scalar(h, msg, mlen, sig->U_prime);

    // h1 = x^{-1}h + y
    bn_mod_inv(x_inv, x, q);
    bn_mul(tmp_bn, x_inv, h);
    bn_mod(tmp_bn, tmp_bn, q);

    bn_add(h1, tmp_bn, y);
    bn_mod(h1, h1, q);

    // Type-III adaptation: verifier simulates directly with P_IDs and Q_IDv.
    bn_add(tmp_bn, r, h1);
    bn_mod(tmp_bn, tmp_bn, q);

    g1_mul(V, K->P_IDs, tmp_bn);
    g1_norm(V, V);
    g1_mul(V_prime, V, x);
    g1_norm(V_prime, V_prime);

    pc_map(sig->sigma, V_prime, K->Q_IDv);

    bn_free(q);
    bn_free(r); bn_free(x); bn_free(y); bn_free(h);
    bn_free(h1); bn_free(x_inv); bn_free(tmp_bn);

    g1_free(U); g1_free(tmp_g1); g1_free(V); g1_free(V_prime);

    return 1;
}

// -----------------------------------------------------------------------------
// Size reporting
// -----------------------------------------------------------------------------

static void print_sizes(const id_sdvbs_keys_t *K,
                        const id_sdvbs_sig_t *sig) {
    int g1_c = g1_size_bin(K->P_IDs, 1);
    int g1_u = g1_size_bin(K->P_IDs, 0);

    int g2_c = g2_size_bin(K->P_IDv, 1);
    int g2_u = g2_size_bin(K->P_IDv, 0);

    int gt_c = gt_size_bin(sig->sigma, 1);
    int gt_u = gt_size_bin(sig->sigma, 0);

    bn_t q;
    bn_null(q); bn_new(q);
    g1_get_ord(q);

    printf("  --- Cryptographic Sizes ---\n");
    printf("  Group order q:               %zu bits\n", bn_bits(q));
    printf("  Master secret s:             %zu bits\n\n", bn_bits(q));

    printf("  Signer public P_IDs:         %d bits compressed, %d bits uncompressed\n",
           g1_c * 8, g1_u * 8);

    printf("  Signer private Q_IDs:        %d bits compressed, %d bits uncompressed\n",
           g1_c * 8, g1_u * 8);

    printf("  Verifier public P_IDv:       %d bits compressed, %d bits uncompressed\n",
           g2_c * 8, g2_u * 8);

    printf("  Verifier private Q_IDv:      %d bits compressed, %d bits uncompressed\n\n",
           g2_c * 8, g2_u * 8);

    printf("  Signature compressed:        %d bits (U'=%d, sigma=%d)\n",
           (g1_c + gt_c) * 8, g1_c * 8, gt_c * 8);

    printf("  Signature uncompressed:      %d bits (U'=%d, sigma=%d)\n\n",
           (g1_u + gt_u) * 8, g1_u * 8, gt_u * 8);

    bn_free(q);
}

// -----------------------------------------------------------------------------
// Benchmark helpers
// -----------------------------------------------------------------------------

static void pregen_messages(uint8_t (*msgs)[MSG_LEN], size_t n) {
    for (size_t i = 0; i < n; ++i) random_bytes(msgs[i], MSG_LEN);
}

static void run_warmup(const bench_config_t *cfg,
                       const id_sdvbs_keys_t *K) {
    id_sdvbs_sig_t sig;
    sig_init(&sig);

    for (size_t i = 0; i < cfg->warmup; ++i) {
        uint8_t msg[MSG_LEN];
        random_bytes(msg, MSG_LEN);
        CHECK_OK(issue_protocol(msg, MSG_LEN, K, &sig), "warm Issue failed");
        CHECK_OK(verify_protocol(msg, MSG_LEN, K, &sig), "warm Verify failed");
    }

    sig_free(&sig);
}

static void benchmark_setup_extract(const bench_config_t *cfg,
                                    const uint8_t *ID_S,
                                    size_t ID_S_len,
                                    const uint8_t *ID_V,
                                    size_t ID_V_len,
                                    double *out_per_rep) {
    for (int r = 0; r < cfg->repetitions; ++r) {
        // Warm-up setup/extract operations, excluded from timing.
        for (size_t w = 0; w < cfg->warmup; ++w) {
            id_sdvbs_keys_t tmp;
            setup_and_extract(&tmp, ID_S, ID_S_len, ID_V, ID_V_len);
            keys_free(&tmp);
        }

        double t0 = now_seconds();
        for (size_t i = 0; i < cfg->iterations; ++i) {
            id_sdvbs_keys_t tmp;
            setup_and_extract(&tmp, ID_S, ID_S_len, ID_V, ID_V_len);
            keys_free(&tmp);
        }
        double t1 = now_seconds();
        out_per_rep[r] = (t1 - t0) / (double)cfg->iterations;
    }
}

static void benchmark_issue(const bench_config_t *cfg,
                            const id_sdvbs_keys_t *K,
                            uint8_t (*msgs)[MSG_LEN],
                            double *issue_total,
                            double *commitment,
                            double *blinding,
                            double *signer_response,
                            double *unblinding) {
    for (int r = 0; r < cfg->repetitions; ++r) {
        run_warmup(cfg, K);

        id_sdvbs_sig_t sig;
        sig_init(&sig);

        issue_phase_time_t T;
        memset(&T, 0, sizeof(T));

        for (size_t i = 0; i < cfg->iterations; ++i) {
            CHECK_OK(issue_protocol_timed(msgs[i], MSG_LEN, K, &sig, &T), "Issue failed");
        }

        issue_total[r]     = T.total_issue / (double)cfg->iterations;
        commitment[r]      = T.commitment / (double)cfg->iterations;
        blinding[r]        = T.blinding / (double)cfg->iterations;
        signer_response[r] = T.signer_response / (double)cfg->iterations;
        unblinding[r]      = T.unblinding / (double)cfg->iterations;

        sig_free(&sig);
    }
}

static void build_verify_dataset(const bench_config_t *cfg,
                                 const id_sdvbs_keys_t *K,
                                 verify_item_t *items) {
    printf("  Building verification dataset outside timed region...\n");
    for (size_t i = 0; i < cfg->iterations; ++i) {
        random_bytes(items[i].msg, MSG_LEN);
        CHECK_OK(issue_protocol(items[i].msg, MSG_LEN, K, &items[i].sig), "dataset Issue failed");
        CHECK_OK(verify_protocol(items[i].msg, MSG_LEN, K, &items[i].sig), "dataset Verify failed");
    }
    printf("    Verification dataset: %zu valid signatures\n", cfg->iterations);
}

static void benchmark_verify(const bench_config_t *cfg,
                             const id_sdvbs_keys_t *K,
                             verify_item_t *items,
                             double *out_per_rep) {
    for (int r = 0; r < cfg->repetitions; ++r) {
        run_warmup(cfg, K);

        long failures = 0;
        double t0 = now_seconds();
        for (size_t i = 0; i < cfg->iterations; ++i) {
            int ok = verify_protocol(items[i].msg, MSG_LEN, K, &items[i].sig);
            if (!ok) failures++;
        }
        double t1 = now_seconds();
        CHECK_OK(failures == 0, "verification dataset failure");
        out_per_rep[r] = (t1 - t0) / (double)cfg->iterations;
    }
}

static void benchmark_simulate(const bench_config_t *cfg,
                               const id_sdvbs_keys_t *K,
                               uint8_t (*msgs)[MSG_LEN],
                               double *out_per_rep) {
    for (int r = 0; r < cfg->repetitions; ++r) {
        run_warmup(cfg, K);

        id_sdvbs_sig_t sig;
        sig_init(&sig);

        double t0 = now_seconds();
        for (size_t i = 0; i < cfg->iterations; ++i) {
            CHECK_OK(simulate_protocol(msgs[i], MSG_LEN, K, &sig), "Simulate failed");
        }
        double t1 = now_seconds();
        out_per_rep[r] = (t1 - t0) / (double)cfg->iterations;

        sig_free(&sig);
    }
}

static void print_metadata(const bench_config_t *cfg) {
    printf("Sharma ID-SDVBS RELIC benchmark metadata\n");
    printf("============================================================\n");
    if (cfg->pin_requested) {
        printf("Benchmark mode: single-threaded process, requested core %d\n", cfg->pin_core);
    } else {
        printf("Benchmark mode: single-threaded process, no CPU pinning requested\n");
    }
    printf("Timing method: clock_gettime(CLOCK_MONOTONIC), wall-clock seconds\n");
    printf("Message length: %d bytes\n", MSG_LEN);
    printf("Iterations per repeated run: %zu\n", cfg->iterations);
    printf("Warm-up iterations: %zu\n", cfg->warmup);
    printf("Repeated timed runs: %d\n", cfg->repetitions);
    printf("Timed region Issue: ID-SDVBS online issue only; message generation excluded\n");
    printf("Timed region Verify: Verify only; message/signature generation excluded\n");
    printf("Timed region Simulate: DVBSim only; message generation excluded\n");
    printf("Compiler flags recorded: %s\n", COMPILE_FLAGS);
#if defined(__GNUC__)
    printf("Compiler: GCC-compatible %d.%d.%d\n", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
    printf("Compiler: unknown\n");
#endif
    int cores = count_online_cores();
    if (cores > 0) printf("CPU online cores: %d\n", cores);

    struct utsname uts;
    if (uname(&uts) == 0) {
        printf("System: %s %s %s %s\n", uts.sysname, uts.release, uts.machine, uts.nodename);
    }
    print_cpu_governors(8);
    printf("RELIC parameter set selected by pc_param_set_any(); details printed below.\n");
    printf("============================================================\n\n");
}

static void benchmark_all(const bench_config_t *cfg) {
    const uint8_t ID_S[] = "Signer";
    const uint8_t ID_V[] = "DesignatedVerifier";

    printf("--- RELIC pairing parameters ---\n");
    pc_param_print();
    printf("--------------------------------\n\n");

    id_sdvbs_keys_t K;
    setup_and_extract(&K, ID_S, sizeof(ID_S) - 1, ID_V, sizeof(ID_V) - 1);

    id_sdvbs_sig_t sig;
    sig_init(&sig);
    uint8_t first_msg[MSG_LEN];
    random_bytes(first_msg, MSG_LEN);
    CHECK_OK(issue_protocol(first_msg, MSG_LEN, &K, &sig), "initial Issue failed");
    CHECK_OK(verify_protocol(first_msg, MSG_LEN, &K, &sig), "initial Verify failed");

    printf("=== Sharma et al. ID-SDVBS over RELIC, Type-III adaptation ===\n");
    print_sizes(&K, &sig);

    uint8_t (*msgs)[MSG_LEN] = malloc(cfg->iterations * sizeof(*msgs));
    CHECK_OK(msgs != NULL, "malloc messages");
    pregen_messages(msgs, cfg->iterations);

    verify_item_t *verify_items = verify_items_alloc(cfg->iterations);
    build_verify_dataset(cfg, &K, verify_items);
    printf("\n");

    double *setup_values = calloc((size_t)cfg->repetitions, sizeof(double));
    double *issue_values = calloc((size_t)cfg->repetitions, sizeof(double));
    double *commit_values = calloc((size_t)cfg->repetitions, sizeof(double));
    double *blind_values = calloc((size_t)cfg->repetitions, sizeof(double));
    double *resp_values = calloc((size_t)cfg->repetitions, sizeof(double));
    double *unblind_values = calloc((size_t)cfg->repetitions, sizeof(double));
    double *verify_values = calloc((size_t)cfg->repetitions, sizeof(double));
    double *simulate_values = calloc((size_t)cfg->repetitions, sizeof(double));
    CHECK_OK(setup_values && issue_values && commit_values && blind_values &&
             resp_values && unblind_values && verify_values && simulate_values,
             "calloc result vectors");

    benchmark_setup_extract(cfg, ID_S, sizeof(ID_S) - 1, ID_V, sizeof(ID_V) - 1, setup_values);
    benchmark_issue(cfg, &K, msgs, issue_values, commit_values, blind_values, resp_values, unblind_values);
    benchmark_verify(cfg, &K, verify_items, verify_values);
    benchmark_simulate(cfg, &K, msgs, simulate_values);

    stats_t setup_s     = compute_stats(setup_values, cfg->repetitions);
    stats_t issue_s     = compute_stats(issue_values, cfg->repetitions);
    stats_t commit_s    = compute_stats(commit_values, cfg->repetitions);
    stats_t blind_s     = compute_stats(blind_values, cfg->repetitions);
    stats_t response_s  = compute_stats(resp_values, cfg->repetitions);
    stats_t unblind_s   = compute_stats(unblind_values, cfg->repetitions);
    stats_t verify_s    = compute_stats(verify_values, cfg->repetitions);
    stats_t simulate_s  = compute_stats(simulate_values, cfg->repetitions);

    printf("  --- Performance Benchmarks ---\n");
    print_result("Setup + Key Extract", &setup_s, cfg->repetitions);

    print_result("Issue / Total", &issue_s, cfg->repetitions);
    printf("  Issue phase breakdown, mean values:\n");
    printf("    commitment:       %.3f us/op\n", commit_s.mean * 1e6);
    printf("    blinding:         %.3f us/op\n", blind_s.mean * 1e6);
    printf("    signer response:  %.3f us/op\n", response_s.mean * 1e6);
    printf("    unblinding:       %.3f us/op\n\n", unblind_s.mean * 1e6);

    print_result("Designated Verify", &verify_s, cfg->repetitions);
    print_result("DVBSim / Simulate", &simulate_s, cfg->repetitions);

    printf("  --- Notes ---\n");
    printf("  This is a RELIC Type-III adaptation of the original symmetric-pairing notation.\n");
    printf("  Online Issue includes commitment, blinding, signer response, unblinding, and final pairing.\n");
    printf("  Setup and identity extraction are measured separately.\n");
    printf("  Verification and simulation datasets are generated outside the timed region.\n\n");

    if (cfg->csv_path) {
        FILE *csv = fopen(cfg->csv_path, "w");
        CHECK_OK(csv != NULL, "open CSV output");
        csv_write_header(csv);
        csv_write_row(csv, "SetupExtract", cfg, &setup_s, "setup_and_extract");
        csv_write_row(csv, "Issue_Total", cfg, &issue_s, "commitment+blinding+signer_response+unblinding");
        csv_write_row(csv, "Issue_Commitment", cfg, &commit_s, "phase_only");
        csv_write_row(csv, "Issue_Blinding", cfg, &blind_s, "phase_only");
        csv_write_row(csv, "Issue_Signer_Response", cfg, &response_s, "phase_only");
        csv_write_row(csv, "Issue_Unblinding", cfg, &unblind_s, "phase_only_includes_pairing");
        csv_write_row(csv, "Designated_Verify", cfg, &verify_s, "verify_only_dataset_pregenerated");
        csv_write_row(csv, "DVBSim", cfg, &simulate_s, "simulate_only");
        fclose(csv);
        printf("CSV results written to: %s\n", cfg->csv_path);
    }

    free(setup_values);
    free(issue_values);
    free(commit_values);
    free(blind_values);
    free(resp_values);
    free(unblind_values);
    free(verify_values);
    free(simulate_values);

    verify_items_free(verify_items, cfg->iterations);
    free(msgs);
    sig_free(&sig);
    keys_free(&K);
}

int main(int argc, char **argv) {
    bench_config_t cfg = parse_args(argc, argv);

    if (cfg.pin_requested) {
        if (pin_process_to_core(cfg.pin_core) != 0) {
            fprintf(stderr, "Warning: failed to pin process to core %d. Continuing without enforced affinity.\n",
                    cfg.pin_core);
        }
    }

    if (core_init() != RLC_OK) {
        core_clean();
        fprintf(stderr, "RELIC core_init failed\n");
        return EXIT_FAILURE;
    }

    if (pc_param_set_any() != RLC_OK) {
        fprintf(stderr, "RELIC pc_param_set_any failed\n");
        core_clean();
        return EXIT_FAILURE;
    }

    print_metadata(&cfg);
    benchmark_all(&cfg);

    core_clean();
    return EXIT_SUCCESS;
}
