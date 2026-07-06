#define _GNU_SOURCE
// main_multicore_repro_benchmark.c
// PB-SDVSS OpenSSL EC multi-core benchmark with reproducibility metadata.
//
// Compile example:
//   gcc -O3 -Wall -Wextra -fopenmp -D COMPILE_FLAGS='"-O3 -Wall -Wextra -fopenmp"' main.c -o main -lcrypto -lm
//
// Run example:
//   ./main --iterations 10000 --warmup 1000 --repetitions 10 --threads 4 --curve p256 --pin-cores 0-3 --csv pb_sdvss_multicore_results.csv
//
// Notes:
//   - Dataset generation is outside the timed region.
//   - Signing timing measures PB-SDVSS.Sign only over pre-generated messages/info.
//   - Verification timing measures PB-SDVSS.Verify only over pre-generated datasets.
//   - Dataset A contains valid signatures.
//   - Dataset B contains hard-invalid signatures created by corrupting sig2 while
//     keeping m/info/a*/x* consistent, forcing the EC verification equation.
//   - Dataset C alternates valid and hard-invalid signatures 50/50.

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

#include <openssl/ec.h>
#include <openssl/bn.h>
#include <openssl/sha.h>
#include <openssl/obj_mac.h>
#include <openssl/opensslv.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define DEFAULT_ITERATIONS 10000
#define DEFAULT_WARMUP     1000
#define DEFAULT_REPETITIONS 10
#define MSG_LEN  200
#define INFO_LEN 32

#define CHECK(c,msg) do { if (!(c)) { fprintf(stderr, "Error: %s\n", (msg)); exit(EXIT_FAILURE); } } while (0)

#ifndef COMPILE_FLAGS
#define COMPILE_FLAGS "unknown"
#endif

typedef struct {
    size_t iterations;
    size_t warmup;
    int repetitions;
    int threads;
    const char *curve_tag;
    const char *csv_path;
    const char *pin_cores;
} bench_config_t;

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

static double mean_double(const double *x, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += x[i];
    return s / (double)n;
}

static double median_double(const double *x, int n) {
    double *tmp = (double *)malloc((size_t)n * sizeof(double));
    CHECK(tmp != NULL, "malloc median tmp");
    memcpy(tmp, x, (size_t)n * sizeof(double));
    qsort(tmp, (size_t)n, sizeof(double), cmp_double);
    double med;
    if (n % 2) med = tmp[n / 2];
    else med = 0.5 * (tmp[n / 2 - 1] + tmp[n / 2]);
    free(tmp);
    return med;
}

static double stddev_double(const double *x, int n) {
    if (n <= 1) return 0.0;
    double m = mean_double(x, n);
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = x[i] - m;
        s += d * d;
    }
    return sqrt(s / (double)(n - 1));
}

static long parse_long_strict(const char *s, const char *name, long min_value) {
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
    fprintf(stderr,
            "Usage:\n"
            "  %s [options]\n\n"
            "Options:\n"
            "  --iterations N       Timed operations per repeated run. Default: %d\n"
            "  --warmup W           Untimed warm-up operations. Default: %d\n"
            "  --repetitions R      Repeated timed runs. Default: %d\n"
            "  --threads T          OpenMP worker threads. Default: OpenMP max or 1\n"
            "  --curve TAG          p192 | p224 | p256 | k256 | p384 | p521 | bp256 | bp384 | all. Default: p256\n"
            "  --pin-cores LIST     Restrict process affinity, e.g., 0, 0-3, or 0,2,4-5\n"
            "  --pin-core C         Alias for --pin-cores C; accepts core 0\n"
            "  --csv FILE           Write CSV results\n"
            "  --help               Show this message\n\n"
            "Example:\n"
            "  %s --iterations 10000 --warmup 1000 --repetitions 10 --threads 4 --curve p256 --pin-cores 0-3 --csv pb_sdvss_multicore_results.csv\n",
            prog, DEFAULT_ITERATIONS, DEFAULT_WARMUP, DEFAULT_REPETITIONS, prog);
}

static bench_config_t parse_args(int argc, char **argv) {
#ifdef _OPENMP
    int default_threads = omp_get_max_threads();
#else
    int default_threads = 1;
#endif
    bench_config_t cfg;
    cfg.iterations = DEFAULT_ITERATIONS;
    cfg.warmup = DEFAULT_WARMUP;
    cfg.repetitions = DEFAULT_REPETITIONS;
    cfg.threads = default_threads;
    cfg.curve_tag = "p256";
    cfg.csv_path = NULL;
    cfg.pin_cores = NULL;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            cfg.iterations = (size_t)parse_long_strict(argv[++i], "iterations", 1);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            cfg.warmup = (size_t)parse_long_strict(argv[++i], "warmup", 0);
        } else if (strcmp(argv[i], "--repetitions") == 0 && i + 1 < argc) {
            cfg.repetitions = (int)parse_long_strict(argv[++i], "repetitions", 1);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            cfg.threads = (int)parse_long_strict(argv[++i], "threads", 1);
        } else if (strcmp(argv[i], "--curve") == 0 && i + 1 < argc) {
            cfg.curve_tag = argv[++i];
        } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            cfg.csv_path = argv[++i];
        } else if (strcmp(argv[i], "--pin-cores") == 0 && i + 1 < argc) {
            cfg.pin_cores = argv[++i];
        } else if (strcmp(argv[i], "--pin-core") == 0 && i + 1 < argc) {
            cfg.pin_cores = argv[++i];
            (void)parse_long_strict(cfg.pin_cores, "pin-core", 0);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            exit(EXIT_SUCCESS);
        } else {
            usage(argv[0]);
            exit(EXIT_FAILURE);
        }
    }

#ifndef _OPENMP
    if (cfg.threads != 1) {
        fprintf(stderr, "Warning: binary compiled without OpenMP; forcing --threads 1.\n");
        cfg.threads = 1;
    }
#endif
    return cfg;
}

#ifdef __linux__
static void add_cpu_range(cpu_set_t *set, long a, long b) {
    if (a < 0 || b < 0 || b < a || b > CPU_SETSIZE - 1) {
        fprintf(stderr, "Invalid CPU range: %ld-%ld\n", a, b);
        exit(EXIT_FAILURE);
    }
    for (long c = a; c <= b; ++c) CPU_SET((int)c, set);
}

static void apply_affinity_if_requested(const char *spec) {
    if (!spec) return;

    cpu_set_t set;
    CPU_ZERO(&set);

    char *copy = strdup(spec);
    CHECK(copy != NULL, "strdup pin-cores");
    char *tok = strtok(copy, ",");
    int count = 0;
    while (tok) {
        char *dash = strchr(tok, '-');
        if (dash) {
            *dash = '\0';
            long a = parse_long_strict(tok, "pin-cores", 0);
            long b = parse_long_strict(dash + 1, "pin-cores", 0);
            add_cpu_range(&set, a, b);
            count += (int)(b - a + 1);
        } else {
            long c = parse_long_strict(tok, "pin-cores", 0);
            add_cpu_range(&set, c, c);
            count += 1;
        }
        tok = strtok(NULL, ",");
    }
    if (count <= 0) {
        fprintf(stderr, "Invalid pin-cores list: %s\n", spec);
        free(copy);
        exit(EXIT_FAILURE);
    }

    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        perror("sched_setaffinity");
        fprintf(stderr, "Warning: continuing without enforced CPU affinity.\n");
    }
    free(copy);
}
#else
static void apply_affinity_if_requested(const char *spec) {
    if (spec) fprintf(stderr, "Warning: CPU affinity is not implemented on this platform.\n");
}
#endif

static void set_thread_count(int threads) {
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
#else
    (void)threads;
#endif
}

static int actual_threads_used(int requested) {
    set_thread_count(requested);
    int actual = 1;
#ifdef _OPENMP
#pragma omp parallel
    {
#pragma omp master
        actual = omp_get_num_threads();
    }
#else
    (void)requested;
#endif
    return actual;
}

static void print_file_first_line(const char *label, const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) return;
    char buf[256];
    if (fgets(buf, sizeof(buf), fp)) {
        buf[strcspn(buf, "\r\n")] = '\0';
        printf("%s: %s\n", label, buf);
    }
    fclose(fp);
}

static void print_cpu_governors(void) {
#ifdef __linux__
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 0) n = 0;
    long limit = n < 8 ? n : 8;
    for (long i = 0; i < limit; ++i) {
        char path[256], label[64];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%ld/cpufreq/scaling_governor", i);
        snprintf(label, sizeof(label), "CPU governor cpu%ld", i);
        print_file_first_line(label, path);
    }
#else
    (void)print_file_first_line;
#endif
}

static void print_metadata(const bench_config_t *cfg) {
    printf("PB-SDVSS OpenSSL multi-core benchmark metadata\n");
    printf("============================================================\n");
    printf("Benchmark mode: OpenMP multi-threaded process\n");
#ifdef _OPENMP
    printf("OpenMP: enabled, _OPENMP=%d\n", _OPENMP);
#else
    printf("OpenMP: not enabled; running single-threaded\n");
#endif
    printf("Requested threads: %d\n", cfg->threads);
    printf("Actual OpenMP threads observed: %d\n", actual_threads_used(cfg->threads));
    printf("Requested CPU affinity: %s\n", cfg->pin_cores ? cfg->pin_cores : "not requested");
    printf("Timing method: clock_gettime(CLOCK_MONOTONIC), wall-clock seconds\n");
    printf("Message length: %d bytes\n", MSG_LEN);
    printf("Info length: %d bytes\n", INFO_LEN);
    printf("Iterations per repeated run: %zu\n", cfg->iterations);
    printf("Warm-up operations per benchmark: %zu\n", cfg->warmup);
    printf("Repeated timed runs: %d\n", cfg->repetitions);
    printf("Timed region Sign: PB-SDVSS.Sign only; message/info generation excluded\n");
    printf("Timed region Verify: PB-SDVSS.Verify only; message/info/signature generation excluded\n");
    printf("Verification datasets: A valid-only; B hard-invalid sig2-corrupted; C mixed 50/50\n");
    printf("Compiler flags recorded: %s\n", COMPILE_FLAGS);
    printf("OpenSSL: %s\n", OPENSSL_VERSION_TEXT);
#if defined(__GNUC__)
    printf("Compiler: GCC-compatible %d.%d.%d\n", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
    printf("Compiler: unknown\n");
#endif
    long cores = sysconf(_SC_NPROCESSORS_ONLN);
    if (cores > 0) printf("CPU online cores: %ld\n", cores);
    struct utsname uts;
    if (uname(&uts) == 0) {
        printf("System: %s %s %s %s\n", uts.sysname, uts.release, uts.machine, uts.nodename);
    }
    const char *omp_num_threads = getenv("OMP_NUM_THREADS");
    const char *omp_proc_bind = getenv("OMP_PROC_BIND");
    const char *omp_places = getenv("OMP_PLACES");
    printf("OMP_NUM_THREADS: %s\n", omp_num_threads ? omp_num_threads : "not set");
    printf("OMP_PROC_BIND: %s\n", omp_proc_bind ? omp_proc_bind : "not set");
    printf("OMP_PLACES: %s\n", omp_places ? omp_places : "not set");
    print_cpu_governors();
    printf("============================================================\n\n");
}

static void random_bytes(unsigned char *buf, size_t len) {
    FILE *fp = fopen("/dev/urandom", "rb");
    if (!fp) { perror("/dev/urandom"); exit(EXIT_FAILURE); }
    if (fread(buf, 1, len, fp) != len) {
        perror("fread /dev/urandom");
        fclose(fp);
        exit(EXIT_FAILURE);
    }
    fclose(fp);
}

static int hash_m_info_xstar(const unsigned char *msg, size_t msg_len,
                             const unsigned char *info, size_t info_len,
                             const EC_POINT *x_star,
                             const EC_GROUP *group,
                             const BIGNUM *order,
                             BIGNUM *out,
                             BN_CTX *ctx) {
    int ret = 0;
    BIGNUM *x = BN_new(); CHECK(x, "BN_new x");

    int field_bits = EC_GROUP_get_degree(group);
    int field_bytes = (field_bits + 7) / 8;
    unsigned char *xb = OPENSSL_malloc((size_t)field_bytes);
    CHECK(xb != NULL, "OPENSSL_malloc xb");

    size_t total = msg_len + info_len + (size_t)field_bytes;
    unsigned char *tmp = OPENSSL_malloc(total);
    CHECK(tmp != NULL, "OPENSSL_malloc hash tmp");

    unsigned char hash[SHA256_DIGEST_LENGTH];

    if (!EC_POINT_get_affine_coordinates(group, x_star, x, NULL, ctx)) goto end;
    if (BN_bn2binpad(x, xb, field_bytes) < 0) goto end;

    memcpy(tmp, msg, msg_len);
    memcpy(tmp + msg_len, info, info_len);
    memcpy(tmp + msg_len + info_len, xb, (size_t)field_bytes);
    SHA256(tmp, total, hash);

    BN_bin2bn(hash, SHA256_DIGEST_LENGTH, out);
    if (!BN_mod(out, out, order, ctx)) goto end;
    ret = 1;

end:
    OPENSSL_free(tmp);
    OPENSSL_free(xb);
    BN_free(x);
    return ret;
}

static int hash_info(const unsigned char *info, size_t info_len,
                     const BIGNUM *order, BIGNUM *out, BN_CTX *ctx) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(info, info_len, hash);
    BN_bin2bn(hash, SHA256_DIGEST_LENGTH, out);
    return BN_mod(out, out, order, ctx);
}

typedef struct {
    BIGNUM *s1, *s2;
    BIGNUM *s;
    EC_POINT *g1;
    EC_POINT *g2;
    EC_POINT *v;
    EC_POINT *k;
} pb_keys_t;

typedef struct {
    EC_POINT *x_star;
    BIGNUM *a_star;
    EC_POINT *sig1;
    BIGNUM *sig2;
} pb_signature_t;

typedef struct {
    unsigned char msg[MSG_LEN];
    unsigned char info[INFO_LEN];
    pb_signature_t sig;
    int expected_valid;
} bench_item_t;

static void pb_keys_init(pb_keys_t *keys, const EC_GROUP *group) {
    keys->s1 = BN_new();
    keys->s2 = BN_new();
    keys->s = BN_new();
    keys->g1 = EC_POINT_new(group);
    keys->g2 = EC_POINT_new(group);
    keys->v = EC_POINT_new(group);
    keys->k = EC_POINT_new(group);
    CHECK(keys->s1 && keys->s2 && keys->s && keys->g1 && keys->g2 && keys->v && keys->k,
          "alloc keys");
}

static void pb_keys_free(pb_keys_t *keys) {
    if (keys->s1) BN_free(keys->s1);
    if (keys->s2) BN_free(keys->s2);
    if (keys->s) BN_free(keys->s);
    if (keys->g1) EC_POINT_free(keys->g1);
    if (keys->g2) EC_POINT_free(keys->g2);
    if (keys->v) EC_POINT_free(keys->v);
    if (keys->k) EC_POINT_free(keys->k);
}

static void pb_sig_init(pb_signature_t *sig, const EC_GROUP *group) {
    sig->x_star = EC_POINT_new(group);
    sig->a_star = BN_new();
    sig->sig1 = EC_POINT_new(group);
    sig->sig2 = BN_new();
    CHECK(sig->x_star && sig->a_star && sig->sig1 && sig->sig2, "alloc sig");
}

static void pb_sig_free(pb_signature_t *sig) {
    if (sig->x_star) EC_POINT_free(sig->x_star);
    if (sig->a_star) BN_free(sig->a_star);
    if (sig->sig1) EC_POINT_free(sig->sig1);
    if (sig->sig2) BN_free(sig->sig2);
}

static void pb_sig_copy(pb_signature_t *dst, const pb_signature_t *src) {
    CHECK(EC_POINT_copy(dst->x_star, src->x_star) == 1, "copy x_star");
    CHECK(BN_copy(dst->a_star, src->a_star) != NULL, "copy a_star");
    CHECK(EC_POINT_copy(dst->sig1, src->sig1) == 1, "copy sig1");
    CHECK(BN_copy(dst->sig2, src->sig2) != NULL, "copy sig2");
}

static bench_item_t *dataset_alloc(size_t n, const EC_GROUP *group) {
    bench_item_t *items = calloc(n, sizeof(*items));
    CHECK(items != NULL, "calloc dataset");
    for (size_t i = 0; i < n; ++i) {
        pb_sig_init(&items[i].sig, group);
        items[i].expected_valid = 0;
    }
    return items;
}

static void dataset_free(bench_item_t *items, size_t n) {
    if (!items) return;
    for (size_t i = 0; i < n; ++i) pb_sig_free(&items[i].sig);
    free(items);
}

static void pb_setup_keys(const EC_GROUP *group, const BIGNUM *order,
                          pb_keys_t *keys, BN_CTX *ctx) {
    pb_keys_init(keys, group);
    const EC_POINT *G = EC_GROUP_get0_generator(group);
    CHECK(EC_POINT_copy(keys->g1, G) == 1, "g1 = G");

    const unsigned char label[] = "g2";
    unsigned char hbuf[SHA256_DIGEST_LENGTH];
    SHA256(label, sizeof(label) - 1, hbuf);

    BIGNUM *h = BN_new(); CHECK(h, "BN_new h");
    BN_bin2bn(hbuf, sizeof(hbuf), h);
    BN_mod(h, h, order, ctx);
    if (BN_is_zero(h)) BN_one(h);

    CHECK(EC_POINT_mul(group, keys->g2, NULL, G, h, ctx) == 1, "g2 = h*G");

    BN_rand_range(keys->s1, order);
    BN_rand_range(keys->s2, order);
    BN_rand_range(keys->s, order);

    BIGNUM *neg1 = BN_new();
    BIGNUM *neg2 = BN_new();
    EC_POINT *t1 = EC_POINT_new(group);
    EC_POINT *t2 = EC_POINT_new(group);
    CHECK(neg1 && neg2 && t1 && t2, "alloc v");

    BN_sub(neg1, order, keys->s1);
    BN_sub(neg2, order, keys->s2);
    EC_POINT_mul(group, t1, NULL, keys->g1, neg1, ctx);
    EC_POINT_mul(group, t2, NULL, keys->g2, neg2, ctx);
    EC_POINT_add(group, keys->v, t1, t2, ctx);

    EC_POINT_mul(group, keys->k, NULL, keys->g1, keys->s, ctx);

    BN_free(h);
    BN_free(neg1);
    BN_free(neg2);
    EC_POINT_free(t1);
    EC_POINT_free(t2);
}

static int pb_sign(const unsigned char *m, size_t mlen,
                   const unsigned char *info, size_t ilen,
                   const EC_GROUP *group, const BIGNUM *order,
                   const pb_keys_t *keys, pb_signature_t *sig,
                   BN_CTX *ctx) {
    int ok = 0;
    BIGNUM *r1 = BN_new(), *r2 = BN_new();
    BIGNUM *u1 = BN_new(), *u2 = BN_new(), *d = BN_new();
    BIGNUM *z = BN_new(), *alpha_bar = BN_new(), *alpha = BN_new();
    BIGNUM *y1 = BN_new(), *y2 = BN_new(), *tmp = BN_new();
    EC_POINT *x = EC_POINT_new(group);
    EC_POINT *t1 = EC_POINT_new(group);
    EC_POINT *t2 = EC_POINT_new(group);
    EC_POINT *t3 = EC_POINT_new(group);
    EC_POINT *bsig1 = EC_POINT_new(group);
    BIGNUM *bsig2 = BN_new();

    CHECK(r1 && r2 && u1 && u2 && d && z && alpha_bar && alpha && y1 && y2 && tmp &&
          x && t1 && t2 && t3 && bsig1 && bsig2, "alloc sign");

    BN_rand_range(r1, order);
    BN_rand_range(r2, order);

    EC_POINT_mul(group, t1, NULL, keys->g1, r1, ctx);
    EC_POINT_mul(group, t2, NULL, keys->g2, r2, ctx);
    EC_POINT_add(group, x, t1, t2, ctx);

    BN_rand_range(u1, order);
    BN_rand_range(u2, order);
    BN_rand_range(d, order);

    EC_POINT_mul(group, t1, NULL, keys->g1, u1, ctx);
    EC_POINT_mul(group, t2, NULL, keys->g2, u2, ctx);
    EC_POINT_mul(group, t3, NULL, keys->v, d, ctx);

    EC_POINT_copy(sig->x_star, x);
    EC_POINT_add(group, sig->x_star, sig->x_star, t1, ctx);
    EC_POINT_add(group, sig->x_star, sig->x_star, t2, ctx);
    EC_POINT_add(group, sig->x_star, sig->x_star, t3, ctx);

    CHECK(hash_m_info_xstar(m, mlen, info, ilen, sig->x_star, group, order, sig->a_star, ctx),
          "hash a*");
    CHECK(hash_info(info, ilen, order, z, ctx) == 1, "hash z");

    BN_mod_sub(tmp, sig->a_star, d, order, ctx);
    BN_mod_add(alpha_bar, tmp, z, order, ctx);
    BN_mod_sub(alpha, alpha_bar, z, order, ctx);

    BN_mod_mul(tmp, alpha, keys->s1, order, ctx);
    BN_mod_add(y1, r1, tmp, order, ctx);
    BN_mod_mul(tmp, alpha, keys->s2, order, ctx);
    BN_mod_add(y2, r2, tmp, order, ctx);

    EC_POINT_mul(group, bsig1, NULL, keys->k, y1, ctx);
    BN_copy(bsig2, y2);

    EC_POINT_mul(group, t1, NULL, keys->k, u1, ctx);
    EC_POINT_add(group, sig->sig1, bsig1, t1, ctx);
    BN_mod_add(sig->sig2, bsig2, u2, order, ctx);

    ok = 1;

    BN_free(r1); BN_free(r2); BN_free(u1); BN_free(u2); BN_free(d);
    BN_free(z); BN_free(alpha_bar); BN_free(alpha); BN_free(y1); BN_free(y2); BN_free(tmp);
    EC_POINT_free(x); EC_POINT_free(t1); EC_POINT_free(t2); EC_POINT_free(t3);
    EC_POINT_free(bsig1); BN_free(bsig2);
    return ok;
}

static int pb_verify(const unsigned char *m, size_t mlen,
                     const unsigned char *info, size_t ilen,
                     const EC_GROUP *group, const BIGNUM *order,
                     const pb_keys_t *keys, const pb_signature_t *sig,
                     BN_CTX *ctx) {
    int ok = 0;
    BIGNUM *a2 = BN_new();
    BIGNUM *sig2s = BN_new();
    BIGNUM *as = BN_new();
    CHECK(a2 && sig2s && as, "alloc verify BN");

    CHECK(hash_m_info_xstar(m, mlen, info, ilen, sig->x_star, group, order, a2, ctx),
          "hash a2");
    if (BN_cmp(a2, sig->a_star) != 0) {
        ok = 0;
        goto end;
    }

    EC_POINT *lhs = EC_POINT_new(group);
    EC_POINT *rhs = EC_POINT_new(group);
    EC_POINT *t1 = EC_POINT_new(group);
    EC_POINT *t2 = EC_POINT_new(group);
    CHECK(lhs && rhs && t1 && t2, "alloc verify points");

    EC_POINT_mul(group, lhs, NULL, sig->x_star, keys->s, ctx);
    BN_mod_mul(sig2s, sig->sig2, keys->s, order, ctx);
    BN_mod_mul(as, sig->a_star, keys->s, order, ctx);
    EC_POINT_mul(group, t1, NULL, keys->g2, sig2s, ctx);
    EC_POINT_mul(group, t2, NULL, keys->v, as, ctx);
    EC_POINT_copy(rhs, sig->sig1);
    EC_POINT_add(group, rhs, rhs, t1, ctx);
    EC_POINT_add(group, rhs, rhs, t2, ctx);
    ok = (EC_POINT_cmp(group, lhs, rhs, ctx) == 0);

    EC_POINT_free(lhs);
    EC_POINT_free(rhs);
    EC_POINT_free(t1);
    EC_POINT_free(t2);

end:
    BN_free(a2);
    BN_free(sig2s);
    BN_free(as);
    return ok;
}

static void print_sizes(const EC_GROUP *group,
                        const BIGNUM *order,
                        const pb_keys_t *keys,
                        const pb_signature_t *sig,
                        BN_CTX *ctx) {
    int scalar_bytes = BN_num_bytes(order);
    int scalar_bits = scalar_bytes * 8;
    size_t v_bytes = EC_POINT_point2oct(group, keys->v, POINT_CONVERSION_COMPRESSED, NULL, 0, ctx);
    size_t k_bytes = EC_POINT_point2oct(group, keys->k, POINT_CONVERSION_COMPRESSED, NULL, 0, ctx);
    size_t x_bytes = EC_POINT_point2oct(group, sig->x_star, POINT_CONVERSION_COMPRESSED, NULL, 0, ctx);
    size_t s1_bytes = EC_POINT_point2oct(group, sig->sig1, POINT_CONVERSION_COMPRESSED, NULL, 0, ctx);
    size_t sig_bytes = x_bytes + (size_t)scalar_bytes + s1_bytes + (size_t)scalar_bytes;

    printf("  Key / Signature sizes, compressed points:\n");
    printf("    scalar size:           %d bytes (%d bits)\n", scalar_bytes, scalar_bits);
    printf("    sk_S = (s1,s2):        %d bytes (%d bits)\n", 2 * scalar_bytes, 2 * scalar_bits);
    printf("    sk_V = s:              %d bytes (%d bits)\n", scalar_bytes, scalar_bits);
    printf("    pk_S = v:              %zu bytes (%zu bits)\n", v_bytes, v_bytes * 8);
    printf("    pk_V = k:              %zu bytes (%zu bits)\n", k_bytes, k_bytes * 8);
    printf("    signature total:       %zu bytes (%zu bits)\n", sig_bytes, sig_bytes * 8);
    printf("      x*:                  %zu bytes\n", x_bytes);
    printf("      a*:                  %d bytes\n", scalar_bytes);
    printf("      sig1:                %zu bytes\n", s1_bytes);
    printf("      sig2:                %d bytes\n", scalar_bytes);
}

static void corrupt_signature_hard(pb_signature_t *sig,
                                   const BIGNUM *order,
                                   BN_CTX *ctx) {
    BIGNUM *one = BN_new(); CHECK(one, "BN_new one");
    BN_one(one);
    CHECK(BN_mod_add(sig->sig2, sig->sig2, one, order, ctx) == 1, "corrupt sig2");
    BN_free(one);
}

static void build_datasets(size_t n,
                           const EC_GROUP *group,
                           const BIGNUM *order,
                           const pb_keys_t *keys,
                           bench_item_t *valid,
                           bench_item_t *invalid,
                           bench_item_t *mixed,
                           BN_CTX *ctx) {
    printf("  Building datasets A/B/C outside timed region...\n");
    for (size_t i = 0; i < n; ++i) {
        random_bytes(valid[i].msg, MSG_LEN);
        random_bytes(valid[i].info, INFO_LEN);
        CHECK(pb_sign(valid[i].msg, MSG_LEN, valid[i].info, INFO_LEN,
                      group, order, keys, &valid[i].sig, ctx), "dataset sign valid");
        CHECK(pb_verify(valid[i].msg, MSG_LEN, valid[i].info, INFO_LEN,
                        group, order, keys, &valid[i].sig, ctx), "dataset verify valid");
        valid[i].expected_valid = 1;

        memcpy(invalid[i].msg, valid[i].msg, MSG_LEN);
        memcpy(invalid[i].info, valid[i].info, INFO_LEN);
        pb_sig_copy(&invalid[i].sig, &valid[i].sig);
        corrupt_signature_hard(&invalid[i].sig, order, ctx);
        invalid[i].expected_valid = 0;
        int got = pb_verify(invalid[i].msg, MSG_LEN, invalid[i].info, INFO_LEN,
                            group, order, keys, &invalid[i].sig, ctx);
        CHECK(got == 0, "invalid signature unexpectedly verified");

        if ((i % 2) == 0) {
            memcpy(mixed[i].msg, valid[i].msg, MSG_LEN);
            memcpy(mixed[i].info, valid[i].info, INFO_LEN);
            pb_sig_copy(&mixed[i].sig, &valid[i].sig);
            mixed[i].expected_valid = 1;
        } else {
            memcpy(mixed[i].msg, invalid[i].msg, MSG_LEN);
            memcpy(mixed[i].info, invalid[i].info, INFO_LEN);
            pb_sig_copy(&mixed[i].sig, &invalid[i].sig);
            mixed[i].expected_valid = 0;
        }
    }
    printf("    A: %zu valid signatures\n", n);
    printf("    B: %zu hard-invalid signatures, sig2 corrupted\n", n);
    printf("    C: %zu mixed signatures, 50%% valid / 50%% hard-invalid\n", n);
}

typedef struct {
    double elapsed;
    long failures;
    long accepted;
    long rejected;
    long mismatches;
} run_result_t;

static run_result_t signing_run_once(size_t n,
                                     int threads,
                                     const EC_GROUP *group,
                                     const BIGNUM *order,
                                     const pb_keys_t *keys,
                                     const bench_item_t *valid_msgs) {
    set_thread_count(threads);
    long failures = 0;
    double t0 = now_seconds();

#ifdef _OPENMP
#pragma omp parallel reduction(+:failures)
#endif
    {
        BN_CTX *lctx = BN_CTX_new(); CHECK(lctx, "thread BN_CTX sign");
        pb_signature_t local_sig;
        pb_sig_init(&local_sig, group);

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (size_t i = 0; i < n; ++i) {
            if (!pb_sign(valid_msgs[i].msg, MSG_LEN, valid_msgs[i].info, INFO_LEN,
                         group, order, keys, &local_sig, lctx)) {
                failures += 1;
            }
        }
        pb_sig_free(&local_sig);
        BN_CTX_free(lctx);
    }

    double t1 = now_seconds();
    run_result_t r;
    r.elapsed = t1 - t0;
    if (r.elapsed <= 0.0) r.elapsed = 1e-9;
    r.failures = failures;
    r.accepted = r.rejected = r.mismatches = 0;
    return r;
}

static run_result_t verification_run_once(size_t n,
                                          int threads,
                                          const EC_GROUP *group,
                                          const BIGNUM *order,
                                          const pb_keys_t *keys,
                                          const bench_item_t *items) {
    set_thread_count(threads);
    long accepted = 0;
    long rejected = 0;
    long mismatches = 0;
    double t0 = now_seconds();

#ifdef _OPENMP
#pragma omp parallel reduction(+:accepted,rejected,mismatches)
#endif
    {
        BN_CTX *lctx = BN_CTX_new(); CHECK(lctx, "thread BN_CTX verify");
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (size_t i = 0; i < n; ++i) {
            int ok = pb_verify(items[i].msg, MSG_LEN, items[i].info, INFO_LEN,
                               group, order, keys, &items[i].sig, lctx);
            if (ok) accepted += 1;
            else rejected += 1;
            if (ok != items[i].expected_valid) mismatches += 1;
        }
        BN_CTX_free(lctx);
    }

    double t1 = now_seconds();
    run_result_t r;
    r.elapsed = t1 - t0;
    if (r.elapsed <= 0.0) r.elapsed = 1e-9;
    r.failures = 0;
    r.accepted = accepted;
    r.rejected = rejected;
    r.mismatches = mismatches;
    return r;
}

static void warmup_sign(size_t warmup,
                        int threads,
                        const EC_GROUP *group,
                        const BIGNUM *order,
                        const pb_keys_t *keys,
                        const bench_item_t *valid_msgs,
                        size_t n) {
    if (warmup == 0) return;
    size_t w = warmup;
    set_thread_count(threads);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        BN_CTX *lctx = BN_CTX_new(); CHECK(lctx, "warmup BN_CTX sign");
        pb_signature_t local_sig;
        pb_sig_init(&local_sig, group);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (size_t i = 0; i < w; ++i) {
            size_t j = i % n;
            (void)pb_sign(valid_msgs[j].msg, MSG_LEN, valid_msgs[j].info, INFO_LEN,
                          group, order, keys, &local_sig, lctx);
        }
        pb_sig_free(&local_sig);
        BN_CTX_free(lctx);
    }
}

static void warmup_verify(size_t warmup,
                          int threads,
                          const EC_GROUP *group,
                          const BIGNUM *order,
                          const pb_keys_t *keys,
                          const bench_item_t *items,
                          size_t n) {
    if (warmup == 0) return;
    size_t w = warmup;
    set_thread_count(threads);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        BN_CTX *lctx = BN_CTX_new(); CHECK(lctx, "warmup BN_CTX verify");
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (size_t i = 0; i < w; ++i) {
            size_t j = i % n;
            (void)pb_verify(items[j].msg, MSG_LEN, items[j].info, INFO_LEN,
                            group, order, keys, &items[j].sig, lctx);
        }
        BN_CTX_free(lctx);
    }
}

typedef struct {
    const char *operation;
    const char *dataset;
    double mean_us;
    double median_us;
    double stddev_us;
    double throughput;
    long failures;
    long accepted;
    long rejected;
    long mismatches;
} summary_t;

static summary_t summarize_runs(const char *operation,
                                const char *dataset,
                                const run_result_t *runs,
                                int repetitions,
                                size_t n) {
    double *us = (double *)malloc((size_t)repetitions * sizeof(double));
    CHECK(us != NULL, "malloc us summary");
    for (int r = 0; r < repetitions; ++r) {
        us[r] = runs[r].elapsed * 1e6 / (double)n;
    }
    double mean_us = mean_double(us, repetitions);
    summary_t s;
    s.operation = operation;
    s.dataset = dataset;
    s.mean_us = mean_us;
    s.median_us = median_double(us, repetitions);
    s.stddev_us = stddev_double(us, repetitions);
    s.throughput = 1e6 / (mean_us > 0.0 ? mean_us : 1e-9);
    s.failures = runs[repetitions - 1].failures;
    s.accepted = runs[repetitions - 1].accepted;
    s.rejected = runs[repetitions - 1].rejected;
    s.mismatches = runs[repetitions - 1].mismatches;
    free(us);
    return s;
}

static void print_summary(const summary_t *s, int threads, size_t n, int repetitions) {
    printf("  %s%s%s, multi-threaded:\n",
           s->operation,
           s->dataset && s->dataset[0] ? " dataset " : "",
           s->dataset && s->dataset[0] ? s->dataset : "");
    printf("    threads:     %d\n", threads);
    printf("    operations:  %zu per repeated run\n", n);
    printf("    mean:        %.3f us/op\n", s->mean_us);
    printf("    median:      %.3f us/op\n", s->median_us);
    printf("    std. dev.:   %.3f us/op over %d repeated runs\n", s->stddev_us, repetitions);
    printf("    throughput:  %.2f ops/sec, based on mean\n", s->throughput);
    if (strcmp(s->operation, "PB-SDVSS.Sign") == 0) {
        printf("    failures:    %ld\n\n", s->failures);
    } else {
        printf("    accepted:    %ld\n", s->accepted);
        printf("    rejected:    %ld\n", s->rejected);
        printf("    mismatches:  %ld\n\n", s->mismatches);
    }
}

static void csv_write_header(FILE *csv) {
    if (!csv) return;
    fprintf(csv,
            "implementation,curve_tag,curve_name,threads,iterations,warmup,repetitions,operation,dataset,mean_us_op,median_us_op,stddev_us_op,throughput_ops_s,failures,accepted,rejected,mismatches\n");
}

static void csv_write_summary(FILE *csv,
                              const char *curve_tag,
                              const char *curve_name,
                              int threads,
                              size_t iterations,
                              size_t warmup,
                              int repetitions,
                              const summary_t *s) {
    if (!csv) return;
    fprintf(csv,
            "PB-SDVSS-OpenSSL-OpenMP,%s,%s,%d,%zu,%zu,%d,%s,%s,%.6f,%.6f,%.6f,%.6f,%ld,%ld,%ld,%ld\n",
            curve_tag, curve_name, threads, iterations, warmup, repetitions,
            s->operation, s->dataset ? s->dataset : "",
            s->mean_us, s->median_us, s->stddev_us, s->throughput,
            s->failures, s->accepted, s->rejected, s->mismatches);
}

typedef struct {
    const char *tag;
    const char *name;
    int nid;
} curve_def_t;

static const curve_def_t CURVES[] = {
    {"p192",  "prime192v1 (NIST P-192)", NID_X9_62_prime192v1},
    {"p224",  "secp224r1 (NIST P-224)", NID_secp224r1},
    {"p256",  "prime256v1 (NIST P-256)", NID_X9_62_prime256v1},
    {"k256",  "secp256k1 (Bitcoin)", NID_secp256k1},
    {"p384",  "secp384r1 (NIST P-384)", NID_secp384r1},
    {"p521",  "secp521r1 (NIST P-521)", NID_secp521r1},
    {"bp256", "brainpoolP256r1", NID_brainpoolP256r1},
    {"bp384", "brainpoolP384r1", NID_brainpoolP384r1}
};

static size_t curve_count(void) {
    return sizeof(CURVES) / sizeof(CURVES[0]);
}

static const curve_def_t *find_curve(const char *tag) {
    for (size_t i = 0; i < curve_count(); ++i) {
        if (strcmp(tag, CURVES[i].tag) == 0) return &CURVES[i];
    }
    return NULL;
}

static void benchmark_curve(const curve_def_t *curve, const bench_config_t *cfg, FILE *csv) {
    printf("============================================================\n");
    printf("Curve: %s [%s], NID=%d\n", curve->name, curve->tag, curve->nid);
    printf("Requested OpenMP threads: %d; actual observed: %d\n", cfg->threads, actual_threads_used(cfg->threads));
    printf("Dataset size / timed operations: %zu\n", cfg->iterations);
    printf("============================================================\n");

    BN_CTX *ctx = BN_CTX_new(); CHECK(ctx, "BN_CTX_new");
    EC_GROUP *group = EC_GROUP_new_by_curve_name(curve->nid);
    CHECK(group != NULL, "EC_GROUP_new_by_curve_name");
    BIGNUM *order = BN_new(); CHECK(order != NULL, "BN_new order");
    CHECK(EC_GROUP_get_order(group, order, ctx) == 1, "get_order");

    pb_keys_t keys;
    pb_setup_keys(group, order, &keys, ctx);

    bench_item_t *A_valid = dataset_alloc(cfg->iterations, group);
    bench_item_t *B_invalid = dataset_alloc(cfg->iterations, group);
    bench_item_t *C_mixed = dataset_alloc(cfg->iterations, group);
    build_datasets(cfg->iterations, group, order, &keys, A_valid, B_invalid, C_mixed, ctx);

    print_sizes(group, order, &keys, &A_valid[0].sig, ctx);
    printf("\n");

    run_result_t *runs = (run_result_t *)malloc((size_t)cfg->repetitions * sizeof(run_result_t));
    CHECK(runs != NULL, "malloc runs");

    printf("  Warm-up: PB-SDVSS.Sign, %zu operations outside timed region...\n", cfg->warmup);
    warmup_sign(cfg->warmup, cfg->threads, group, order, &keys, A_valid, cfg->iterations);
    for (int r = 0; r < cfg->repetitions; ++r) {
        runs[r] = signing_run_once(cfg->iterations, cfg->threads, group, order, &keys, A_valid);
    }
    summary_t sign_sum = summarize_runs("PB-SDVSS.Sign", "", runs, cfg->repetitions, cfg->iterations);
    print_summary(&sign_sum, cfg->threads, cfg->iterations, cfg->repetitions);
    csv_write_summary(csv, curve->tag, curve->name, cfg->threads, cfg->iterations, cfg->warmup, cfg->repetitions, &sign_sum);

    struct {
        const char *label;
        bench_item_t *items;
    } datasets[] = {
        {"A-valid", A_valid},
        {"B-invalid", B_invalid},
        {"C-mixed", C_mixed}
    };

    for (size_t d = 0; d < sizeof(datasets) / sizeof(datasets[0]); ++d) {
        printf("  Warm-up: PB-SDVSS.Verify dataset %s, %zu operations outside timed region...\n",
               datasets[d].label, cfg->warmup);
        warmup_verify(cfg->warmup, cfg->threads, group, order, &keys,
                      datasets[d].items, cfg->iterations);
        for (int r = 0; r < cfg->repetitions; ++r) {
            runs[r] = verification_run_once(cfg->iterations, cfg->threads, group, order, &keys, datasets[d].items);
        }
        summary_t ver_sum = summarize_runs("PB-SDVSS.Verify", datasets[d].label, runs, cfg->repetitions, cfg->iterations);
        print_summary(&ver_sum, cfg->threads, cfg->iterations, cfg->repetitions);
        csv_write_summary(csv, curve->tag, curve->name, cfg->threads, cfg->iterations, cfg->warmup, cfg->repetitions, &ver_sum);
    }

    free(runs);
    dataset_free(A_valid, cfg->iterations);
    dataset_free(B_invalid, cfg->iterations);
    dataset_free(C_mixed, cfg->iterations);
    pb_keys_free(&keys);
    BN_free(order);
    EC_GROUP_free(group);
    BN_CTX_free(ctx);
}

int main(int argc, char **argv) {
    bench_config_t cfg = parse_args(argc, argv);
    apply_affinity_if_requested(cfg.pin_cores);
    set_thread_count(cfg.threads);
    print_metadata(&cfg);

    FILE *csv = NULL;
    if (cfg.csv_path) {
        csv = fopen(cfg.csv_path, "w");
        if (!csv) {
            perror("fopen csv");
            return EXIT_FAILURE;
        }
        csv_write_header(csv);
    }

    if (strcmp(cfg.curve_tag, "all") == 0) {
        for (size_t i = 0; i < curve_count(); ++i) {
            benchmark_curve(&CURVES[i], &cfg, csv);
        }
    } else {
        const curve_def_t *curve = find_curve(cfg.curve_tag);
        if (!curve) {
            usage(argv[0]);
            if (csv) fclose(csv);
            return EXIT_FAILURE;
        }
        benchmark_curve(curve, &cfg, csv);
    }

    if (csv) {
        fclose(csv);
        printf("CSV results written to: %s\n", cfg.csv_path);
    }
    return EXIT_SUCCESS;
}
