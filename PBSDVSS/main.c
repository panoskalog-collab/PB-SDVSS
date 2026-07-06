// pb_sdvss_repro_benchmark.c
// PB-SDVSS implementation & reproducible benchmark in C using OpenSSL EC
//
// Improvements over the initial benchmark:
//   * wall-clock timing with clock_gettime(CLOCK_MONOTONIC)
//   * configurable iterations, warm-up iterations, and repeated runs
//   * mean / median / standard deviation reporting
//   * optional CPU pinning for single-core Raspberry Pi experiments
//   * OpenSSL version, compiler flags, CPU count, and governor reporting
//   * message/info generation excluded from timed regions
//   * explicit single-thread benchmark label
//
// Suggested Raspberry Pi compilation:
//   gcc -O3 -Wall -Wextra -D COMPILE_FLAGS='"-O3 -Wall -Wextra"' pb_sdvss_repro_benchmark.c -o pb_sdvss_repro_benchmark -lcrypto -lm
//
// Example run:
//   ./pb_sdvss_repro_benchmark --iterations 1000 --warmup 100 --repetitions 10 --pin-core 0 --csv pb_sdvss_results.csv

#define _GNU_SOURCE

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif

#include <openssl/bn.h>
#include <openssl/crypto.h>
#include <openssl/ec.h>
#include <openssl/evp.h>
#include <openssl/obj_mac.h>
#include <openssl/rand.h>
#include <openssl/sha.h>

#ifndef COMPILE_FLAGS
#define COMPILE_FLAGS "not recorded; compile with -DCOMPILE_FLAGS='\"-O3 -Wall -Wextra\"'"
#endif

#define DEFAULT_ITERATIONS 1000
#define DEFAULT_WARMUP    100
#define DEFAULT_REPETITIONS 10

#define MSG_LEN    200   // VANET-like beacon length
#define INFO_LEN   32    // metadata/info length

#define CHECK(c,msg) do{ if(!(c)){ fprintf(stderr,"Error: %s\n", msg); exit(EXIT_FAILURE);} }while(0)

static double now_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// ---------------------------------------------------------------------
// Random data/scalar helpers
// ---------------------------------------------------------------------

static void random_bytes(unsigned char *buf, size_t len) {
    CHECK(len <= INT32_MAX, "random_bytes length too large for RAND_bytes");
    CHECK(RAND_bytes(buf, (int)len) == 1, "RAND_bytes");
}

static void bn_rand_nonzero_range(BIGNUM *out, const BIGNUM *order) {
    do {
        CHECK(BN_rand_range(out, order) == 1, "BN_rand_range");
    } while (BN_is_zero(out));
}

static void sha256_parts(unsigned char out[SHA256_DIGEST_LENGTH],
                         const unsigned char **parts,
                         const size_t *lens,
                         size_t n_parts) {
    EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
    CHECK(mdctx, "EVP_MD_CTX_new");
    CHECK(EVP_DigestInit_ex(mdctx, EVP_sha256(), NULL) == 1, "EVP_DigestInit_ex");
    for (size_t i = 0; i < n_parts; ++i) {
        CHECK(EVP_DigestUpdate(mdctx, parts[i], lens[i]) == 1, "EVP_DigestUpdate");
    }
    unsigned int out_len = 0;
    CHECK(EVP_DigestFinal_ex(mdctx, out, &out_len) == 1, "EVP_DigestFinal_ex");
    CHECK(out_len == SHA256_DIGEST_LENGTH, "unexpected SHA256 length");
    EVP_MD_CTX_free(mdctx);
}

// ---------------------------------------------------------------------
// Hash helpers
// ---------------------------------------------------------------------

// H(m, info, x*) in Z_q
static int hash_m_info_xstar(const unsigned char *msg,  size_t msg_len,
                             const unsigned char *info, size_t info_len,
                             const EC_POINT *x_star,
                             const EC_GROUP *group,
                             const BIGNUM *order,
                             BIGNUM *out,
                             BN_CTX *ctx) {
    int ret = 0;
    BIGNUM *x = BN_new(); CHECK(x, "BN_new x");

    int field_bits  = EC_GROUP_get_degree(group);
    int field_bytes = (field_bits + 7) / 8;

    unsigned char *buf = OPENSSL_malloc((size_t)field_bytes);
    CHECK(buf, "OPENSSL_malloc");

    unsigned char hash[SHA256_DIGEST_LENGTH];

    if (!EC_POINT_get_affine_coordinates(group, x_star, x, NULL, ctx))
        goto end;
    if (BN_bn2binpad(x, buf, field_bytes) < 0)
        goto end;

    const unsigned char *parts[3] = { msg, info, buf };
    size_t lens[3] = { msg_len, info_len, (size_t)field_bytes };
    sha256_parts(hash, parts, lens, 3);

    BN_bin2bn(hash, SHA256_DIGEST_LENGTH, out);
    if (!BN_mod(out, out, order, ctx)) goto end;

    ret = 1;

end:
    OPENSSL_free(buf);
    BN_free(x);
    return ret;
}

// H(info) in Z_q
static int hash_info(const unsigned char *info, size_t info_len,
                     const BIGNUM *order, BIGNUM *out, BN_CTX *ctx) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    const unsigned char *parts[1] = { info };
    size_t lens[1] = { info_len };
    sha256_parts(hash, parts, lens, 1);

    BN_bin2bn(hash, SHA256_DIGEST_LENGTH, out);
    return BN_mod(out, out, order, ctx);
}

// ---------------------------------------------------------------------
// Key & signature structures
// ---------------------------------------------------------------------

typedef struct {
    // Signer secret
    BIGNUM *s1, *s2;
    // Designated verifier secret
    BIGNUM *s;
    // Public params / keys
    EC_POINT *g1;
    EC_POINT *g2;
    EC_POINT *v;   // pkS
    EC_POINT *k;   // pkV
} pb_keys_t;

typedef struct {
    EC_POINT *x_star; // x*
    BIGNUM   *a_star; // a*
    EC_POINT *sig1;   // sig1
    BIGNUM   *sig2;   // sig2
} pb_signature_t;

// ---------------------------------------------------------------------
// Allocation helpers
// ---------------------------------------------------------------------

static void pb_keys_init(pb_keys_t *keys, const EC_GROUP *group) {
    keys->s1 = BN_new();
    keys->s2 = BN_new();
    keys->s  = BN_new();
    keys->g1 = EC_POINT_new(group);
    keys->g2 = EC_POINT_new(group);
    keys->v  = EC_POINT_new(group);
    keys->k  = EC_POINT_new(group);
    CHECK(keys->s1 && keys->s2 && keys->s &&
          keys->g1 && keys->g2 && keys->v && keys->k,
          "alloc keys");
}

static void pb_keys_free(pb_keys_t *keys) {
    if (keys->s1) BN_free(keys->s1);
    if (keys->s2) BN_free(keys->s2);
    if (keys->s)  BN_free(keys->s);
    if (keys->g1) EC_POINT_free(keys->g1);
    if (keys->g2) EC_POINT_free(keys->g2);
    if (keys->v)  EC_POINT_free(keys->v);
    if (keys->k)  EC_POINT_free(keys->k);
}

static void pb_sig_init(pb_signature_t *sig, const EC_GROUP *group) {
    sig->x_star = EC_POINT_new(group);
    sig->a_star = BN_new();
    sig->sig1   = EC_POINT_new(group);
    sig->sig2   = BN_new();
    CHECK(sig->x_star && sig->a_star && sig->sig1 && sig->sig2,
          "alloc sig");
}

static void pb_sig_free(pb_signature_t *sig) {
    if (sig->x_star) EC_POINT_free(sig->x_star);
    if (sig->a_star) BN_free(sig->a_star);
    if (sig->sig1)   EC_POINT_free(sig->sig1);
    if (sig->sig2)   BN_free(sig->sig2);
}

// ---------------------------------------------------------------------
// Setup keys (pp, skS, skV, pkS, pkV)
// ---------------------------------------------------------------------

static void pb_setup_keys(const EC_GROUP *group, const BIGNUM *order,
                          pb_keys_t *keys, BN_CTX *ctx) {
    pb_keys_init(keys, group);

    const EC_POINT *G = EC_GROUP_get0_generator(group);
    CHECK(EC_POINT_copy(keys->g1, G) == 1, "g1 = G");

    // g2 = h * G, with h = SHA256("g2") mod q
    unsigned char hbuf[SHA256_DIGEST_LENGTH];
    const unsigned char label[] = "g2";
    const unsigned char *parts[1] = { label };
    size_t lens[1] = { sizeof(label) - 1 };
    sha256_parts(hbuf, parts, lens, 1);

    BIGNUM *h = BN_new(); CHECK(h, "BN_new h");
    BN_bin2bn(hbuf, sizeof(hbuf), h);
    BN_mod(h, h, order, ctx);
    if (BN_is_zero(h)) BN_one(h);

    CHECK(EC_POINT_mul(group, keys->g2, NULL, G, h, ctx) == 1,
          "g2 = h*G");

    // Secrets
    bn_rand_nonzero_range(keys->s1, order);
    bn_rand_nonzero_range(keys->s2, order);
    bn_rand_nonzero_range(keys->s,  order);

    // v = g1^{-s1} g2^{-s2}
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

    // k = g1^s
    EC_POINT_mul(group, keys->k, NULL, keys->g1, keys->s, ctx);

    BN_free(h);
    BN_free(neg1);
    BN_free(neg2);
    EC_POINT_free(t1);
    EC_POINT_free(t2);
}

// ---------------------------------------------------------------------
// PB-SDVSS.Sign (signer + user simulated)
// ---------------------------------------------------------------------

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

    EC_POINT *x    = EC_POINT_new(group);
    EC_POINT *t1   = EC_POINT_new(group);
    EC_POINT *t2   = EC_POINT_new(group);
    EC_POINT *t3   = EC_POINT_new(group);
    EC_POINT *bsig1 = EC_POINT_new(group);
    BIGNUM   *bsig2 = BN_new();

    CHECK(r1 && r2 && u1 && u2 && d && z &&
          alpha_bar && alpha && y1 && y2 && tmp &&
          x && t1 && t2 && t3 && bsig1 && bsig2,
          "alloc sign");

    // r1, r2
    bn_rand_nonzero_range(r1, order);
    bn_rand_nonzero_range(r2, order);

    // x = g1^r1 g2^r2
    EC_POINT_mul(group, t1, NULL, keys->g1, r1, ctx);
    EC_POINT_mul(group, t2, NULL, keys->g2, r2, ctx);
    EC_POINT_add(group, x, t1, t2, ctx);

    // user: u1, u2, d
    bn_rand_nonzero_range(u1, order);
    bn_rand_nonzero_range(u2, order);
    bn_rand_nonzero_range(d,  order);

    // x* = x g1^u1 g2^u2 v^d
    EC_POINT_mul(group, t1, NULL, keys->g1, u1, ctx);
    EC_POINT_mul(group, t2, NULL, keys->g2, u2, ctx);
    EC_POINT_mul(group, t3, NULL, keys->v,  d,  ctx);

    EC_POINT_copy(sig->x_star, x);
    EC_POINT_add(group, sig->x_star, sig->x_star, t1, ctx);
    EC_POINT_add(group, sig->x_star, sig->x_star, t2, ctx);
    EC_POINT_add(group, sig->x_star, sig->x_star, t3, ctx);

    // a* = H(m, info, x*)
    CHECK(hash_m_info_xstar(m, mlen, info, ilen,
                            sig->x_star, group, order,
                            sig->a_star, ctx),
          "hash a*");

    // z = H(info)
    CHECK(hash_info(info, ilen, order, z, ctx) == 1, "hash z");

    // alpha_bar = a* - d + z
    BN_mod_sub(tmp, sig->a_star, d, order, ctx);
    BN_mod_add(alpha_bar, tmp, z, order, ctx);

    // signer: alpha = alpha_bar - z
    BN_mod_sub(alpha, alpha_bar, z, order, ctx);

    // y1 = r1 + alpha*s1 ; y2 = r2 + alpha*s2
    BN_mod_mul(tmp, alpha, keys->s1, order, ctx);
    BN_mod_add(y1, r1, tmp, order, ctx);

    BN_mod_mul(tmp, alpha, keys->s2, order, ctx);
    BN_mod_add(y2, r2, tmp, order, ctx);

    // bsig1 = k^y1 ; bsig2 = y2
    EC_POINT_mul(group, bsig1, NULL, keys->k, y1, ctx);
    BN_copy(bsig2, y2);

    // unblind:
    //   sig1 = bsig1 * k^u1
    //   sig2 = bsig2 + u2
    EC_POINT_mul(group, t1, NULL, keys->k, u1, ctx);
    EC_POINT_add(group, sig->sig1, bsig1, t1, ctx);

    BN_mod_add(sig->sig2, bsig2, u2, order, ctx);

    ok = 1;

    BN_free(r1); BN_free(r2);
    BN_free(u1); BN_free(u2); BN_free(d);
    BN_free(z);  BN_free(alpha_bar); BN_free(alpha);
    BN_free(y1); BN_free(y2); BN_free(tmp);
    EC_POINT_free(x); EC_POINT_free(t1);
    EC_POINT_free(t2); EC_POINT_free(t3);
    EC_POINT_free(bsig1); BN_free(bsig2);

    return ok;
}

// ---------------------------------------------------------------------
// PB-SDVSS.Verify
// ---------------------------------------------------------------------

static int pb_verify(const unsigned char *m, size_t mlen,
                     const unsigned char *info, size_t ilen,
                     const EC_GROUP *group, const BIGNUM *order,
                     const pb_keys_t *keys, const pb_signature_t *sig,
                     BN_CTX *ctx) {
    int ok = 0;

    BIGNUM *a2   = BN_new();
    BIGNUM *sig2s = BN_new();
    BIGNUM *as   = BN_new();
    CHECK(a2 && sig2s && as, "alloc verify BN");

    // recompute a''
    CHECK(hash_m_info_xstar(m, mlen, info, ilen,
                            sig->x_star, group, order,
                            a2, ctx),
          "hash a2");

    if (BN_cmp(a2, sig->a_star) != 0) {
        ok = 0;
        goto end;
    }

    EC_POINT *lhs = EC_POINT_new(group);
    EC_POINT *rhs = EC_POINT_new(group);
    EC_POINT *t1  = EC_POINT_new(group);
    EC_POINT *t2  = EC_POINT_new(group);
    CHECK(lhs && rhs && t1 && t2, "alloc verify points");

    // lhs = (x*)^s
    EC_POINT_mul(group, lhs, NULL, sig->x_star, keys->s, ctx);

    // rhs = sig1 g2^{sig2 s} v^{a* s}
    BN_mod_mul(sig2s, sig->sig2,   keys->s, order, ctx);
    BN_mod_mul(as,    sig->a_star, keys->s, order, ctx);

    EC_POINT_mul(group, t1, NULL, keys->g2, sig2s, ctx);
    EC_POINT_mul(group, t2, NULL, keys->v,  as,    ctx);

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

// ---------------------------------------------------------------------
// Size helper: compute sizes of keys & signature (bytes / bits)
// ---------------------------------------------------------------------

static void print_sizes(const EC_GROUP *group,
                        const BIGNUM *order,
                        const pb_keys_t *keys,
                        const pb_signature_t *sig,
                        BN_CTX *ctx) {
    int scalar_bytes = BN_num_bytes(order);
    int scalar_bits  = scalar_bytes * 8;

    // Compressed point sizes
    size_t v_bytes = EC_POINT_point2oct(
        group, keys->v, POINT_CONVERSION_COMPRESSED,
        NULL, 0, ctx);
    size_t k_bytes = EC_POINT_point2oct(
        group, keys->k, POINT_CONVERSION_COMPRESSED,
        NULL, 0, ctx);
    size_t x_bytes = EC_POINT_point2oct(
        group, sig->x_star, POINT_CONVERSION_COMPRESSED,
        NULL, 0, ctx);
    size_t s1_bytes = EC_POINT_point2oct(
        group, sig->sig1, POINT_CONVERSION_COMPRESSED,
        NULL, 0, ctx);

    // Signature components:
    //   x*   : EC point
    //   a*   : scalar mod q
    //   sig1 : EC point
    //   sig2 : scalar mod q
    size_t sig_bytes = x_bytes + scalar_bytes + s1_bytes + scalar_bytes;
    size_t sig_bits  = sig_bytes * 8;

    printf("  Key / Signature sizes (compressed points):\n");
    printf("    scalar size (Z_q):       %d bytes (%d bits)\n",
           scalar_bytes, scalar_bits);
    printf("    sk_S = (s1,s2):          %d bytes (%d bits)\n",
           2*scalar_bytes, 2*scalar_bits);
    printf("    sk_V = s:                %d bytes (%d bits)\n",
           scalar_bytes, scalar_bits);
    printf("    pk_S = v (EC point):     %zu bytes (%zu bits)\n",
           v_bytes, v_bytes*8);
    printf("    pk_V = k (EC point):     %zu bytes (%zu bits)\n",
           k_bytes, k_bytes*8);
    printf("    PB-SDVSS signature (x*, a*, sig1, sig2):\n");
    printf("      total:                 %zu bytes (%zu bits)\n",
           sig_bytes, sig_bits);
    printf("      breakdown:\n");
    printf("        x*    (EC point):    %zu bytes\n", x_bytes);
    printf("        a*    (scalar):      %d bytes\n",  scalar_bytes);
    printf("        sig1  (EC point):    %zu bytes\n", s1_bytes);
    printf("        sig2  (scalar):      %d bytes\n",  scalar_bytes);
}


// ---------------------------------------------------------------------
// Reproducible benchmark infrastructure
// ---------------------------------------------------------------------

typedef struct {
    int iterations;
    int warmup;
    int repetitions;
    int pin_core;
    int pin_requested;
    const char *csv_path;
    FILE *csv;
} bench_config_t;

typedef struct {
    unsigned char *messages;
    unsigned char *infos;
    size_t count;
} dataset_t;

static void usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --iterations N     Timed iterations per repeated run [default: %d]\n", DEFAULT_ITERATIONS);
    printf("  --warmup W         Untimed warm-up sign+verify iterations [default: %d]\n", DEFAULT_WARMUP);
    printf("  --repetitions R    Number of repeated timed runs [default: %d]\n", DEFAULT_REPETITIONS);
    printf("  --pin-core C       Pin the process to CPU core C, e.g., 0 for single-core tests\n");
    printf("  --csv FILE         Also write machine-readable CSV results\n");
    printf("  --help             Show this message\n");
}

static int parse_positive_int(const char *s, const char *name) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s[0] || *end != '\0' || v <= 0 || v > INT32_MAX) {
        fprintf(stderr, "Invalid %s: %s\n", name, s);
        exit(EXIT_FAILURE);
    }
    return (int)v;
}

static void parse_args(int argc, char **argv, bench_config_t *cfg) {
    cfg->iterations = DEFAULT_ITERATIONS;
    cfg->warmup = DEFAULT_WARMUP;
    cfg->repetitions = DEFAULT_REPETITIONS;
    cfg->pin_core = -1;
    cfg->pin_requested = 0;
    cfg->csv_path = NULL;
    cfg->csv = NULL;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            cfg->iterations = parse_positive_int(argv[++i], "iterations");
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            cfg->warmup = parse_positive_int(argv[++i], "warmup");
        } else if (strcmp(argv[i], "--repetitions") == 0 && i + 1 < argc) {
            cfg->repetitions = parse_positive_int(argv[++i], "repetitions");
        } else if (strcmp(argv[i], "--pin-core") == 0 && i + 1 < argc) {
            char *end = NULL;
            const char *arg = argv[++i];
            long core = strtol(arg, &end, 10);
            if (!arg[0] || *end != '\0' || core < 0 || core > INT32_MAX) {
                fprintf(stderr, "Invalid pin-core: %s\n", arg);
                exit(EXIT_FAILURE);
            }
            cfg->pin_core = (int)core;
            cfg->pin_requested = 1;
        } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            cfg->csv_path = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            exit(EXIT_SUCCESS);
        } else {
            fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
            usage(argv[0]);
            exit(EXIT_FAILURE);
        }
    }
}

static void maybe_pin_to_core(const bench_config_t *cfg) {
#ifdef __linux__
    if (!cfg->pin_requested) return;

    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cfg->pin_core, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        fprintf(stderr, "Warning: could not pin to core %d: %s\n", cfg->pin_core, strerror(errno));
    } else {
        printf("CPU affinity: pinned to core %d\n", cfg->pin_core);
    }
#else
    (void)cfg;
    fprintf(stderr, "Warning: CPU pinning is only implemented on Linux.\n");
#endif
}

static void print_text_file_first_line(const char *label, const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) return;
    char line[512];
    if (fgets(line, sizeof(line), fp)) {
        line[strcspn(line, "\r\n")] = '\0';
        if (line[0]) printf("%s: %s\n", label, line);
    }
    fclose(fp);
}

static void print_cpu_governors(void) {
#ifdef __linux__
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpu < 1) return;

    printf("CPU online cores: %ld\n", ncpu);
    for (long i = 0; i < ncpu && i < 8; ++i) {
        char path[256];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%ld/cpufreq/scaling_governor", i);
        FILE *fp = fopen(path, "r");
        if (!fp) continue;
        char gov[128];
        if (fgets(gov, sizeof(gov), fp)) {
            gov[strcspn(gov, "\r\n")] = '\0';
            printf("CPU governor cpu%ld: %s\n", i, gov);
        }
        fclose(fp);
    }
#endif
}

static void print_environment(const bench_config_t *cfg) {
    printf("============================================================\n");
    printf("PB-SDVSS OpenSSL benchmark metadata\n");
    printf("============================================================\n");
    printf("Benchmark mode: single-threaded process");
    if (cfg->pin_requested) printf(", requested core %d", cfg->pin_core);
    printf("\n");
    printf("Timing method: clock_gettime(CLOCK_MONOTONIC), wall-clock seconds\n");
    printf("Message length: %d bytes\n", MSG_LEN);
    printf("Info length: %d bytes\n", INFO_LEN);
    printf("Iterations per repeated run: %d\n", cfg->iterations);
    printf("Warm-up iterations: %d\n", cfg->warmup);
    printf("Repeated timed runs: %d\n", cfg->repetitions);
    printf("Timed region Sign: PB-SDVSS.Sign only; message/info generation excluded\n");
    printf("Timed region Verify: PB-SDVSS.Verify only; message/info/signature generation excluded\n");
    printf("Compiler flags recorded: %s\n", COMPILE_FLAGS);
    printf("OpenSSL: %s\n", OpenSSL_version(OPENSSL_VERSION));
#ifdef __GNUC__
    printf("Compiler: GCC-compatible %s\n", __VERSION__);
#endif
    print_text_file_first_line("Raspberry/Pi model", "/proc/device-tree/model");
    print_cpu_governors();
    printf("============================================================\n\n");
}

static int compare_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

static double mean(const double *x, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += x[i];
    return s / (double)n;
}

static double stddev_sample(const double *x, int n) {
    if (n < 2) return 0.0;
    double m = mean(x, n);
    double ss = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = x[i] - m;
        ss += d * d;
    }
    return sqrt(ss / (double)(n - 1));
}

static double median(double *tmp, const double *x, int n) {
    memcpy(tmp, x, (size_t)n * sizeof(double));
    qsort(tmp, (size_t)n, sizeof(double), compare_double);
    if (n % 2) return tmp[n / 2];
    return 0.5 * (tmp[n / 2 - 1] + tmp[n / 2]);
}

static void dataset_init(dataset_t *ds, size_t count) {
    ds->count = count;
    ds->messages = calloc(count, MSG_LEN);
    ds->infos = calloc(count, INFO_LEN);
    CHECK(ds->messages && ds->infos, "dataset allocation");
}

static void dataset_randomize(dataset_t *ds) {
    random_bytes(ds->messages, ds->count * MSG_LEN);
    random_bytes(ds->infos, ds->count * INFO_LEN);
}

static void dataset_free(dataset_t *ds) {
    free(ds->messages);
    free(ds->infos);
    ds->messages = NULL;
    ds->infos = NULL;
    ds->count = 0;
}

static unsigned char *dataset_msg(dataset_t *ds, int i) {
    return ds->messages + ((size_t)i * MSG_LEN);
}

static unsigned char *dataset_info(dataset_t *ds, int i) {
    return ds->infos + ((size_t)i * INFO_LEN);
}

static void benchmark_curve(int nid, const char *name, bench_config_t *cfg) {
    printf("=== Curve: %s (NID=%d) ===\n", name, nid);

    BN_CTX *ctx = BN_CTX_new();       CHECK(ctx,   "BN_CTX_new");
    EC_GROUP *group = EC_GROUP_new_by_curve_name(nid);
    CHECK(group, "EC_GROUP_new_by_curve_name");

    BIGNUM *order = BN_new();         CHECK(order, "BN_new order");
    CHECK(EC_GROUP_get_order(group, order, ctx) == 1, "get_order");

    pb_keys_t keys;
    pb_setup_keys(group, order, &keys, ctx);

    pb_signature_t sig;
    pb_sig_init(&sig, group);

    dataset_t warm;
    dataset_init(&warm, (size_t)cfg->warmup);
    dataset_randomize(&warm);

    for (int i = 0; i < cfg->warmup; ++i) {
        CHECK(pb_sign(dataset_msg(&warm, i), MSG_LEN, dataset_info(&warm, i), INFO_LEN,
                      group, order, &keys, &sig, ctx), "warm sign");
        CHECK(pb_verify(dataset_msg(&warm, i), MSG_LEN, dataset_info(&warm, i), INFO_LEN,
                        group, order, &keys, &sig, ctx), "warm verify");
    }
    dataset_free(&warm);

    print_sizes(group, order, &keys, &sig, ctx);
    printf("\n");

    double *sign_us = calloc((size_t)cfg->repetitions, sizeof(double));
    double *verify_us = calloc((size_t)cfg->repetitions, sizeof(double));
    double *tmp = calloc((size_t)cfg->repetitions, sizeof(double));
    CHECK(sign_us && verify_us && tmp, "stats allocation");

    dataset_t ds;
    dataset_init(&ds, (size_t)cfg->iterations);

    for (int r = 0; r < cfg->repetitions; ++r) {
        dataset_randomize(&ds);

        double t0 = now_seconds();
        for (int i = 0; i < cfg->iterations; ++i) {
            CHECK(pb_sign(dataset_msg(&ds, i), MSG_LEN, dataset_info(&ds, i), INFO_LEN,
                          group, order, &keys, &sig, ctx), "timed sign");
        }
        double t1 = now_seconds();
        sign_us[r] = ((t1 - t0) / (double)cfg->iterations) * 1e6;
    }

    pb_signature_t *sigs = calloc((size_t)cfg->iterations, sizeof(pb_signature_t));
    CHECK(sigs, "verify signature array allocation");
    for (int i = 0; i < cfg->iterations; ++i) pb_sig_init(&sigs[i], group);

    for (int r = 0; r < cfg->repetitions; ++r) {
        dataset_randomize(&ds);
        for (int i = 0; i < cfg->iterations; ++i) {
            CHECK(pb_sign(dataset_msg(&ds, i), MSG_LEN, dataset_info(&ds, i), INFO_LEN,
                          group, order, &keys, &sigs[i], ctx), "precompute signature for verify");
        }

        double t0 = now_seconds();
        for (int i = 0; i < cfg->iterations; ++i) {
            CHECK(pb_verify(dataset_msg(&ds, i), MSG_LEN, dataset_info(&ds, i), INFO_LEN,
                            group, order, &keys, &sigs[i], ctx), "timed verify");
        }
        double t1 = now_seconds();
        verify_us[r] = ((t1 - t0) / (double)cfg->iterations) * 1e6;
    }

    double sign_mean = mean(sign_us, cfg->repetitions);
    double sign_median = median(tmp, sign_us, cfg->repetitions);
    double sign_sd = stddev_sample(sign_us, cfg->repetitions);
    double verify_mean = mean(verify_us, cfg->repetitions);
    double verify_median = median(tmp, verify_us, cfg->repetitions);
    double verify_sd = stddev_sample(verify_us, cfg->repetitions);

    printf("  Algorithm PB-SDVSS.Sign, single-threaded:\n");
    printf("    mean:       %.3f µs/op\n", sign_mean);
    printf("    median:     %.3f µs/op\n", sign_median);
    printf("    std. dev.:  %.3f µs/op over %d repeated runs\n", sign_sd, cfg->repetitions);
    printf("    throughput: %.2f ops/sec, based on mean\n\n", 1e6 / sign_mean);

    printf("  Algorithm PB-SDVSS.Verify, single-threaded:\n");
    printf("    mean:       %.3f µs/op\n", verify_mean);
    printf("    median:     %.3f µs/op\n", verify_median);
    printf("    std. dev.:  %.3f µs/op over %d repeated runs\n", verify_sd, cfg->repetitions);
    printf("    throughput: %.2f ops/sec, based on mean\n\n", 1e6 / verify_mean);

    if (cfg->csv) {
        fprintf(cfg->csv,
                "single,%s,%d,%d,%d,sign,%.6f,%.6f,%.6f,%.6f\n",
                name, cfg->iterations, cfg->warmup, cfg->repetitions,
                sign_mean, sign_median, sign_sd, 1e6 / sign_mean);
        fprintf(cfg->csv,
                "single,%s,%d,%d,%d,verify,%.6f,%.6f,%.6f,%.6f\n",
                name, cfg->iterations, cfg->warmup, cfg->repetitions,
                verify_mean, verify_median, verify_sd, 1e6 / verify_mean);
        fflush(cfg->csv);
    }

    for (int i = 0; i < cfg->iterations; ++i) pb_sig_free(&sigs[i]);
    free(sigs);
    dataset_free(&ds);
    free(sign_us);
    free(verify_us);
    free(tmp);

    pb_sig_free(&sig);
    pb_keys_free(&keys);
    BN_free(order);
    EC_GROUP_free(group);
    BN_CTX_free(ctx);
}

int main(int argc, char **argv) {
    bench_config_t cfg;
    parse_args(argc, argv, &cfg);
    maybe_pin_to_core(&cfg);

    if (cfg.csv_path) {
        cfg.csv = fopen(cfg.csv_path, "w");
        CHECK(cfg.csv, "open CSV output");
        fprintf(cfg.csv, "mode,curve,iterations,warmup,repetitions,operation,mean_us,median_us,stddev_us,throughput_ops_s\n");
    }

    print_environment(&cfg);

    int nids[] = {
        NID_X9_62_prime192v1,    // NIST P-192
        NID_secp224r1,           // NIST P-224
        NID_X9_62_prime256v1,    // NIST P-256
        NID_secp256k1,           // SECP256k1 (Bitcoin)
        NID_secp384r1,           // NIST P-384
        NID_secp521r1,           // NIST P-521
        NID_brainpoolP256r1,     // Brainpool P256
        NID_brainpoolP384r1      // Brainpool P384
    };
    const char *names[] = {
        "prime192v1 (NIST P-192)",
        "secp224r1 (NIST P-224)",
        "prime256v1 (NIST P-256)",
        "secp256k1 (Bitcoin)",
        "secp384r1 (NIST P-384)",
        "secp521r1 (NIST P-521)",
        "brainpoolP256r1",
        "brainpoolP384r1"
    };

    int num = (int)(sizeof(nids) / sizeof(nids[0]));
    for (int i = 0; i < num; ++i) benchmark_curve(nids[i], names[i], &cfg);

    if (cfg.csv) fclose(cfg.csv);
    return 0;
}
