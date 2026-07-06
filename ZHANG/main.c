#define _GNU_SOURCE
// zhang_wen_bidsdvs_relic_repro_benchmark.c
// -----------------------------------------------------------------------------
// RELIC implementation-level benchmark of Zhang-Wen BIDSDVS
// "Provably Secure Blind ID-Based Strong Designated Verifier Signature Scheme"
//
// This file implements the Blind ID-Based Strong Designated Verifier Signature
// (BIDSDVS) protocol described in Section V-B of the paper, with a practical
// Type-III pairing adaptation:
//
//   Original paper notation: symmetric pairing e : G1 x G1 -> G2.
//   This implementation:     asymmetric pairing e : G1 x G2 -> GT.
//
// Type-III placement used here:
//   signer identity public key      Q_IDA in G1
//   signer identity private key     S_IDA = s Q_IDA in G1
//   verifier identity public key    Q_IDB in G2
//   verifier identity private key   S_IDB = s Q_IDB in G2
//
// This preserves the key equality:
//   e(S_IDA, Q_IDB) = e(Q_IDA, S_IDB)
//
// Blind-signing protocol implemented:
//   Signer:
//     r1,r2 <- Z_q^*
//     u1 = r1 Q_IDB                      in G2
//     u2 = r1 r2 Q_IDB                   in G2
//     k  = e(S_IDA, Q_IDB)               in GT
//     send (u1,u2,k)
//
//   User:
//     alpha,beta <- Z_q^*
//     u1_hat = alpha u1                  in G2
//     u2_hat = alpha beta u2             in G2
//     h      = beta H1(m,k)              in Z_q
//     send h
//
//   Signer:
//     X = r2 h Q_IDA                     in G1
//     Y = r1^{-1} S_IDA                  in G1
//     send (X,Y)
//
//   User:
//     Y_hat = alpha^{-1} Y               in G1
//     V_hat = X + Y_hat                  in G1
//     sigma = (u1_hat, u2_hat, V_hat)
//
// Verification implemented:
//   k_v = e(Q_IDA, S_IDB)
//   h0  = H1(m,k_v)
//   accept iff
//     e(V_hat, u1_hat) = e(h0 Q_IDA, u2_hat) * e(Q_IDA, S_IDB)
//
// Notes:
//   - The paper uses symmetric-pairing notation. RELIC typically exposes
//     efficient Type-III pairings, so this implementation uses a standard
//     asymmetric adaptation.
//   - The file includes an optional verifier-side simulator that creates valid
//     designated-verifier transcripts using only the verifier private key S_IDB.
//   - This is benchmark/prototype code, not constant-time production code.
//
// Build example:
//   gcc -O3 -Wall -Wextra main.c -o main -I/path/to/relic/include -L/path/to/relic/lib -lrelic -lgmp -lm
//
// Run example:
//   ./main --iterations 1000 --warmup 100 --repetitions 10 --pin-core 0 --csv zhang_wen_bidsdvs_results.csv
// -----------------------------------------------------------------------------

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

#define DEFAULT_ITERATIONS 1000
#define DEFAULT_WARMUP     100
#define DEFAULT_REPETITIONS 10
#define MSG_LEN            200

#define CHECK_OK(cond, msg) \
    do { if (!(cond)) { fprintf(stderr, "Error: %s\n", msg); exit(EXIT_FAILURE); } } while (0)

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void random_bytes(uint8_t *buf, size_t len) {
    rand_bytes(buf, (int)len);
}

static void bn_rand_nonzero_mod(bn_t out, const bn_t q) {
    do {
        bn_rand_mod(out, q);
    } while (bn_is_zero(out));
}

// -----------------------------------------------------------------------------
// Hash helpers
// -----------------------------------------------------------------------------

static void hash_digest_to_bn_mod_q(bn_t out, const uint8_t h[32]) {
    bn_t q, t;
    bn_null(q); bn_null(t);
    bn_new(q);  bn_new(t);

    g1_get_ord(q);
    bn_read_bin(t, h, 32);
    bn_mod(out, t, q);
    if (bn_is_zero(out)) {
        bn_set_dig(out, 1);
    }

    bn_free(t);
    bn_free(q);
}

// H0_A: signer identity -> G1
static void H0_id_to_g1(g1_t out, const uint8_t *id, size_t id_len) {
    uint8_t h[32];
    md_map_sh256(h, id, (int)id_len);
    g1_map(out, h, 32);
}

// H0_B: verifier identity -> G2
static void H0_id_to_g2(g2_t out, const uint8_t *id, size_t id_len) {
    uint8_t h[32];
    md_map_sh256(h, id, (int)id_len);
    g2_map(out, h, 32);
}

// H1: {0,1}* x GT -> Z_q^*
static void H1_msg_gt_to_scalar(bn_t out,
                                const uint8_t *msg,
                                size_t msg_len,
                                const gt_t k) {
    int k_len = gt_size_bin(k, 1);
    CHECK_OK(k_len > 0, "gt_size_bin failed");

    size_t total = msg_len + (size_t)k_len;
    uint8_t *buf = (uint8_t *)malloc(total);
    CHECK_OK(buf != NULL, "malloc H1 buffer");

    memcpy(buf, msg, msg_len);
    gt_write_bin(buf + msg_len, k_len, k, 1);

    uint8_t h[32];
    md_map_sh256(h, buf, (int)total);
    hash_digest_to_bn_mod_q(out, h);

    free(buf);
}

// -----------------------------------------------------------------------------
// Data structures
// -----------------------------------------------------------------------------

typedef struct {
    bn_t s;          // KGC master secret

    g1_t Q_IDA;      // signer public identity point in G1
    g1_t S_IDA;      // signer private key s*Q_IDA in G1

    g2_t Q_IDB;      // designated verifier public identity point in G2
    g2_t S_IDB;      // designated verifier private key s*Q_IDB in G2
} zw_keys_t;

typedef struct {
    g2_t u1_hat;     // \hat{u}_1 in G2
    g2_t u2_hat;     // \hat{u}_2 in G2
    g1_t V_hat;      // \hat{V} in G1
} zw_sig_t;

typedef struct {
    double setup_extract;
    double signer_commitment;
    double user_blinding;
    double signer_response;
    double user_unblinding;
    double total_issue;
    double verify;
    double simulate;
} timing_t;

// -----------------------------------------------------------------------------
// Init / free helpers
// -----------------------------------------------------------------------------

static void keys_init(zw_keys_t *K) {
    (void)K;
    bn_null(K->s);       bn_new(K->s);

    g1_null(K->Q_IDA);   g1_new(K->Q_IDA);
    g1_null(K->S_IDA);   g1_new(K->S_IDA);

    g2_null(K->Q_IDB);   g2_new(K->Q_IDB);
    g2_null(K->S_IDB);   g2_new(K->S_IDB);
}

static void keys_free(zw_keys_t *K) {
    (void)K;
    bn_free(K->s);
    g1_free(K->Q_IDA);
    g1_free(K->S_IDA);
    g2_free(K->Q_IDB);
    g2_free(K->S_IDB);
}

static void sig_init(zw_sig_t *sig) {
    (void)sig;
    g2_null(sig->u1_hat); g2_new(sig->u1_hat);
    g2_null(sig->u2_hat); g2_new(sig->u2_hat);
    g1_null(sig->V_hat);  g1_new(sig->V_hat);
}

static void sig_free(zw_sig_t *sig) {
    (void)sig;
    g2_free(sig->u1_hat);
    g2_free(sig->u2_hat);
    g1_free(sig->V_hat);
}

// -----------------------------------------------------------------------------
// Setup + Extract
// -----------------------------------------------------------------------------

static void setup_and_extract(zw_keys_t *K,
                              const uint8_t *id_signer,
                              size_t id_signer_len,
                              const uint8_t *id_verifier,
                              size_t id_verifier_len) {
    keys_init(K);

    bn_t q;
    bn_null(q); bn_new(q);
    g1_get_ord(q);

    bn_rand_nonzero_mod(K->s, q);

    H0_id_to_g1(K->Q_IDA, id_signer, id_signer_len);
    g1_mul(K->S_IDA, K->Q_IDA, K->s);
    g1_norm(K->S_IDA, K->S_IDA);

    H0_id_to_g2(K->Q_IDB, id_verifier, id_verifier_len);
    g2_mul(K->S_IDB, K->Q_IDB, K->s);
    g2_norm(K->S_IDB, K->S_IDB);

    bn_free(q);
}

// -----------------------------------------------------------------------------
// Zhang-Wen BIDSDVS blind-signing protocol
// -----------------------------------------------------------------------------

static int zw_bidsdvs_issue_timed(const uint8_t *msg,
                                  size_t msg_len,
                                  const zw_keys_t *K,
                                  zw_sig_t *sig,
                                  timing_t *T) {
    double total0 = now_seconds();
    double t0, t1;

    bn_t q;
    bn_null(q); bn_new(q);
    g1_get_ord(q);

    bn_t r1, r2, alpha, beta;
    bn_t h0, h_blind, r1_inv, alpha_inv;
    bn_t tmp, scalar;

    bn_null(r1);        bn_null(r2);
    bn_null(alpha);     bn_null(beta);
    bn_null(h0);        bn_null(h_blind);
    bn_null(r1_inv);    bn_null(alpha_inv);
    bn_null(tmp);       bn_null(scalar);

    bn_new(r1);         bn_new(r2);
    bn_new(alpha);      bn_new(beta);
    bn_new(h0);         bn_new(h_blind);
    bn_new(r1_inv);     bn_new(alpha_inv);
    bn_new(tmp);        bn_new(scalar);

    g2_t u1, u2;
    g2_null(u1); g2_null(u2);
    g2_new(u1);  g2_new(u2);

    g1_t X, Y, Y_hat;
    g1_null(X); g1_null(Y); g1_null(Y_hat);
    g1_new(X);  g1_new(Y);  g1_new(Y_hat);

    gt_t k;
    gt_null(k); gt_new(k);

    // -------------------------------------------------------------------------
    // Signer commitment: u1, u2, k
    // -------------------------------------------------------------------------
    t0 = now_seconds();

    bn_rand_nonzero_mod(r1, q);
    bn_rand_nonzero_mod(r2, q);

    // u1 = r1 Q_IDB
    g2_mul(u1, K->Q_IDB, r1);
    g2_norm(u1, u1);

    // u2 = r1 r2 Q_IDB
    bn_mul(tmp, r1, r2);
    bn_mod(tmp, tmp, q);
    g2_mul(u2, K->Q_IDB, tmp);
    g2_norm(u2, u2);

    // k = e(S_IDA, Q_IDB)
    pc_map(k, K->S_IDA, K->Q_IDB);

    t1 = now_seconds();
    if (T) T->signer_commitment += t1 - t0;

    // -------------------------------------------------------------------------
    // User blinding: u1_hat, u2_hat, h = beta H1(m,k)
    // -------------------------------------------------------------------------
    t0 = now_seconds();

    bn_rand_nonzero_mod(alpha, q);
    bn_rand_nonzero_mod(beta, q);

    g2_mul(sig->u1_hat, u1, alpha);
    g2_norm(sig->u1_hat, sig->u1_hat);

    bn_mul(tmp, alpha, beta);
    bn_mod(tmp, tmp, q);
    g2_mul(sig->u2_hat, u2, tmp);
    g2_norm(sig->u2_hat, sig->u2_hat);

    H1_msg_gt_to_scalar(h0, msg, msg_len, k);
    bn_mul(h_blind, beta, h0);
    bn_mod(h_blind, h_blind, q);

    t1 = now_seconds();
    if (T) T->user_blinding += t1 - t0;

    // -------------------------------------------------------------------------
    // Signer response: X = r2 h Q_IDA, Y = r1^{-1} S_IDA
    // -------------------------------------------------------------------------
    t0 = now_seconds();

    bn_mul(scalar, r2, h_blind);
    bn_mod(scalar, scalar, q);
    g1_mul(X, K->Q_IDA, scalar);
    g1_norm(X, X);

    bn_mod_inv(r1_inv, r1, q);
    g1_mul(Y, K->S_IDA, r1_inv);
    g1_norm(Y, Y);

    t1 = now_seconds();
    if (T) T->signer_response += t1 - t0;

    // -------------------------------------------------------------------------
    // User unblinding: V_hat = X + alpha^{-1}Y
    // -------------------------------------------------------------------------
    t0 = now_seconds();

    bn_mod_inv(alpha_inv, alpha, q);
    g1_mul(Y_hat, Y, alpha_inv);
    g1_norm(Y_hat, Y_hat);

    g1_add(sig->V_hat, X, Y_hat);
    g1_norm(sig->V_hat, sig->V_hat);

    t1 = now_seconds();
    if (T) T->user_unblinding += t1 - t0;

    if (T) T->total_issue += now_seconds() - total0;

    gt_free(k);
    g2_free(u1); g2_free(u2);
    g1_free(X); g1_free(Y); g1_free(Y_hat);

    bn_free(r1);     bn_free(r2);
    bn_free(alpha);  bn_free(beta);
    bn_free(h0);     bn_free(h_blind);
    bn_free(r1_inv); bn_free(alpha_inv);
    bn_free(tmp);    bn_free(scalar);
    bn_free(q);

    return 1;
}

static int zw_bidsdvs_issue(const uint8_t *msg,
                            size_t msg_len,
                            const zw_keys_t *K,
                            zw_sig_t *sig) {
    return zw_bidsdvs_issue_timed(msg, msg_len, K, sig, NULL);
}

// -----------------------------------------------------------------------------
// Designated verification
// -----------------------------------------------------------------------------

static int zw_bidsdvs_verify_timed(const uint8_t *msg,
                                   size_t msg_len,
                                   const zw_keys_t *K,
                                   const zw_sig_t *sig,
                                   timing_t *T) {
    double t0 = now_seconds();

    bn_t h0;
    bn_null(h0); bn_new(h0);

    gt_t k_v, lhs, rhs1, rhs2, rhs;
    gt_null(k_v); gt_null(lhs); gt_null(rhs1); gt_null(rhs2); gt_null(rhs);
    gt_new(k_v);  gt_new(lhs);  gt_new(rhs1);  gt_new(rhs2);  gt_new(rhs);

    g1_t hQ;
    g1_null(hQ); g1_new(hQ);

    // k_v = e(Q_IDA, S_IDB)
    pc_map(k_v, K->Q_IDA, K->S_IDB);
    H1_msg_gt_to_scalar(h0, msg, msg_len, k_v);

    // h0 Q_IDA
    g1_mul(hQ, K->Q_IDA, h0);
    g1_norm(hQ, hQ);

    // lhs = e(V_hat, u1_hat)
    pc_map(lhs, sig->V_hat, sig->u1_hat);

    // rhs = e(h0 Q_IDA, u2_hat) * e(Q_IDA, S_IDB)
    pc_map(rhs1, hQ, sig->u2_hat);
    pc_map(rhs2, K->Q_IDA, K->S_IDB);
    gt_mul(rhs, rhs1, rhs2);

    int ok = (gt_cmp(lhs, rhs) == RLC_EQ);

    g1_free(hQ);
    gt_free(k_v); gt_free(lhs); gt_free(rhs1); gt_free(rhs2); gt_free(rhs);
    bn_free(h0);

    double t1 = now_seconds();
    if (T) T->verify += t1 - t0;

    return ok;
}

static int zw_bidsdvs_verify(const uint8_t *msg,
                             size_t msg_len,
                             const zw_keys_t *K,
                             const zw_sig_t *sig) {
    return zw_bidsdvs_verify_timed(msg, msg_len, K, sig, NULL);
}

// -----------------------------------------------------------------------------
// Optional Type-III verifier-side simulator
// -----------------------------------------------------------------------------
// Produces a valid transcript using only verifier private material S_IDB.
// This is useful for checking the designated-verifier non-transferability
// intuition in the asymmetric implementation.
// -----------------------------------------------------------------------------

static int zw_bidsdvs_simulate_timed(const uint8_t *msg,
                                     size_t msg_len,
                                     const zw_keys_t *K,
                                     zw_sig_t *sig,
                                     timing_t *T) {
    double t0 = now_seconds();

    bn_t q;
    bn_null(q); bn_new(q);
    g1_get_ord(q);

    bn_t a, b, a_inv, b_over_a, h0, coeff, tmp;
    bn_null(a);       bn_null(b);
    bn_null(a_inv);   bn_null(b_over_a);
    bn_null(h0);      bn_null(coeff);
    bn_null(tmp);

    bn_new(a);        bn_new(b);
    bn_new(a_inv);    bn_new(b_over_a);
    bn_new(h0);       bn_new(coeff);
    bn_new(tmp);

    gt_t k_v;
    gt_null(k_v); gt_new(k_v);

    // Choose random-looking G2 components using the verifier secret key.
    bn_rand_nonzero_mod(a, q);
    bn_rand_nonzero_mod(b, q);

    g2_mul(sig->u1_hat, K->S_IDB, a);
    g2_norm(sig->u1_hat, sig->u1_hat);

    g2_mul(sig->u2_hat, K->S_IDB, b);
    g2_norm(sig->u2_hat, sig->u2_hat);

    // h0 = H1(m, e(Q_IDA, S_IDB))
    pc_map(k_v, K->Q_IDA, K->S_IDB);
    H1_msg_gt_to_scalar(h0, msg, msg_len, k_v);

    // V_hat = ((b/a) h0 + 1/a) Q_IDA
    bn_mod_inv(a_inv, a, q);
    bn_mul(b_over_a, b, a_inv);
    bn_mod(b_over_a, b_over_a, q);

    bn_mul(tmp, b_over_a, h0);
    bn_mod(tmp, tmp, q);
    bn_add(coeff, tmp, a_inv);
    bn_mod(coeff, coeff, q);

    g1_mul(sig->V_hat, K->Q_IDA, coeff);
    g1_norm(sig->V_hat, sig->V_hat);

    gt_free(k_v);

    bn_free(a);      bn_free(b);
    bn_free(a_inv);  bn_free(b_over_a);
    bn_free(h0);     bn_free(coeff);
    bn_free(tmp);    bn_free(q);

    double t1 = now_seconds();
    if (T) T->simulate += t1 - t0;

    return 1;
}

static int zw_bidsdvs_simulate(const uint8_t *msg,
                               size_t msg_len,
                               const zw_keys_t *K,
                               zw_sig_t *sig) {
    return zw_bidsdvs_simulate_timed(msg, msg_len, K, sig, NULL);
}

// -----------------------------------------------------------------------------
// Size reporting
// -----------------------------------------------------------------------------

static void print_sizes(const zw_keys_t *K, const zw_sig_t *sig) {
    bn_t q;
    bn_null(q); bn_new(q);
    g1_get_ord(q);

    int scalar_bytes = bn_size_bin(q);
    int g1_c = g1_size_bin(K->Q_IDA, 1);
    int g1_u = g1_size_bin(K->Q_IDA, 0);
    int g2_c = g2_size_bin(K->Q_IDB, 1);
    int g2_u = g2_size_bin(K->Q_IDB, 0);

    int sig_c = g2_size_bin(sig->u1_hat, 1) +
                g2_size_bin(sig->u2_hat, 1) +
                g1_size_bin(sig->V_hat, 1);

    int sig_u = g2_size_bin(sig->u1_hat, 0) +
                g2_size_bin(sig->u2_hat, 0) +
                g1_size_bin(sig->V_hat, 0);

    printf("  --- Cryptographic Sizes ---\n");
    printf("  Group order q:                         %d bytes (%d bits)\n", scalar_bytes, scalar_bytes * 8);
    printf("  Master secret s:                       %d bytes (%d bits)\n\n", scalar_bytes, scalar_bytes * 8);

    printf("  Signer public Q_IDA  (G1):             %d bits compressed, %d bits uncompressed\n", g1_c * 8, g1_u * 8);
    printf("  Signer private S_IDA (G1):             %d bits compressed, %d bits uncompressed\n", g1_c * 8, g1_u * 8);
    printf("  Verifier public Q_IDB  (G2):           %d bits compressed, %d bits uncompressed\n", g2_c * 8, g2_u * 8);
    printf("  Verifier private S_IDB (G2):           %d bits compressed, %d bits uncompressed\n\n", g2_c * 8, g2_u * 8);

    printf("  Zhang-Wen BIDSDVS signature, Type-III: (u1_hat, u2_hat, V_hat)\n");
    printf("    compressed:                          %d bytes (%d bits)\n", sig_c, sig_c * 8);
    printf("      u1_hat in G2:                      %zu bytes\n", (size_t)g2_size_bin(sig->u1_hat, 1));
    printf("      u2_hat in G2:                      %zu bytes\n", (size_t)g2_size_bin(sig->u2_hat, 1));
    printf("      V_hat  in G1:                      %zu bytes\n", (size_t)g1_size_bin(sig->V_hat, 1));
    printf("    uncompressed:                        %d bytes (%d bits)\n\n", sig_u, sig_u * 8);

    bn_free(q);
}


// -----------------------------------------------------------------------------
// Reproducible benchmark harness
// -----------------------------------------------------------------------------

#ifndef COMPILE_FLAGS
#define COMPILE_FLAGS "unknown"
#endif

typedef struct {
    size_t iterations;
    size_t warmup;
    size_t repetitions;
    int pin_requested;
    int pin_core;
    const char *csv_path;
} bench_config_t;

typedef struct {
    double mean_us;
    double median_us;
    double stddev_us;
    double throughput_ops_s;
} stats_t;

typedef struct {
    uint8_t msg[MSG_LEN];
    zw_sig_t sig;
} verify_item_t;

static int cmp_double_for_qsort(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

static stats_t compute_stats_us(const double *values, size_t n) {
    stats_t s;
    memset(&s, 0, sizeof(s));
    CHECK_OK(n > 0, "compute_stats_us with n=0");

    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum += values[i];
    s.mean_us = sum / (double)n;

    double *tmp = (double *)malloc(n * sizeof(double));
    CHECK_OK(tmp != NULL, "malloc stats tmp");
    memcpy(tmp, values, n * sizeof(double));
    qsort(tmp, n, sizeof(double), cmp_double_for_qsort);
    if ((n % 2) == 1) {
        s.median_us = tmp[n / 2];
    } else {
        s.median_us = 0.5 * (tmp[(n / 2) - 1] + tmp[n / 2]);
    }
    free(tmp);

    if (n > 1) {
        double acc = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double d = values[i] - s.mean_us;
            acc += d * d;
        }
        s.stddev_us = sqrt(acc / (double)(n - 1));
    }

    s.throughput_ops_s = 1e6 / (s.mean_us > 0.0 ? s.mean_us : 1e-12);
    return s;
}

static size_t parse_positive_size(const char *value, const char *name) {
    errno = 0;
    char *end = NULL;
    unsigned long long v = strtoull(value, &end, 10);
    if (errno != 0 || value[0] == '\0' || *end != '\0' || v == 0ULL) {
        fprintf(stderr, "Invalid %s: %s\n", name, value);
        exit(EXIT_FAILURE);
    }
    return (size_t)v;
}

static int parse_nonnegative_int(const char *value, const char *name) {
    errno = 0;
    char *end = NULL;
    long v = strtol(value, &end, 10);
    if (errno != 0 || value[0] == '\0' || *end != '\0' || v < 0 || v > INT_MAX) {
        fprintf(stderr, "Invalid %s: %s\n", name, value);
        exit(EXIT_FAILURE);
    }
    return (int)v;
}

static void usage(const char *prog) {
    fprintf(stderr,
            "Usage:\n"
            "  %s [--iterations N] [--warmup W] [--repetitions R] [--pin-core C] [--csv file]\n\n"
            "Defaults:\n"
            "  --iterations %d\n"
            "  --warmup %d\n"
            "  --repetitions %d\n\n"
            "Example:\n"
            "  %s --iterations 1000 --warmup 100 --repetitions 10 --pin-core 0 --csv zhang_wen_bidsdvs_results.csv\n",
            prog, DEFAULT_ITERATIONS, DEFAULT_WARMUP, DEFAULT_REPETITIONS, prog);
}

static bench_config_t parse_args(int argc, char **argv) {
    bench_config_t cfg;
    cfg.iterations = DEFAULT_ITERATIONS;
    cfg.warmup = DEFAULT_WARMUP;
    cfg.repetitions = DEFAULT_REPETITIONS;
    cfg.pin_requested = 0;
    cfg.pin_core = -1;
    cfg.csv_path = NULL;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            cfg.iterations = parse_positive_size(argv[++i], "iterations");
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            cfg.warmup = parse_positive_size(argv[++i], "warmup");
        } else if (strcmp(argv[i], "--repetitions") == 0 && i + 1 < argc) {
            cfg.repetitions = parse_positive_size(argv[++i], "repetitions");
        } else if (strcmp(argv[i], "--pin-core") == 0 && i + 1 < argc) {
            cfg.pin_core = parse_nonnegative_int(argv[++i], "pin-core");
            cfg.pin_requested = 1;
        } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            cfg.csv_path = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            exit(EXIT_SUCCESS);
        } else {
            usage(argv[0]);
            exit(EXIT_FAILURE);
        }
    }
    return cfg;
}

static void try_pin_core(const bench_config_t *cfg) {
    if (!cfg->pin_requested) return;
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cfg->pin_core, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        fprintf(stderr, "Warning: could not pin process to core %d: %s\n", cfg->pin_core, strerror(errno));
    }
#else
    fprintf(stderr, "Warning: CPU pinning requested but not supported on this platform.\n");
#endif
}

static long read_online_cores(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? n : -1;
}

static void print_governors(void) {
#ifdef __linux__
    long cores = read_online_cores();
    if (cores < 0) return;
    long limit = cores < 8 ? cores : 8;
    for (long i = 0; i < limit; ++i) {
        char path[256];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%ld/cpufreq/scaling_governor", i);
        FILE *fp = fopen(path, "r");
        if (!fp) continue;
        char buf[128];
        if (fgets(buf, sizeof(buf), fp)) {
            buf[strcspn(buf, "\r\n")] = '\0';
            printf("CPU governor cpu%ld: %s\n", i, buf);
        }
        fclose(fp);
    }
#endif
}

static void print_metadata(const bench_config_t *cfg) {
    struct utsname uts;
    memset(&uts, 0, sizeof(uts));
    uname(&uts);

    printf("Zhang-Wen BIDSDVS RELIC benchmark metadata\n");
    printf("============================================================\n");
    printf("Benchmark mode: single-threaded process%s", cfg->pin_requested ? ", requested core " : "");
    if (cfg->pin_requested) printf("%d", cfg->pin_core);
    printf("\n");
    printf("Timing method: clock_gettime(CLOCK_MONOTONIC), wall-clock seconds\n");
    printf("Message length: %d bytes\n", MSG_LEN);
    printf("Iterations per repeated run: %zu\n", cfg->iterations);
    printf("Warm-up iterations: %zu\n", cfg->warmup);
    printf("Repeated timed runs: %zu\n", cfg->repetitions);
    printf("Timed region Issue: Zhang-Wen blind-signing issue protocol only; message generation excluded\n");
    printf("Timed region Verify: designated verification only; message/signature generation excluded\n");
    printf("Timed region Simulate: verifier-side simulation only; message generation excluded\n");
    printf("Compiler flags recorded: %s\n", COMPILE_FLAGS);
#if defined(__GNUC__)
    printf("Compiler: GCC-compatible %d.%d.%d\n", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
    printf("Compiler: unknown\n");
#endif
    printf("CPU online cores: %ld\n", read_online_cores());
    if (uts.sysname[0]) {
        printf("System: %s %s %s %s\n", uts.sysname, uts.release, uts.machine, uts.nodename);
    }
    print_governors();
    printf("RELIC parameter set selected by pc_param_set_any(); details printed below.\n");
    printf("============================================================\n\n");
}

static verify_item_t *verify_dataset_alloc(size_t n) {
    verify_item_t *items = (verify_item_t *)calloc(n, sizeof(*items));
    CHECK_OK(items != NULL, "calloc verify dataset");
    for (size_t i = 0; i < n; ++i) sig_init(&items[i].sig);
    return items;
}

static void verify_dataset_free(verify_item_t *items, size_t n) {
    if (!items) return;
    for (size_t i = 0; i < n; ++i) sig_free(&items[i].sig);
    free(items);
}

static void fill_random_messages(uint8_t *msgs, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        random_bytes(msgs + i * MSG_LEN, MSG_LEN);
    }
}

static void build_verify_dataset(verify_item_t *items,
                                 size_t n,
                                 const zw_keys_t *K) {
    for (size_t i = 0; i < n; ++i) {
        random_bytes(items[i].msg, MSG_LEN);
        CHECK_OK(zw_bidsdvs_issue(items[i].msg, MSG_LEN, K, &items[i].sig), "build verify dataset issue");
        CHECK_OK(zw_bidsdvs_verify(items[i].msg, MSG_LEN, K, &items[i].sig), "build verify dataset verify");
    }
}

static void run_warmup(size_t warmup, const zw_keys_t *K) {
    zw_sig_t sig, sim_sig;
    sig_init(&sig);
    sig_init(&sim_sig);
    uint8_t msg[MSG_LEN];

    for (size_t i = 0; i < warmup; ++i) {
        random_bytes(msg, MSG_LEN);
        CHECK_OK(zw_bidsdvs_issue(msg, MSG_LEN, K, &sig), "warm issue");
        CHECK_OK(zw_bidsdvs_verify(msg, MSG_LEN, K, &sig), "warm verify");
        CHECK_OK(zw_bidsdvs_simulate(msg, MSG_LEN, K, &sim_sig), "warm simulate");
        CHECK_OK(zw_bidsdvs_verify(msg, MSG_LEN, K, &sim_sig), "warm simulated verify");
    }

    sig_free(&sig);
    sig_free(&sim_sig);
}

static void write_csv_row(FILE *csv,
                          const char *operation,
                          stats_t st,
                          const bench_config_t *cfg,
                          const char *notes) {
    if (!csv) return;
    fprintf(csv,
            "Zhang-Wen BIDSDVS,BN-P256 Type-III RELIC,%s,%.6f,%.6f,%.6f,%.6f,%zu,%zu,%zu,%s\n",
            operation,
            st.mean_us,
            st.median_us,
            st.stddev_us,
            st.throughput_ops_s,
            cfg->repetitions,
            cfg->iterations,
            cfg->warmup,
            notes ? notes : "");
}

static void print_stat_line(const char *label, stats_t st, size_t repetitions) {
    printf("  %-35s mean: %10.3f us/op | median: %10.3f | std. dev.: %9.3f | throughput: %9.2f ops/s over %zu runs\n",
           label, st.mean_us, st.median_us, st.stddev_us, st.throughput_ops_s, repetitions);
}

static void run_correctness_and_sizes(void) {
    const uint8_t ID_A[] = "Signer-A";
    const uint8_t ID_B[] = "Designated-Verifier-B";

    zw_keys_t K;
    setup_and_extract(&K, ID_A, sizeof(ID_A) - 1, ID_B, sizeof(ID_B) - 1);

    zw_sig_t sig, sim_sig;
    sig_init(&sig);
    sig_init(&sim_sig);

    uint8_t msg[MSG_LEN];
    random_bytes(msg, MSG_LEN);

    CHECK_OK(zw_bidsdvs_issue(msg, MSG_LEN, &K, &sig), "sample issue failed");
    CHECK_OK(zw_bidsdvs_verify(msg, MSG_LEN, &K, &sig), "sample verify failed");
    CHECK_OK(zw_bidsdvs_simulate(msg, MSG_LEN, &K, &sim_sig), "sample simulate failed");
    CHECK_OK(zw_bidsdvs_verify(msg, MSG_LEN, &K, &sim_sig), "sample simulated verify failed");

    uint8_t bad_msg[MSG_LEN];
    random_bytes(bad_msg, MSG_LEN);
    int bad_ok = zw_bidsdvs_verify(bad_msg, MSG_LEN, &K, &sig);
    CHECK_OK(bad_ok == 0, "negative test failed: changed message verified");

    print_sizes(&K, &sig);
    printf("  --- Correctness Tests ---\n");
    printf("  Honest blind-issued signature verifies:        yes\n");
    printf("  Verifier-simulated signature verifies:         yes\n");
    printf("  Changed-message negative test rejects:         yes\n\n");

    sig_free(&sig);
    sig_free(&sim_sig);
    keys_free(&K);
}

static void benchmark_all(const bench_config_t *cfg) {
    const uint8_t ID_A[] = "Signer-A";
    const uint8_t ID_B[] = "Designated-Verifier-B";

    double *setup_us     = (double *)calloc(cfg->repetitions, sizeof(double));
    double *issue_us     = (double *)calloc(cfg->repetitions, sizeof(double));
    double *commit_us    = (double *)calloc(cfg->repetitions, sizeof(double));
    double *blind_us     = (double *)calloc(cfg->repetitions, sizeof(double));
    double *response_us  = (double *)calloc(cfg->repetitions, sizeof(double));
    double *unblind_us   = (double *)calloc(cfg->repetitions, sizeof(double));
    double *verify_us    = (double *)calloc(cfg->repetitions, sizeof(double));
    double *simulate_us  = (double *)calloc(cfg->repetitions, sizeof(double));

    CHECK_OK(setup_us && issue_us && commit_us && blind_us && response_us && unblind_us && verify_us && simulate_us,
             "calloc benchmark arrays");

    for (size_t rep = 0; rep < cfg->repetitions; ++rep) {
        zw_keys_t K;
        double t0 = now_seconds();
        setup_and_extract(&K, ID_A, sizeof(ID_A) - 1, ID_B, sizeof(ID_B) - 1);
        double t1 = now_seconds();
        setup_us[rep] = (t1 - t0) * 1e6;

        run_warmup(cfg->warmup, &K);

        uint8_t *issue_msgs = (uint8_t *)malloc(cfg->iterations * MSG_LEN);
        uint8_t *sim_msgs   = (uint8_t *)malloc(cfg->iterations * MSG_LEN);
        CHECK_OK(issue_msgs && sim_msgs, "malloc message arrays");
        fill_random_messages(issue_msgs, cfg->iterations);
        fill_random_messages(sim_msgs, cfg->iterations);

        zw_sig_t sig;
        sig_init(&sig);

        timing_t issue_T;
        memset(&issue_T, 0, sizeof(issue_T));
        for (size_t i = 0; i < cfg->iterations; ++i) {
            CHECK_OK(zw_bidsdvs_issue_timed(issue_msgs + i * MSG_LEN, MSG_LEN, &K, &sig, &issue_T),
                     "timed issue failed");
        }
        issue_us[rep]    = (issue_T.total_issue / (double)cfg->iterations) * 1e6;
        commit_us[rep]   = (issue_T.signer_commitment / (double)cfg->iterations) * 1e6;
        blind_us[rep]    = (issue_T.user_blinding / (double)cfg->iterations) * 1e6;
        response_us[rep] = (issue_T.signer_response / (double)cfg->iterations) * 1e6;
        unblind_us[rep]  = (issue_T.user_unblinding / (double)cfg->iterations) * 1e6;

        verify_item_t *verify_items = verify_dataset_alloc(cfg->iterations);
        build_verify_dataset(verify_items, cfg->iterations, &K);

        timing_t verify_T;
        memset(&verify_T, 0, sizeof(verify_T));
        for (size_t i = 0; i < cfg->iterations; ++i) {
            CHECK_OK(zw_bidsdvs_verify_timed(verify_items[i].msg, MSG_LEN, &K, &verify_items[i].sig, &verify_T),
                     "timed verify failed");
        }
        verify_us[rep] = (verify_T.verify / (double)cfg->iterations) * 1e6;

        zw_sig_t sim_sig;
        sig_init(&sim_sig);
        timing_t sim_T;
        memset(&sim_T, 0, sizeof(sim_T));
        for (size_t i = 0; i < cfg->iterations; ++i) {
            CHECK_OK(zw_bidsdvs_simulate_timed(sim_msgs + i * MSG_LEN, MSG_LEN, &K, &sim_sig, &sim_T),
                     "timed simulate failed");
        }
        simulate_us[rep] = (sim_T.simulate / (double)cfg->iterations) * 1e6;

        sig_free(&sim_sig);
        verify_dataset_free(verify_items, cfg->iterations);
        sig_free(&sig);
        free(issue_msgs);
        free(sim_msgs);
        keys_free(&K);
    }

    stats_t st_setup    = compute_stats_us(setup_us, cfg->repetitions);
    stats_t st_issue    = compute_stats_us(issue_us, cfg->repetitions);
    stats_t st_commit   = compute_stats_us(commit_us, cfg->repetitions);
    stats_t st_blind    = compute_stats_us(blind_us, cfg->repetitions);
    stats_t st_response = compute_stats_us(response_us, cfg->repetitions);
    stats_t st_unblind  = compute_stats_us(unblind_us, cfg->repetitions);
    stats_t st_verify   = compute_stats_us(verify_us, cfg->repetitions);
    stats_t st_sim      = compute_stats_us(simulate_us, cfg->repetitions);

    printf("  --- Reproducible Performance Benchmarks ---\n");
    print_stat_line("Setup + Extract", st_setup, cfg->repetitions);
    print_stat_line("Issue total", st_issue, cfg->repetitions);
    print_stat_line("  Signer commitment", st_commit, cfg->repetitions);
    print_stat_line("  User blinding", st_blind, cfg->repetitions);
    print_stat_line("  Signer response", st_response, cfg->repetitions);
    print_stat_line("  User unblinding", st_unblind, cfg->repetitions);
    print_stat_line("Designated Verify", st_verify, cfg->repetitions);
    print_stat_line("Verifier Simulate", st_sim, cfg->repetitions);
    printf("\n");

    FILE *csv = NULL;
    if (cfg->csv_path) {
        csv = fopen(cfg->csv_path, "w");
        CHECK_OK(csv != NULL, "open csv output");
        fprintf(csv, "scheme,params,operation,mean_us,median_us,stddev_us,throughput_ops_s,repetitions,iterations,warmup,notes\n");
        write_csv_row(csv, "Setup + Extract", st_setup, cfg, "identity extraction measured separately");
        write_csv_row(csv, "Issue total", st_issue, cfg, "blind-signing protocol, message generation excluded");
        write_csv_row(csv, "Issue / Signer commitment", st_commit, cfg, "phase breakdown");
        write_csv_row(csv, "Issue / User blinding", st_blind, cfg, "phase breakdown");
        write_csv_row(csv, "Issue / Signer response", st_response, cfg, "phase breakdown");
        write_csv_row(csv, "Issue / User unblinding", st_unblind, cfg, "phase breakdown");
        write_csv_row(csv, "Designated Verify", st_verify, cfg, "verification dataset pre-generated outside timed region");
        write_csv_row(csv, "Verifier Simulate", st_sim, cfg, "simulator only, message generation excluded");
        fclose(csv);
        printf("CSV results written to: %s\n", cfg->csv_path);
    }

    free(setup_us);
    free(issue_us);
    free(commit_us);
    free(blind_us);
    free(response_us);
    free(unblind_us);
    free(verify_us);
    free(simulate_us);
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------

int main(int argc, char **argv) {
    bench_config_t cfg = parse_args(argc, argv);
    try_pin_core(&cfg);

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

    printf("============================================================\n");
    printf("=== Zhang-Wen BIDSDVS, RELIC Type-III implementation     ===\n");
    printf("============================================================\n\n");

    printf("  --- RELIC Pairing Parameters ---\n");
    pc_param_print();
    printf("\n");

    run_correctness_and_sizes();
    benchmark_all(&cfg);

    printf("  --- Notes ---\n");
    printf("  The original Zhang-Wen paper is written with symmetric pairing notation.\n");
    printf("  This implementation uses RELIC Type-III pairings: G1 x G2 -> GT.\n");
    printf("  Therefore u1_hat and u2_hat are represented in G2, while V_hat is in G1.\n");
    printf("  Random message generation and verification-dataset construction are outside the timed regions.\n\n");

    core_clean();
    return EXIT_SUCCESS;
}
