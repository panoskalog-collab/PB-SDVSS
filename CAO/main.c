
#define _GNU_SOURCE
/*
 * main_cao_repro_benchmark.c
 *
 * Reproducible benchmark harness for the CAO / UDV-PBS pairing-based
 * scheme implemented with RELIC.
 *
 * The cryptographic implementation follows the uploaded baseline:
 *   - asymmetric pairing e: G1 x G2 -> GT,
 *   - partially blind signature (Y', S') in G1,
 *   - designated-verifier signature (Y', phi) with phi in GT.
 *
 * Build example:
 *   gcc -O3 -Wall -Wextra \
 *     -D COMPILE_FLAGS='"-O3 -Wall -Wextra"' \
 *     main.c -o main \
 *     -I/home/panagiotis/relic_install/include \
 *     -L/home/panagiotis/relic_install/lib \
 *     -Wl,-rpath,/home/panagiotis/relic_install/lib \
 *     -lrelic -lgmp -lm
 *
 * Example run:
 *   ./main --iterations 1000 --warmup 100 --repetitions 10 \
 *     --pin-core 0 --csv cao_udv_pbs_results.csv
 */

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__linux__)
#include <sched.h>
#include <sys/utsname.h>
#include <unistd.h>
#endif

#include <relic/relic.h>
#include <relic/relic_core.h>

#ifndef COMPILE_FLAGS
#define COMPILE_FLAGS "not recorded; compile with -D COMPILE_FLAGS='\"...\"'"
#endif

#define DEFAULT_ITERATIONS 1000
#define DEFAULT_WARMUP     100
#define DEFAULT_REPETITIONS 10

#define MSG_LEN 200
#define C_LEN   32

#define CHECK_OK(cond, msg)                                                       \
  do {                                                                            \
    if (!(cond)) {                                                                \
      fprintf(stderr, "Error: %s\n", (msg));                                      \
      exit(EXIT_FAILURE);                                                         \
    }                                                                             \
  } while (0)

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
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
    perror("clock_gettime");
    exit(EXIT_FAILURE);
  }
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static int cmp_double(const void *a, const void *b) {
  double da = *(const double *)a;
  double db = *(const double *)b;
  return (da > db) - (da < db);
}

static stats_t compute_stats(const double *values, int n) {
  CHECK_OK(n > 0, "compute_stats with n <= 0");

  stats_t st;
  st.mean = 0.0;
  st.median = 0.0;
  st.stddev = 0.0;

  for (int i = 0; i < n; ++i) st.mean += values[i];
  st.mean /= (double)n;

  double *copy = (double *)malloc((size_t)n * sizeof(double));
  CHECK_OK(copy != NULL, "malloc stats copy");
  memcpy(copy, values, (size_t)n * sizeof(double));
  qsort(copy, (size_t)n, sizeof(double), cmp_double);

  if ((n % 2) == 1) {
    st.median = copy[n / 2];
  } else {
    st.median = 0.5 * (copy[n / 2 - 1] + copy[n / 2]);
  }

  if (n > 1) {
    double ss = 0.0;
    for (int i = 0; i < n; ++i) {
      double d = values[i] - st.mean;
      ss += d * d;
    }
    st.stddev = sqrt(ss / (double)(n - 1));
  }

  free(copy);
  return st;
}

static long parse_long_strict(const char *s, const char *name) {
  errno = 0;
  char *end = NULL;
  long v = strtol(s, &end, 10);
  if (errno != 0 || !s[0] || *end != '\0') {
    fprintf(stderr, "Invalid %s: %s\n", name, s);
    exit(EXIT_FAILURE);
  }
  return v;
}

static size_t parse_positive_size(const char *s, const char *name) {
  long v = parse_long_strict(s, name);
  if (v <= 0) {
    fprintf(stderr, "%s must be positive: %s\n", name, s);
    exit(EXIT_FAILURE);
  }
  return (size_t)v;
}

static size_t parse_nonnegative_size(const char *s, const char *name) {
  long v = parse_long_strict(s, name);
  if (v < 0) {
    fprintf(stderr, "%s must be non-negative: %s\n", name, s);
    exit(EXIT_FAILURE);
  }
  return (size_t)v;
}

static int parse_positive_int(const char *s, const char *name) {
  long v = parse_long_strict(s, name);
  if (v <= 0 || v > INT_MAX) {
    fprintf(stderr, "%s must be a positive integer: %s\n", name, s);
    exit(EXIT_FAILURE);
  }
  return (int)v;
}

static int parse_nonnegative_int(const char *s, const char *name) {
  long v = parse_long_strict(s, name);
  if (v < 0 || v > INT_MAX) {
    fprintf(stderr, "%s must be a non-negative integer: %s\n", name, s);
    exit(EXIT_FAILURE);
  }
  return (int)v;
}

static void usage(const char *prog) {
  fprintf(stderr,
          "Usage:\n"
          "  %s [options]\n\n"
          "Options:\n"
          "  --iterations N      Timed iterations per repeated run. Default: %d\n"
          "  --warmup W          Untimed warm-up iterations per operation. Default: %d\n"
          "  --repetitions R     Number of repeated timed runs. Default: %d\n"
          "  --pin-core C        Pin process to CPU core C. Core 0 is valid.\n"
          "  --csv PATH          Write CSV summary to PATH.\n"
          "  --help              Show this help message.\n\n"
          "Example:\n"
          "  %s --iterations 1000 --warmup 100 --repetitions 10 --pin-core 0 "
          "--csv cao_udv_pbs_results.csv\n",
          prog, DEFAULT_ITERATIONS, DEFAULT_WARMUP, DEFAULT_REPETITIONS, prog);
}

static void parse_args(int argc, char **argv, bench_config_t *cfg) {
  cfg->iterations = DEFAULT_ITERATIONS;
  cfg->warmup = DEFAULT_WARMUP;
  cfg->repetitions = DEFAULT_REPETITIONS;
  cfg->pin_requested = 0;
  cfg->pin_core = -1;
  cfg->csv_path = NULL;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
      cfg->iterations = parse_positive_size(argv[++i], "iterations");
    } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      cfg->warmup = parse_nonnegative_size(argv[++i], "warmup");
    } else if (strcmp(argv[i], "--repetitions") == 0 && i + 1 < argc) {
      cfg->repetitions = parse_positive_int(argv[++i], "repetitions");
    } else if (strcmp(argv[i], "--pin-core") == 0 && i + 1 < argc) {
      cfg->pin_core = parse_nonnegative_int(argv[++i], "pin-core");
      cfg->pin_requested = 1;
    } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
      cfg->csv_path = argv[++i];
    } else if (strcmp(argv[i], "--help") == 0) {
      usage(argv[0]);
      exit(EXIT_SUCCESS);
    } else {
      usage(argv[0]);
      exit(EXIT_FAILURE);
    }
  }
}

static void try_pin_core(const bench_config_t *cfg) {
  if (!cfg->pin_requested) return;

#if defined(__linux__)
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cfg->pin_core, &set);
  if (sched_setaffinity(0, sizeof(set), &set) != 0) {
    fprintf(stderr, "Warning: failed to pin to core %d: %s\n",
            cfg->pin_core, strerror(errno));
  }
#else
  fprintf(stderr, "Warning: CPU pinning is not supported on this platform.\n");
#endif
}

static long online_cpu_count(void) {
#if defined(__linux__)
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  return n > 0 ? n : -1;
#else
  return -1;
#endif
}

static void print_system_line(void) {
#if defined(__linux__)
  struct utsname u;
  if (uname(&u) == 0) {
    printf("System: %s %s %s %s\n", u.sysname, u.release, u.machine, u.nodename);
  } else {
    printf("System: unavailable\n");
  }
#else
  printf("System: unavailable\n");
#endif
}

static void print_cpu_governors(void) {
#if defined(__linux__)
  long n = online_cpu_count();
  if (n <= 0) return;
  long limit = n < 8 ? n : 8;

  for (long cpu = 0; cpu < limit; ++cpu) {
    char path[256];
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu%ld/cpufreq/scaling_governor", cpu);
    FILE *fp = fopen(path, "r");
    if (!fp) continue;

    char buf[128];
    if (fgets(buf, sizeof(buf), fp)) {
      buf[strcspn(buf, "\r\n")] = '\0';
      printf("CPU governor cpu%ld: %s\n", cpu, buf);
    }
    fclose(fp);
  }
#endif
}

static void print_metadata(const bench_config_t *cfg) {
  printf("CAO / UDV-PBS RELIC benchmark metadata\n");
  printf("============================================================\n");
  if (cfg->pin_requested) {
    printf("Benchmark mode: single-threaded process, requested core %d\n", cfg->pin_core);
  } else {
    printf("Benchmark mode: single-threaded process, no CPU pinning requested\n");
  }
  printf("Timing method: clock_gettime(CLOCK_MONOTONIC), wall-clock seconds\n");
  printf("Message length: %d bytes\n", MSG_LEN);
  printf("Common-info c length: %d bytes\n", C_LEN);
  printf("Iterations per repeated run: %zu\n", cfg->iterations);
  printf("Warm-up iterations: %zu\n", cfg->warmup);
  printf("Repeated timed runs: %d\n", cfg->repetitions);
  printf("Timed region Issue: UDV-PBS issue protocol only; m/c generation excluded\n");
  printf("Timed region Verify: public PB verification only; issue and m/c generation excluded\n");
  printf("Timed region Designate: designation only; issue generation excluded\n");
  printf("Timed region DVVerify: designated verification only; issue/designation generation excluded\n");
  printf("Compiler flags recorded: %s\n", COMPILE_FLAGS);
#if defined(__GNUC__)
  printf("Compiler: GCC-compatible %d.%d.%d\n", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
  printf("Compiler: unavailable\n");
#endif
  long cpus = online_cpu_count();
  if (cpus > 0) printf("CPU online cores: %ld\n", cpus);
  print_system_line();
  print_cpu_governors();
  printf("RELIC parameter set selected by pc_param_set_any(); details printed below.\n");
  printf("============================================================\n\n");
}

static void random_bytes(uint8_t *buf, size_t len) {
  rand_bytes(buf, (int)len);
}

/* -----------------------------------------------------------------------------
 * Hash helpers
 * H0: {0,1}* -> Z_q*
 * H : {0,1}* -> G1
 * -------------------------------------------------------------------------- */

static void hash_to_bn_mod_order(bn_t out, const uint8_t *in, size_t in_len) {
  bn_t t, q;
  bn_null(t); bn_null(q);
  bn_new(t);  bn_new(q);

  g1_get_ord(q);
  bn_read_bin(t, in, (int)in_len);
  bn_mod(out, t, q);
  if (bn_is_zero(out)) bn_set_dig(out, 1);

  bn_free(t);
  bn_free(q);
}

static void H0_bn(bn_t out, const uint8_t *m, size_t mlen, const g1_t Yp) {
  uint8_t ybuf[1024];
  int ylen = g1_size_bin(Yp, 1);
  CHECK_OK(ylen > 0 && ylen <= (int)sizeof(ybuf), "g1_size_bin(Y') too large");

  g1_write_bin(ybuf, ylen, Yp, 1);

  uint8_t h1[32];
  md_map_sh256(h1, m, (int)mlen);

  uint8_t tmp[32 + 1024];
  memcpy(tmp, h1, 32);
  memcpy(tmp + 32, ybuf, (size_t)ylen);

  uint8_t h2[32];
  md_map_sh256(h2, tmp, 32 + ylen);

  hash_to_bn_mod_order(out, h2, sizeof(h2));
}

static void H_to_g1(g1_t out, const uint8_t *in, size_t in_len) {
  uint8_t h[32];
  md_map_sh256(h, in, (int)in_len);
  g1_map(out, h, 32);
}

/* -----------------------------------------------------------------------------
 * Keys and signatures
 * -------------------------------------------------------------------------- */

typedef struct {
  bn_t sk1;
  bn_t sk3;
  g2_t pk1;
  g2_t pk3;
  g1_t P;
  g2_t Q;
} udv_keys_t;

typedef struct {
  g1_t Yp;
  g1_t Sp;
} pb_sig_t;

typedef struct {
  g1_t Yp;
  gt_t phi;
} dv_sig_t;

typedef struct {
  uint8_t m[MSG_LEN];
  uint8_t c[C_LEN];
  pb_sig_t pbs;
  dv_sig_t dvs;
} bench_item_t;

typedef struct {
  int scalar_bytes;
  int g1_bytes;
  int g2_bytes;
  int gt_bytes;
  int pk1_bytes;
  int pk3_bytes;
  int Yp_bytes;
  int Sp_bytes;
  size_t pb_sig_bytes;
  size_t dv_sig_bytes;
} size_info_t;

static void udv_keys_init(udv_keys_t *K) {
  bn_null(K->sk1); bn_null(K->sk3);
  g2_null(K->pk1); g2_null(K->pk3);
  g1_null(K->P);   g2_null(K->Q);

  bn_new(K->sk1); bn_new(K->sk3);
  g2_new(K->pk1); g2_new(K->pk3);
  g1_new(K->P);   g2_new(K->Q);
}

static void udv_keys_free(udv_keys_t *K) {
  (void)K;
  bn_free(K->sk1); bn_free(K->sk3);
  g2_free(K->pk1); g2_free(K->pk3);
  g1_free(K->P);   g2_free(K->Q);
}

static void pb_sig_init(pb_sig_t *s) {
  (void)s;
  g1_null(s->Yp); g1_null(s->Sp);
  g1_new(s->Yp);  g1_new(s->Sp);
}

static void pb_sig_free(pb_sig_t *s) {
  (void)s;
  g1_free(s->Yp);
  g1_free(s->Sp);
}

static void dv_sig_init(dv_sig_t *d) {
  (void)d;
  g1_null(d->Yp); gt_null(d->phi);
  g1_new(d->Yp);  gt_new(d->phi);
}

static void dv_sig_free(dv_sig_t *d) {
  (void)d;
  g1_free(d->Yp);
  gt_free(d->phi);
}

static bench_item_t *dataset_alloc(size_t n) {
  bench_item_t *items = (bench_item_t *)calloc(n, sizeof(*items));
  CHECK_OK(items != NULL, "calloc dataset");
  for (size_t i = 0; i < n; ++i) {
    pb_sig_init(&items[i].pbs);
    dv_sig_init(&items[i].dvs);
  }
  return items;
}

static void dataset_free(bench_item_t *items, size_t n) {
  if (!items) return;
  for (size_t i = 0; i < n; ++i) {
    pb_sig_free(&items[i].pbs);
    dv_sig_free(&items[i].dvs);
  }
  free(items);
}

static void udv_setup_and_keygen(udv_keys_t *K) {
  udv_keys_init(K);

  g1_get_gen(K->P);
  g2_get_gen(K->Q);

  bn_t q;
  bn_null(q); bn_new(q);
  g1_get_ord(q);

  bn_rand_mod(K->sk1, q);
  if (bn_is_zero(K->sk1)) bn_set_dig(K->sk1, 1);
  g2_mul(K->pk1, K->Q, K->sk1);

  bn_rand_mod(K->sk3, q);
  if (bn_is_zero(K->sk3)) bn_set_dig(K->sk3, 1);
  g2_mul(K->pk3, K->Q, K->sk3);

  bn_free(q);
}

/* -----------------------------------------------------------------------------
 * Issue protocol
 * -------------------------------------------------------------------------- */

static int udv_issue_full(const uint8_t *m, size_t mlen,
                          const uint8_t *c, size_t clen,
                          const udv_keys_t *K,
                          pb_sig_t *out_sig) {
  bn_t q;
  bn_null(q); bn_new(q);
  g1_get_ord(q);

  g1_t Z;
  g1_null(Z); g1_new(Z);
  H_to_g1(Z, c, clen);

  bn_t r;
  bn_null(r); bn_new(r);
  bn_rand_mod(r, q);
  if (bn_is_zero(r)) bn_set_dig(r, 1);

  g1_t Y;
  g1_null(Y); g1_new(Y);
  g1_mul(Y, Z, r);

  bn_t alpha, beta;
  bn_null(alpha); bn_null(beta);
  bn_new(alpha);  bn_new(beta);

  bn_rand_mod(alpha, q); if (bn_is_zero(alpha)) bn_set_dig(alpha, 1);
  bn_rand_mod(beta,  q); if (bn_is_zero(beta))  bn_set_dig(beta,  1);

  bn_t alphabeta;
  bn_null(alphabeta); bn_new(alphabeta);
  bn_mul(alphabeta, alpha, beta);
  bn_mod(alphabeta, alphabeta, q);

  g1_t t1, t2;
  g1_null(t1); g1_null(t2);
  g1_new(t1);  g1_new(t2);

  g1_mul(t1, Y, alpha);
  g1_mul(t2, Z, alphabeta);
  g1_add(out_sig->Yp, t1, t2);
  g1_norm(out_sig->Yp, out_sig->Yp);

  bn_t H0v;
  bn_null(H0v); bn_new(H0v);
  H0_bn(H0v, m, mlen, out_sig->Yp);

  bn_t alpha_inv;
  bn_null(alpha_inv); bn_new(alpha_inv);
  bn_mod_inv(alpha_inv, alpha, q);

  bn_t h;
  bn_null(h); bn_new(h);
  bn_mul(h, alpha_inv, H0v);
  bn_mod(h, h, q);
  bn_add(h, h, beta);
  bn_mod(h, h, q);

  bn_t rh, rhs;
  bn_null(rh);  bn_null(rhs);
  bn_new(rh);   bn_new(rhs);

  bn_add(rh, r, h);
  bn_mod(rh, rh, q);

  bn_mul(rhs, rh, K->sk1);
  bn_mod(rhs, rhs, q);

  g1_t S;
  g1_null(S); g1_new(S);
  g1_mul(S, Z, rhs);

  g1_mul(out_sig->Sp, S, alpha);
  g1_norm(out_sig->Sp, out_sig->Sp);

  g1_free(S);
  bn_free(rhs); bn_free(rh);
  bn_free(h); bn_free(alpha_inv); bn_free(H0v);
  g1_free(t1); g1_free(t2);
  bn_free(alphabeta);
  bn_free(alpha); bn_free(beta);
  g1_free(Y);
  bn_free(r);
  g1_free(Z);
  bn_free(q);

  return 1;
}

static int udv_verify_pb(const uint8_t *m, size_t mlen,
                         const uint8_t *c, size_t clen,
                         const udv_keys_t *K,
                         const pb_sig_t *sig) {
  g1_t Z;
  g1_null(Z); g1_new(Z);
  H_to_g1(Z, c, clen);

  bn_t H0v, q;
  bn_null(H0v); bn_null(q);
  bn_new(H0v);  bn_new(q);
  g1_get_ord(q);
  H0_bn(H0v, m, mlen, sig->Yp);

  g1_t H0Z, T;
  g1_null(H0Z); g1_null(T);
  g1_new(H0Z);  g1_new(T);

  g1_mul(H0Z, Z, H0v);
  g1_add(T, sig->Yp, H0Z);
  g1_norm(T, T);

  gt_t lhs, rhs;
  gt_null(lhs); gt_null(rhs);
  gt_new(lhs);  gt_new(rhs);

  pc_map(lhs, sig->Sp, K->Q);
  pc_map(rhs, T, K->pk1);

  int ok = (gt_cmp(lhs, rhs) == RLC_EQ);

  gt_free(lhs); gt_free(rhs);
  g1_free(H0Z); g1_free(T);
  g1_free(Z);
  bn_free(H0v); bn_free(q);

  return ok;
}

static int udv_designate(const udv_keys_t *K, const pb_sig_t *sig, dv_sig_t *out_dv) {
  g1_copy(out_dv->Yp, sig->Yp);
  pc_map(out_dv->phi, sig->Sp, K->pk3);
  return 1;
}

static int udv_designated_verify(const uint8_t *m, size_t mlen,
                                 const uint8_t *c, size_t clen,
                                 const udv_keys_t *K,
                                 const dv_sig_t *dv) {
  g1_t Z;
  g1_null(Z); g1_new(Z);
  H_to_g1(Z, c, clen);

  bn_t H0v, q;
  bn_null(H0v); bn_null(q);
  bn_new(H0v);  bn_new(q);
  g1_get_ord(q);
  H0_bn(H0v, m, mlen, dv->Yp);

  g1_t H0Z, T;
  g1_null(H0Z); g1_null(T);
  g1_new(H0Z);  g1_new(T);

  g1_mul(H0Z, Z, H0v);
  g1_add(T, dv->Yp, H0Z);
  g1_norm(T, T);

  g1_t sk3T;
  g1_null(sk3T); g1_new(sk3T);
  g1_mul(sk3T, T, K->sk3);
  g1_norm(sk3T, sk3T);

  gt_t rhs;
  gt_null(rhs); gt_new(rhs);
  pc_map(rhs, sk3T, K->pk1);

  int ok = (gt_cmp(dv->phi, rhs) == RLC_EQ);

  gt_free(rhs);
  g1_free(sk3T);
  g1_free(H0Z); g1_free(T);
  g1_free(Z);
  bn_free(H0v); bn_free(q);

  return ok;
}

/* -----------------------------------------------------------------------------
 * Sizes and datasets
 * -------------------------------------------------------------------------- */

static size_info_t collect_sizes(const udv_keys_t *K, const pb_sig_t *pbs, const dv_sig_t *dvs) {
  size_info_t si;
  memset(&si, 0, sizeof(si));

  bn_t q;
  bn_null(q); bn_new(q);
  g1_get_ord(q);

  si.scalar_bytes = bn_size_bin(q);
  si.g1_bytes = g1_size_bin(K->P, 1);
  si.g2_bytes = g2_size_bin(K->Q, 1);
  si.gt_bytes = gt_size_bin(dvs->phi, 1);
  si.pk1_bytes = g2_size_bin(K->pk1, 1);
  si.pk3_bytes = g2_size_bin(K->pk3, 1);
  si.Yp_bytes = g1_size_bin(pbs->Yp, 1);
  si.Sp_bytes = g1_size_bin(pbs->Sp, 1);
  si.pb_sig_bytes = (size_t)si.Yp_bytes + (size_t)si.Sp_bytes;
  si.dv_sig_bytes = (size_t)si.Yp_bytes + (size_t)si.gt_bytes;

  bn_free(q);
  return si;
}

static void print_sizes(const size_info_t *si) {
  printf("  Size summary:\n");
  printf("    scalar size (Z_q):                 %d bytes (%d bits)\n",
         si->scalar_bytes, si->scalar_bytes * 8);
  printf("    G1 element (compressed):           %d bytes (%d bits)\n",
         si->g1_bytes, si->g1_bytes * 8);
  printf("    G2 element (compressed):           %d bytes (%d bits)\n",
         si->g2_bytes, si->g2_bytes * 8);
  printf("    pk1 (signer pub, in G2):           %d bytes (%d bits)\n",
         si->pk1_bytes, si->pk1_bytes * 8);
  printf("    pk3 (verifier pub, in G2):         %d bytes (%d bits)\n",
         si->pk3_bytes, si->pk3_bytes * 8);
  printf("    sk1, sk3 (scalars):                %d bytes each (%d bits)\n",
         si->scalar_bytes, si->scalar_bytes * 8);
  printf("    Partially blind sig (Y', S'):      %zu bytes (%zu bits)\n",
         si->pb_sig_bytes, si->pb_sig_bytes * 8);
  printf("      breakdown: Y'=%d, S'=%d\n", si->Yp_bytes, si->Sp_bytes);
  printf("    DV sig (Y', phi in GT):            %zu bytes (%zu bits)\n",
         si->dv_sig_bytes, si->dv_sig_bytes * 8);
  printf("      breakdown: Y'=%d, phi=%d\n", si->Yp_bytes, si->gt_bytes);
}

static void build_dataset(bench_item_t *items, size_t n, const udv_keys_t *K) {
  printf("Building CAO/UDV-PBS dataset outside timed region...\n");
  for (size_t i = 0; i < n; ++i) {
    random_bytes(items[i].m, MSG_LEN);
    random_bytes(items[i].c, C_LEN);

    CHECK_OK(udv_issue_full(items[i].m, MSG_LEN, items[i].c, C_LEN,
                            K, &items[i].pbs),
             "dataset Issue");
    CHECK_OK(udv_verify_pb(items[i].m, MSG_LEN, items[i].c, C_LEN,
                           K, &items[i].pbs),
             "dataset PB Verify");
    CHECK_OK(udv_designate(K, &items[i].pbs, &items[i].dvs),
             "dataset Designate");
    CHECK_OK(udv_designated_verify(items[i].m, MSG_LEN, items[i].c, C_LEN,
                                   K, &items[i].dvs),
             "dataset DV Verify");
  }
  printf("Dataset ready: %zu valid messages/signatures/designated signatures.\n\n", n);
}

/* -----------------------------------------------------------------------------
 * Operation benchmarks
 * -------------------------------------------------------------------------- */

static double run_issue_once(const bench_item_t *items, size_t n,
                             size_t warmup, const udv_keys_t *K) {
  pb_sig_t tmp;
  pb_sig_init(&tmp);

  for (size_t i = 0; i < warmup; ++i) {
    size_t j = i % n;
    CHECK_OK(udv_issue_full(items[j].m, MSG_LEN, items[j].c, C_LEN, K, &tmp),
             "warmup Issue");
  }

  double t0 = now_seconds();
  for (size_t i = 0; i < n; ++i) {
    CHECK_OK(udv_issue_full(items[i].m, MSG_LEN, items[i].c, C_LEN, K, &tmp),
             "Issue");
  }
  double t1 = now_seconds();

  pb_sig_free(&tmp);
  return t1 - t0;
}

static double run_verify_once(const bench_item_t *items, size_t n,
                              size_t warmup, const udv_keys_t *K) {
  for (size_t i = 0; i < warmup; ++i) {
    size_t j = i % n;
    CHECK_OK(udv_verify_pb(items[j].m, MSG_LEN, items[j].c, C_LEN, K, &items[j].pbs),
             "warmup Verify");
  }

  double t0 = now_seconds();
  for (size_t i = 0; i < n; ++i) {
    CHECK_OK(udv_verify_pb(items[i].m, MSG_LEN, items[i].c, C_LEN, K, &items[i].pbs),
             "Verify");
  }
  double t1 = now_seconds();

  return t1 - t0;
}

static double run_designate_once(const bench_item_t *items, size_t n,
                                 size_t warmup, const udv_keys_t *K) {
  dv_sig_t tmp;
  dv_sig_init(&tmp);

  for (size_t i = 0; i < warmup; ++i) {
    size_t j = i % n;
    CHECK_OK(udv_designate(K, &items[j].pbs, &tmp), "warmup Designate");
  }

  double t0 = now_seconds();
  for (size_t i = 0; i < n; ++i) {
    CHECK_OK(udv_designate(K, &items[i].pbs, &tmp), "Designate");
  }
  double t1 = now_seconds();

  dv_sig_free(&tmp);
  return t1 - t0;
}

static double run_dvverify_once(const bench_item_t *items, size_t n,
                                size_t warmup, const udv_keys_t *K) {
  for (size_t i = 0; i < warmup; ++i) {
    size_t j = i % n;
    CHECK_OK(udv_designated_verify(items[j].m, MSG_LEN, items[j].c, C_LEN,
                                   K, &items[j].dvs),
             "warmup DVVerify");
  }

  double t0 = now_seconds();
  for (size_t i = 0; i < n; ++i) {
    CHECK_OK(udv_designated_verify(items[i].m, MSG_LEN, items[i].c, C_LEN,
                                   K, &items[i].dvs),
             "DVVerify");
  }
  double t1 = now_seconds();

  return t1 - t0;
}

typedef double (*operation_runner_t)(const bench_item_t *, size_t, size_t, const udv_keys_t *);

static stats_t benchmark_operation(const char *label,
                                   operation_runner_t runner,
                                   const bench_config_t *cfg,
                                   const bench_item_t *items,
                                   const udv_keys_t *K,
                                   double *throughput_out) {
  double *per_run_us = (double *)calloc((size_t)cfg->repetitions, sizeof(double));
  CHECK_OK(per_run_us != NULL, "calloc per_run_us");

  for (int r = 0; r < cfg->repetitions; ++r) {
    double elapsed = runner(items, cfg->iterations, cfg->warmup, K);
    if (elapsed <= 0.0) elapsed = 1e-9;
    per_run_us[r] = (elapsed * 1e6) / (double)cfg->iterations;
  }

  stats_t st = compute_stats(per_run_us, cfg->repetitions);
  *throughput_out = st.mean > 0.0 ? 1e6 / st.mean : 0.0;

  printf("  Algorithm %s, single-threaded:\n", label);
  printf("    mean:       %.3f us/op\n", st.mean);
  printf("    median:     %.3f us/op\n", st.median);
  printf("    std. dev.:  %.3f us/op over %d repeated runs\n",
         st.stddev, cfg->repetitions);
  printf("    throughput: %.2f ops/sec, based on mean\n\n", *throughput_out);

  free(per_run_us);
  return st;
}

static void write_csv(const char *path,
                      const bench_config_t *cfg,
                      const size_info_t *si,
                      const stats_t *issue, double issue_thr,
                      const stats_t *verify, double verify_thr,
                      const stats_t *designate, double designate_thr,
                      const stats_t *dvverify, double dvverify_thr) {
  if (!path) return;

  FILE *fp = fopen(path, "w");
  if (!fp) {
    fprintf(stderr, "Warning: could not open CSV file '%s': %s\n",
            path, strerror(errno));
    return;
  }

  fprintf(fp,
          "scheme,implementation,parameter_set,operation,iterations,warmup,repetitions,"
          "mean_us,median_us,stddev_us,throughput_ops_sec,signature_bytes,notes\n");

  fprintf(fp,
          "CAO_UDV_PBS,RELIC,pc_param_set_any,Issue,%zu,%zu,%d,"
          "%.6f,%.6f,%.6f,%.6f,%zu,PB_signature_Yp_Sp\n",
          cfg->iterations, cfg->warmup, cfg->repetitions,
          issue->mean, issue->median, issue->stddev, issue_thr, si->pb_sig_bytes);

  fprintf(fp,
          "CAO_UDV_PBS,RELIC,pc_param_set_any,VerifyPB,%zu,%zu,%d,"
          "%.6f,%.6f,%.6f,%.6f,%zu,public_pairing_verify\n",
          cfg->iterations, cfg->warmup, cfg->repetitions,
          verify->mean, verify->median, verify->stddev, verify_thr, si->pb_sig_bytes);

  fprintf(fp,
          "CAO_UDV_PBS,RELIC,pc_param_set_any,Designate,%zu,%zu,%d,"
          "%.6f,%.6f,%.6f,%.6f,%zu,phi_equals_pairing\n",
          cfg->iterations, cfg->warmup, cfg->repetitions,
          designate->mean, designate->median, designate->stddev,
          designate_thr, si->dv_sig_bytes);

  fprintf(fp,
          "CAO_UDV_PBS,RELIC,pc_param_set_any,DVVerify,%zu,%zu,%d,"
          "%.6f,%.6f,%.6f,%.6f,%zu,designated_pairing_verify\n",
          cfg->iterations, cfg->warmup, cfg->repetitions,
          dvverify->mean, dvverify->median, dvverify->stddev,
          dvverify_thr, si->dv_sig_bytes);

  fclose(fp);
  printf("CSV results written to: %s\n", path);
}

int main(int argc, char **argv) {
  bench_config_t cfg;
  parse_args(argc, argv, &cfg);
  try_pin_core(&cfg);

  if (core_init() != RLC_OK) {
    core_clean();
    fprintf(stderr, "RELIC core_init failed\n");
    return EXIT_FAILURE;
  }

  if (pc_param_set_any() != RLC_OK) {
    fprintf(stderr, "RELIC pc_param_set_any failed (pairing params not available)\n");
    core_clean();
    return EXIT_FAILURE;
  }

  print_metadata(&cfg);

  printf("--- RELIC pairing parameters ---\n");
  pc_param_print();
  printf("--------------------------------\n\n");

  printf("=== CAO / UDV-Partially-Blind Signatures, pairing-based RELIC ===\n");

  udv_keys_t K;
  udv_setup_and_keygen(&K);

  bench_item_t *items = dataset_alloc(cfg.iterations);
  build_dataset(items, cfg.iterations, &K);

  size_info_t si = collect_sizes(&K, &items[0].pbs, &items[0].dvs);
  print_sizes(&si);
  printf("\n");

  double issue_thr = 0.0;
  double verify_thr = 0.0;
  double designate_thr = 0.0;
  double dvverify_thr = 0.0;

  stats_t issue = benchmark_operation("Issue full protocol -> (Y',S')",
                                      run_issue_once, &cfg, items, &K, &issue_thr);

  stats_t verify = benchmark_operation("Verify public PB signature",
                                       run_verify_once, &cfg, items, &K, &verify_thr);

  stats_t designate = benchmark_operation("Designate phi = e(S', pk3)",
                                          run_designate_once, &cfg, items, &K,
                                          &designate_thr);

  stats_t dvverify = benchmark_operation("Designated Verification",
                                         run_dvverify_once, &cfg, items, &K,
                                         &dvverify_thr);

  write_csv(cfg.csv_path, &cfg, &si,
            &issue, issue_thr,
            &verify, verify_thr,
            &designate, designate_thr,
            &dvverify, dvverify_thr);

  dataset_free(items, cfg.iterations);
  udv_keys_free(&K);
  core_clean();

  return EXIT_SUCCESS;
}
