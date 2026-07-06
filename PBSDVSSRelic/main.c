#define _GNU_SOURCE

// main_relic_repro_benchmark.c
// Reproducible PB-SDVSS benchmark in RELIC over the active pairing parameters.
// Intended use: BN-P256 pairing parameters with PB-SDVSS implemented only in G1.
// No pairing operation is invoked by PB-SDVSS.Sign or PB-SDVSS.Verify.
//
/*
Example build:
  gcc -O3 -Wall -Wextra \
    -D COMPILE_FLAGS='"-O3 -Wall -Wextra"' \
    main.c -o main \
    -I/home/panagiotis/relic_install/include \
    -L/home/panagiotis/relic_install/lib \
    -Wl,-rpath,/home/panagiotis/relic_install/lib \
    -lrelic -lgmp -lm

Example run:
  ./main --iterations 1000 --warmup 100 --repetitions 10 \
         --pin-core 0 --csv pb_sdvss_relic_results.csv | tee pb_sdvss_relic_output.txt
*/

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifdef __linux__
#include <sched.h>
#include <sys/utsname.h>
#endif

#include <relic/relic.h>

#ifndef COMPILE_FLAGS
#define COMPILE_FLAGS "unknown"
#endif

#define DEFAULT_ITERATIONS  1000
#define DEFAULT_WARMUP      100
#define DEFAULT_REPETITIONS 10

#define MSG_LEN    200   // VANET-like beacon length
#define INFO_LEN   32    // metadata/info length

#define CHECK_OK(cond, msg) \
  do { \
    if (!(cond)) { \
      fprintf(stderr, "Error: %s\n", (msg)); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

typedef struct {
  int iterations;
  int warmup;
  int repetitions;
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

// -----------------------------------------------------------------------------
// Timing and platform helpers
// -----------------------------------------------------------------------------

static double now_seconds(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
    perror("clock_gettime(CLOCK_MONOTONIC)");
    exit(EXIT_FAILURE);
  }
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void *xmalloc(size_t n) {
  void *p = malloc(n);
  if (p == NULL) {
    fprintf(stderr, "malloc failed for %zu bytes\n", n);
    exit(EXIT_FAILURE);
  }
  return p;
}

static int parse_positive_int(const char *s, const char *name) {
  char *end = NULL;
  errno = 0;
  long v = strtol(s, &end, 10);
  if (errno != 0 || s[0] == '\0' || *end != '\0' || v <= 0 || v > INT_MAX) {
    fprintf(stderr, "Invalid %s: %s\n", name, s);
    exit(EXIT_FAILURE);
  }
  return (int)v;
}

static int parse_nonnegative_int(const char *s, const char *name) {
  char *end = NULL;
  errno = 0;
  long v = strtol(s, &end, 10);
  if (errno != 0 || s[0] == '\0' || *end != '\0' || v < 0 || v > INT_MAX) {
    fprintf(stderr, "Invalid %s: %s\n", name, s);
    exit(EXIT_FAILURE);
  }
  return (int)v;
}

static void usage(const char *prog) {
  printf("Usage: %s [options]\n", prog);
  printf("Options:\n");
  printf("  --iterations N      Timed iterations per repeated run, default %d\n", DEFAULT_ITERATIONS);
  printf("  --warmup W          Untimed warm-up iterations, default %d\n", DEFAULT_WARMUP);
  printf("  --repetitions R     Number of repeated timed runs, default %d\n", DEFAULT_REPETITIONS);
  printf("  --pin-core C        Pin process to CPU core C. Core 0 is valid.\n");
  printf("  --csv PATH          Write machine-readable CSV results to PATH.\n");
  printf("  --help              Show this help text.\n");
}

static bench_config_t parse_args(int argc, char **argv) {
  bench_config_t cfg;
  cfg.iterations = DEFAULT_ITERATIONS;
  cfg.warmup = DEFAULT_WARMUP;
  cfg.repetitions = DEFAULT_REPETITIONS;
  cfg.pin_requested = 0;
  cfg.pin_core = -1;
  cfg.csv_path = NULL;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
      cfg.iterations = parse_positive_int(argv[++i], "iterations");
    } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      cfg.warmup = parse_nonnegative_int(argv[++i], "warmup");
    } else if (strcmp(argv[i], "--repetitions") == 0 && i + 1 < argc) {
      cfg.repetitions = parse_positive_int(argv[++i], "repetitions");
    } else if (strcmp(argv[i], "--pin-core") == 0 && i + 1 < argc) {
      cfg.pin_core = parse_nonnegative_int(argv[++i], "pin-core");
      cfg.pin_requested = 1;
    } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
      cfg.csv_path = argv[++i];
    } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      usage(argv[0]);
      exit(EXIT_SUCCESS);
    } else {
      fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
      usage(argv[0]);
      exit(EXIT_FAILURE);
    }
  }

  return cfg;
}

static void maybe_pin_core(const bench_config_t *cfg) {
  if (!cfg->pin_requested) {
    return;
  }

#ifdef __linux__
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cfg->pin_core, &set);

  if (sched_setaffinity(0, sizeof(set), &set) != 0) {
    fprintf(stderr,
            "Warning: could not pin process to core %d: %s\n",
            cfg->pin_core,
            strerror(errno));
  }
#else
  fprintf(stderr, "Warning: CPU pinning is only implemented for Linux.\n");
#endif
}

static void print_cpu_governors(void) {
#ifdef __linux__
  long cores = sysconf(_SC_NPROCESSORS_ONLN);
  if (cores < 1) {
    return;
  }

  long limit = cores < 8 ? cores : 8;  // keep metadata compact
  for (long i = 0; i < limit; i++) {
    char path[256];
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu%ld/cpufreq/scaling_governor", i);

    FILE *fp = fopen(path, "r");
    if (!fp) {
      continue;
    }

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
  printf("PB-SDVSS RELIC benchmark metadata\n");
  printf("============================================================\n");
  if (cfg->pin_requested) {
    printf("Benchmark mode: single-threaded process, requested core %d\n", cfg->pin_core);
  } else {
    printf("Benchmark mode: single-threaded process, no core pinning requested\n");
  }
  printf("Timing method: clock_gettime(CLOCK_MONOTONIC), wall-clock seconds\n");
  printf("Message length: %d bytes\n", MSG_LEN);
  printf("Info length: %d bytes\n", INFO_LEN);
  printf("Iterations per repeated run: %d\n", cfg->iterations);
  printf("Warm-up iterations: %d\n", cfg->warmup);
  printf("Repeated timed runs: %d\n", cfg->repetitions);
  printf("Timed region Sign: PB-SDVSS.Sign only; message/info generation excluded\n");
  printf("Timed region Verify: PB-SDVSS.Verify only; message/info/signature generation excluded\n");
  printf("Compiler flags recorded: %s\n", COMPILE_FLAGS);

#if defined(__clang__)
  printf("Compiler: Clang %d.%d.%d\n", __clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(__GNUC__)
  printf("Compiler: GCC-compatible %d.%d.%d\n", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
  printf("Compiler: unknown\n");
#endif

#ifdef __linux__
  long cores = sysconf(_SC_NPROCESSORS_ONLN);
  if (cores > 0) {
    printf("CPU online cores: %ld\n", cores);
  }

  struct utsname uts;
  if (uname(&uts) == 0) {
    printf("System: %s %s %s %s\n", uts.sysname, uts.release, uts.machine, uts.nodename);
  }
#endif

  print_cpu_governors();
  printf("RELIC parameter set selected by pc_param_set_any(); details printed below.\n");
  printf("============================================================\n\n");
}

// -----------------------------------------------------------------------------
// Statistics
// -----------------------------------------------------------------------------

static int cmp_double(const void *a, const void *b) {
  const double da = *(const double *)a;
  const double db = *(const double *)b;
  return (da > db) - (da < db);
}

static stats_t compute_stats(const double *values, int n) {
  stats_t st;
  st.mean_us = 0.0;
  st.median_us = 0.0;
  st.stddev_us = 0.0;
  st.throughput_ops_s = 0.0;

  for (int i = 0; i < n; i++) {
    st.mean_us += values[i];
  }
  st.mean_us /= (double)n;

  double *copy = (double *)xmalloc((size_t)n * sizeof(double));
  memcpy(copy, values, (size_t)n * sizeof(double));
  qsort(copy, (size_t)n, sizeof(double), cmp_double);

  if (n % 2 == 1) {
    st.median_us = copy[n / 2];
  } else {
    st.median_us = 0.5 * (copy[n / 2 - 1] + copy[n / 2]);
  }
  free(copy);

  if (n > 1) {
    double acc = 0.0;
    for (int i = 0; i < n; i++) {
      double d = values[i] - st.mean_us;
      acc += d * d;
    }
    st.stddev_us = sqrt(acc / (double)(n - 1));
  }

  if (st.mean_us > 0.0) {
    st.throughput_ops_s = 1e6 / st.mean_us;
  }

  return st;
}

// -----------------------------------------------------------------------------
// Random data generator. RELIC rand_bytes is not included in timed regions.
// -----------------------------------------------------------------------------

static void random_bytes(uint8_t *buf, size_t len) {
  rand_bytes(buf, (int)len);
}

static void make_dataset(uint8_t *msgs, uint8_t *infos, int n) {
  for (int i = 0; i < n; i++) {
    random_bytes(msgs + (size_t)i * MSG_LEN, MSG_LEN);
    random_bytes(infos + (size_t)i * INFO_LEN, INFO_LEN);
  }
}

// -----------------------------------------------------------------------------
// Hash to scalar mod q
// -----------------------------------------------------------------------------

static void hash_to_bn(bn_t out, const uint8_t *in, size_t in_len) {
  uint8_t h[32];
  md_map_sh256(h, in, (int)in_len);

  bn_t q, t;
  bn_null(q); bn_null(t);
  bn_new(q);  bn_new(t);

  g1_get_ord(q);
  bn_read_bin(t, h, 32);
  bn_mod(out, t, q);

  bn_free(t);
  bn_free(q);
}

// H(m, info, x*) -> Z_q using bytes of (m || info || enc(x*))
static void hash_m_info_xstar(bn_t out,
                              const uint8_t *m, size_t mlen,
                              const uint8_t *info, size_t ilen,
                              const g1_t x_star) {
  uint8_t xb[1024];
  int xlen = g1_size_bin(x_star, 1);
  CHECK_OK(xlen > 0 && xlen <= (int)sizeof(xb), "x* encoding too large");
  g1_write_bin(xb, xlen, x_star, 1);

  size_t tot = mlen + ilen + (size_t)xlen;
  uint8_t *tmp = (uint8_t *)xmalloc(tot);

  memcpy(tmp, m, mlen);
  memcpy(tmp + mlen, info, ilen);
  memcpy(tmp + mlen + ilen, xb, (size_t)xlen);

  hash_to_bn(out, tmp, tot);
  free(tmp);
}

// -----------------------------------------------------------------------------
// PB-SDVSS keys and signatures
// -----------------------------------------------------------------------------

typedef struct {
  bn_t s1, s2;  // signer secret
  bn_t s;       // verifier secret
  g1_t g1, g2;  // generators in G1
  g1_t v;       // pkS
  g1_t k;       // pkV
} pb_keys_t;

typedef struct {
  g1_t x_star;
  bn_t a_star;
  g1_t sig1;
  bn_t sig2;
} pb_sig_t;

static void keys_init(pb_keys_t *K) {
  bn_null(K->s1); bn_null(K->s2); bn_null(K->s);
  g1_null(K->g1); g1_null(K->g2); g1_null(K->v); g1_null(K->k);

  bn_new(K->s1); bn_new(K->s2); bn_new(K->s);
  g1_new(K->g1); g1_new(K->g2); g1_new(K->v); g1_new(K->k);
}

static void keys_free(pb_keys_t *K) {
  (void)K;
  bn_free(K->s1); bn_free(K->s2); bn_free(K->s);
  g1_free(K->g1); g1_free(K->g2); g1_free(K->v); g1_free(K->k);
}

static void sig_init(pb_sig_t *S) {
  g1_null(S->x_star); g1_null(S->sig1);
  bn_null(S->a_star); bn_null(S->sig2);

  g1_new(S->x_star); g1_new(S->sig1);
  bn_new(S->a_star); bn_new(S->sig2);
}

static void sig_free(pb_sig_t *S) {
  (void)S;
  g1_free(S->x_star); g1_free(S->sig1);
  bn_free(S->a_star); bn_free(S->sig2);
}

// Setup, G1-only instantiation:
// g1 = generator, g2 = h*g1 where h = H("g2") mod q,
// v = (-s1)*g1 + (-s2)*g2, k = s*g1.
static void pb_setup(pb_keys_t *K) {
  keys_init(K);

  bn_t q;
  bn_null(q); bn_new(q);
  g1_get_ord(q);

  // g1 = generator
  g1_get_gen(K->g1);

  // g2 = h*g1 with h = H("g2") mod q
  bn_t h;
  bn_null(h); bn_new(h);
  const uint8_t label[] = "g2";
  hash_to_bn(h, label, sizeof(label) - 1);
  bn_mod(h, h, q);
  if (bn_is_zero(h)) {
    bn_set_dig(h, 1);
  }

  g1_mul(K->g2, K->g1, h);
  g1_norm(K->g2, K->g2);

  // Secrets
  bn_rand_mod(K->s1, q); if (bn_is_zero(K->s1)) bn_set_dig(K->s1, 1);
  bn_rand_mod(K->s2, q); if (bn_is_zero(K->s2)) bn_set_dig(K->s2, 1);
  bn_rand_mod(K->s,  q); if (bn_is_zero(K->s))  bn_set_dig(K->s,  1);

  // v = (-s1)*g1 + (-s2)*g2
  bn_t ns1, ns2;
  bn_null(ns1); bn_null(ns2);
  bn_new(ns1);  bn_new(ns2);

  bn_neg(ns1, K->s1); bn_mod(ns1, ns1, q);
  bn_neg(ns2, K->s2); bn_mod(ns2, ns2, q);

  g1_t t1, t2;
  g1_null(t1); g1_null(t2);
  g1_new(t1);  g1_new(t2);

  g1_mul(t1, K->g1, ns1);
  g1_mul(t2, K->g2, ns2);
  g1_add(K->v, t1, t2);
  g1_norm(K->v, K->v);

  // k = s*g1
  g1_mul(K->k, K->g1, K->s);
  g1_norm(K->k, K->k);

  g1_free(t1); g1_free(t2);
  bn_free(ns1); bn_free(ns2);
  bn_free(h);
  bn_free(q);
}

// PB-SDVSS.Sign, additive notation:
// x = r1*g1 + r2*g2
// x* = x + u1*g1 + u2*g2 + d*v
// a* = H(m, info, x*)
// z = H(info)
// alpha_bar = a* - d + z
// alpha = alpha_bar - z
// y1 = r1 + alpha*s1
// y2 = r2 + alpha*s2
// bsig1 = y1*k
// sig1 = bsig1 + u1*k
// sig2 = y2 + u2
static int pb_sign(const uint8_t *m, size_t mlen,
                   const uint8_t *info, size_t ilen,
                   const pb_keys_t *K, pb_sig_t *S) {
  bn_t q;
  bn_null(q); bn_new(q);
  g1_get_ord(q);

  bn_t r1, r2, u1, u2, d, z, alpha_bar, alpha, y1, y2, tmp;
  bn_null(r1); bn_null(r2); bn_null(u1); bn_null(u2); bn_null(d);
  bn_null(z); bn_null(alpha_bar); bn_null(alpha); bn_null(y1); bn_null(y2); bn_null(tmp);

  bn_new(r1); bn_new(r2); bn_new(u1); bn_new(u2); bn_new(d);
  bn_new(z);  bn_new(alpha_bar); bn_new(alpha); bn_new(y1); bn_new(y2); bn_new(tmp);

  // r1, r2
  bn_rand_mod(r1, q); if (bn_is_zero(r1)) bn_set_dig(r1, 1);
  bn_rand_mod(r2, q); if (bn_is_zero(r2)) bn_set_dig(r2, 1);

  // x = r1*g1 + r2*g2
  g1_t x, t1, t2, t3, bsig1;
  g1_null(x); g1_null(t1); g1_null(t2); g1_null(t3); g1_null(bsig1);
  g1_new(x);  g1_new(t1);  g1_new(t2);  g1_new(t3);  g1_new(bsig1);

  g1_mul(t1, K->g1, r1);
  g1_mul(t2, K->g2, r2);
  g1_add(x, t1, t2);
  g1_norm(x, x);

  // u1, u2, d
  bn_rand_mod(u1, q); if (bn_is_zero(u1)) bn_set_dig(u1, 1);
  bn_rand_mod(u2, q); if (bn_is_zero(u2)) bn_set_dig(u2, 1);
  bn_rand_mod(d,  q); if (bn_is_zero(d))  bn_set_dig(d,  1);

  // x* = x + u1*g1 + u2*g2 + d*v
  g1_mul(t1, K->g1, u1);
  g1_mul(t2, K->g2, u2);
  g1_mul(t3, K->v,  d);

  g1_copy(S->x_star, x);
  g1_add(S->x_star, S->x_star, t1);
  g1_add(S->x_star, S->x_star, t2);
  g1_add(S->x_star, S->x_star, t3);
  g1_norm(S->x_star, S->x_star);

  // a* = H(m, info, x*)
  hash_m_info_xstar(S->a_star, m, mlen, info, ilen, S->x_star);

  // z = H(info)
  hash_to_bn(z, info, ilen);

  // alpha_bar = a* - d + z
  bn_sub(tmp, S->a_star, d); bn_mod(tmp, tmp, q);
  bn_add(alpha_bar, tmp, z); bn_mod(alpha_bar, alpha_bar, q);

  // alpha = alpha_bar - z
  bn_sub(alpha, alpha_bar, z); bn_mod(alpha, alpha, q);

  // y1 = r1 + alpha*s1
  bn_mul(tmp, alpha, K->s1); bn_mod(tmp, tmp, q);
  bn_add(y1, r1, tmp);       bn_mod(y1, y1, q);

  // y2 = r2 + alpha*s2
  bn_mul(tmp, alpha, K->s2); bn_mod(tmp, tmp, q);
  bn_add(y2, r2, tmp);       bn_mod(y2, y2, q);

  // bsig1 = y1*k
  g1_mul(bsig1, K->k, y1);

  // sig1 = bsig1 + u1*k
  g1_mul(t1, K->k, u1);
  g1_add(S->sig1, bsig1, t1);
  g1_norm(S->sig1, S->sig1);

  // sig2 = y2 + u2
  bn_add(S->sig2, y2, u2);
  bn_mod(S->sig2, S->sig2, q);

  g1_free(x); g1_free(t1); g1_free(t2); g1_free(t3); g1_free(bsig1);
  bn_free(r1); bn_free(r2); bn_free(u1); bn_free(u2); bn_free(d);
  bn_free(z); bn_free(alpha_bar); bn_free(alpha); bn_free(y1); bn_free(y2); bn_free(tmp);
  bn_free(q);

  return 1;
}

// PB-SDVSS.Verify:
// check a* == H(m, info, x*)
// check s*x* == sig1 + (sig2*s)g2 + (a* s)v
static int pb_verify(const uint8_t *m, size_t mlen,
                     const uint8_t *info, size_t ilen,
                     const pb_keys_t *K, const pb_sig_t *S) {
  bn_t q;
  bn_null(q); bn_new(q);
  g1_get_ord(q);

  bn_t a2;
  bn_null(a2); bn_new(a2);
  hash_m_info_xstar(a2, m, mlen, info, ilen, S->x_star);

  if (bn_cmp(a2, S->a_star) != RLC_EQ) {
    bn_free(a2);
    bn_free(q);
    return 0;
  }

  g1_t lhs, rhs, t1, t2;
  g1_null(lhs); g1_null(rhs); g1_null(t1); g1_null(t2);
  g1_new(lhs);  g1_new(rhs);  g1_new(t1);  g1_new(t2);

  // lhs = s*x*
  g1_mul(lhs, S->x_star, K->s);
  g1_norm(lhs, lhs);

  // rhs = sig1 + (sig2*s)g2 + (a* s)v
  bn_t sig2s, as;
  bn_null(sig2s); bn_null(as);
  bn_new(sig2s);  bn_new(as);

  bn_mul(sig2s, S->sig2,   K->s); bn_mod(sig2s, sig2s, q);
  bn_mul(as,    S->a_star, K->s); bn_mod(as, as, q);

  g1_mul(t1, K->g2, sig2s);
  g1_mul(t2, K->v,  as);

  g1_copy(rhs, S->sig1);
  g1_add(rhs, rhs, t1);
  g1_add(rhs, rhs, t2);
  g1_norm(rhs, rhs);

  int ok = (g1_cmp(lhs, rhs) == RLC_EQ);

  bn_free(sig2s); bn_free(as);
  g1_free(lhs); g1_free(rhs); g1_free(t1); g1_free(t2);
  bn_free(a2);
  bn_free(q);

  return ok;
}

static size_t signature_size_bytes(const pb_sig_t *S) {
  bn_t q;
  bn_null(q); bn_new(q);
  g1_get_ord(q);

  int scalar_bytes = bn_size_bin(q);
  int x_bytes = g1_size_bin(S->x_star, 1);
  int sig1_bytes = g1_size_bin(S->sig1, 1);
  size_t sig_bytes = (size_t)x_bytes + (size_t)scalar_bytes +
                     (size_t)sig1_bytes + (size_t)scalar_bytes;

  bn_free(q);
  return sig_bytes;
}

static void print_sizes_and_params(const pb_keys_t *K, const pb_sig_t *S) {
  bn_t q;
  bn_null(q); bn_new(q);
  g1_get_ord(q);

  int scalar_bytes = bn_size_bin(q);
  int scalar_bits = scalar_bytes * 8;

  int v_bytes = g1_size_bin(K->v, 1);
  int k_bytes = g1_size_bin(K->k, 1);
  int x_bytes = g1_size_bin(S->x_star, 1);
  int sig1_bytes = g1_size_bin(S->sig1, 1);

  size_t sig_bytes = signature_size_bytes(S);

  printf("  Key / Signature sizes (compressed):\n");
  printf("    scalar size (Z_q):       %d bytes (%d bits)\n", scalar_bytes, scalar_bits);
  printf("    sk_S=(s1,s2):            %d bytes (%d bits)\n", 2 * scalar_bytes, 2 * scalar_bits);
  printf("    sk_V=s:                  %d bytes (%d bits)\n", scalar_bytes, scalar_bits);
  printf("    pk_S=v (G1):             %d bytes (%d bits)\n", v_bytes, v_bytes * 8);
  printf("    pk_V=k (G1):             %d bytes (%d bits)\n", k_bytes, k_bytes * 8);
  printf("    PB-SDVSS signature:      %zu bytes (%zu bits)\n", sig_bytes, sig_bytes * 8);
  printf("      breakdown: x*=%d, a*=%d, sig1=%d, sig2=%d\n",
         x_bytes, scalar_bytes, sig1_bytes, scalar_bytes);

  printf("\n--- RELIC pairing parameters ---\n");
  pc_param_print();
  printf("--------------------------------\n\n");

  bn_free(q);
}

// -----------------------------------------------------------------------------
// Benchmark routines
// -----------------------------------------------------------------------------

static void warmup_sign(const bench_config_t *cfg, const pb_keys_t *K, pb_sig_t *S) {
  uint8_t m[MSG_LEN];
  uint8_t info[INFO_LEN];

  for (int i = 0; i < cfg->warmup; i++) {
    random_bytes(m, MSG_LEN);
    random_bytes(info, INFO_LEN);
    CHECK_OK(pb_sign(m, MSG_LEN, info, INFO_LEN, K, S), "warm-up sign failed");
  }
}

static void warmup_verify(const bench_config_t *cfg, const pb_keys_t *K, pb_sig_t *S) {
  uint8_t m[MSG_LEN];
  uint8_t info[INFO_LEN];

  for (int i = 0; i < cfg->warmup; i++) {
    random_bytes(m, MSG_LEN);
    random_bytes(info, INFO_LEN);
    CHECK_OK(pb_sign(m, MSG_LEN, info, INFO_LEN, K, S), "warm-up sign for verify failed");
    CHECK_OK(pb_verify(m, MSG_LEN, info, INFO_LEN, K, S), "warm-up verify failed");
  }
}

static stats_t benchmark_sign(const bench_config_t *cfg, const pb_keys_t *K, pb_sig_t *S) {
  double *run_us = (double *)xmalloc((size_t)cfg->repetitions * sizeof(double));

  for (int r = 0; r < cfg->repetitions; r++) {
    uint8_t *msgs = (uint8_t *)xmalloc((size_t)cfg->iterations * MSG_LEN);
    uint8_t *infos = (uint8_t *)xmalloc((size_t)cfg->iterations * INFO_LEN);

    // Not timed: dataset construction.
    make_dataset(msgs, infos, cfg->iterations);

    double t0 = now_seconds();
    for (int i = 0; i < cfg->iterations; i++) {
      CHECK_OK(pb_sign(msgs + (size_t)i * MSG_LEN,
                       MSG_LEN,
                       infos + (size_t)i * INFO_LEN,
                       INFO_LEN,
                       K,
                       S),
               "PB-SDVSS.Sign failed");
    }
    double t1 = now_seconds();

    run_us[r] = ((t1 - t0) / (double)cfg->iterations) * 1e6;

    free(msgs);
    free(infos);
  }

  stats_t st = compute_stats(run_us, cfg->repetitions);
  free(run_us);
  return st;
}

static stats_t benchmark_verify(const bench_config_t *cfg, const pb_keys_t *K) {
  double *run_us = (double *)xmalloc((size_t)cfg->repetitions * sizeof(double));

  for (int r = 0; r < cfg->repetitions; r++) {
    uint8_t *msgs = (uint8_t *)xmalloc((size_t)cfg->iterations * MSG_LEN);
    uint8_t *infos = (uint8_t *)xmalloc((size_t)cfg->iterations * INFO_LEN);
    pb_sig_t *sigs = (pb_sig_t *)xmalloc((size_t)cfg->iterations * sizeof(pb_sig_t));

    // Not timed: dataset construction and valid signature generation.
    make_dataset(msgs, infos, cfg->iterations);
    for (int i = 0; i < cfg->iterations; i++) {
      sig_init(&sigs[i]);
      CHECK_OK(pb_sign(msgs + (size_t)i * MSG_LEN,
                       MSG_LEN,
                       infos + (size_t)i * INFO_LEN,
                       INFO_LEN,
                       K,
                       &sigs[i]),
               "PB-SDVSS.Sign for verify dataset failed");
    }

    double t0 = now_seconds();
    for (int i = 0; i < cfg->iterations; i++) {
      CHECK_OK(pb_verify(msgs + (size_t)i * MSG_LEN,
                         MSG_LEN,
                         infos + (size_t)i * INFO_LEN,
                         INFO_LEN,
                         K,
                         &sigs[i]),
               "PB-SDVSS.Verify failed");
    }
    double t1 = now_seconds();

    run_us[r] = ((t1 - t0) / (double)cfg->iterations) * 1e6;

    for (int i = 0; i < cfg->iterations; i++) {
      sig_free(&sigs[i]);
    }
    free(sigs);
    free(msgs);
    free(infos);
  }

  stats_t st = compute_stats(run_us, cfg->repetitions);
  free(run_us);
  return st;
}

static void print_algorithm_stats(const char *label, const stats_t *st, int repetitions) {
  printf("  Algorithm %s, single-threaded:\n", label);
  printf("    mean:       %.3f us/op\n", st->mean_us);
  printf("    median:     %.3f us/op\n", st->median_us);
  printf("    std. dev.:  %.3f us/op over %d repeated runs\n", st->stddev_us, repetitions);
  printf("    throughput: %.2f ops/sec, based on mean\n\n", st->throughput_ops_s);
}

static void csv_write_header(FILE *csv) {
  fprintf(csv,
          "implementation,parameter_set,operation,iterations,warmup,repetitions,"
          "mean_us,median_us,stddev_us,throughput_ops_s,msg_len,info_len,signature_bytes\n");
}

static void csv_write_row(FILE *csv,
                          const char *operation,
                          const bench_config_t *cfg,
                          const stats_t *st,
                          size_t sig_bytes) {
  fprintf(csv,
          "PB-SDVSS-RELIC,pc_param_set_any_G1_only,%s,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%d,%d,%zu\n",
          operation,
          cfg->iterations,
          cfg->warmup,
          cfg->repetitions,
          st->mean_us,
          st->median_us,
          st->stddev_us,
          st->throughput_ops_s,
          MSG_LEN,
          INFO_LEN,
          sig_bytes);
}

int main(int argc, char **argv) {
  bench_config_t cfg = parse_args(argc, argv);
  maybe_pin_core(&cfg);

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
  printf("=== PB-SDVSS over RELIC G1, active pairing parameters, no pairings ===\n");

  FILE *csv = NULL;
  if (cfg.csv_path != NULL) {
    csv = fopen(cfg.csv_path, "w");
    if (csv == NULL) {
      fprintf(stderr, "Could not open CSV file '%s': %s\n", cfg.csv_path, strerror(errno));
      core_clean();
      return EXIT_FAILURE;
    }
    csv_write_header(csv);
  }

  pb_keys_t K;
  pb_setup(&K);

  pb_sig_t S;
  sig_init(&S);

  uint8_t m[MSG_LEN];
  uint8_t info[INFO_LEN];
  random_bytes(m, MSG_LEN);
  random_bytes(info, INFO_LEN);

  CHECK_OK(pb_sign(m, MSG_LEN, info, INFO_LEN, &K, &S), "initial sign failed");
  CHECK_OK(pb_verify(m, MSG_LEN, info, INFO_LEN, &K, &S), "initial verify failed");

  print_sizes_and_params(&K, &S);
  size_t sig_bytes = signature_size_bytes(&S);

  warmup_sign(&cfg, &K, &S);
  stats_t sign_stats = benchmark_sign(&cfg, &K, &S);

  warmup_verify(&cfg, &K, &S);
  stats_t verify_stats = benchmark_verify(&cfg, &K);

  print_algorithm_stats("PB-SDVSS.Sign", &sign_stats, cfg.repetitions);
  print_algorithm_stats("PB-SDVSS.Verify", &verify_stats, cfg.repetitions);

  if (csv != NULL) {
    csv_write_row(csv, "Sign", &cfg, &sign_stats, sig_bytes);
    csv_write_row(csv, "Verify", &cfg, &verify_stats, sig_bytes);
    fclose(csv);
    printf("CSV results written to: %s\n", cfg.csv_path);
  }

  sig_free(&S);
  keys_free(&K);
  core_clean();

  return EXIT_SUCCESS;
}
