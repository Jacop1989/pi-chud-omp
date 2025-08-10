// pi_chud_omp_plus.c
// Fast π using Chudnovsky series + binary splitting, parallel chunking (OpenMP) and GMP.
// Adds: 1) Progress log per chunk, 2) "--nosave" option to skip writing pi.txt.
//
// --------------------------------------------------------
// Build on Intel macOS (Homebrew clang + GMP + libomp):
//   /usr/local/opt/llvm/bin/clang -std=c11 -O3 -march=native pi_chud_omp_plus.c -o pi_chud_omp_plus \
//     -I"$(brew --prefix gmp)/include"   -L"$(brew --prefix gmp)/lib" \
//     -I"$(brew --prefix libomp)/include" -L"$(brew --prefix libomp)/lib" \
//     -fopenmp -lomp -lgmp \
//     -Wl,-rpath,"$(brew --prefix libomp)/lib" -Wl,-rpath,"$(brew --prefix gmp)/lib"
//
// --------------------------------------------------------
// Run normally (save pi.txt):
//   OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE OMP_PROC_BIND=TRUE ./pi_chud_omp_plus 1000000
//
// Run without saving file (--nosave):
//   OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE OMP_PROC_BIND=TRUE ./pi_chud_omp_plus 1000000 --nosave
// --------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gmp.h>

#ifdef _OPENMP
  #include <omp.h>
#endif

#ifdef __APPLE__
  #include <mach/mach_time.h>
  static double now(void) {
      static mach_timebase_info_data_t info = {0,0};
      if (info.denom == 0) mach_timebase_info(&info);
      uint64_t t = mach_absolute_time();
      double ns = (double)t * (double)info.numer / (double)info.denom;
      return ns * 1e-9;
  }
#else
  #include <time.h>
  static double now(void) {
      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
  }
#endif

typedef struct { mpz_t P, Q, T; } PQT;
static mpz_t C3;

static void pqt_init(PQT* s){ mpz_init(s->P); mpz_init(s->Q); mpz_init(s->T); }
static void pqt_clear(PQT* s){ mpz_clear(s->P); mpz_clear(s->Q); mpz_clear(s->T); }

static void pqt_combine(PQT* out, const PQT* L, const PQT* R){
    mpz_mul(out->P, L->P, R->P);
    mpz_mul(out->Q, L->Q, R->Q);
    mpz_t t1,t2; mpz_inits(t1,t2,NULL);
    mpz_mul(t1, L->T, R->Q);
    mpz_mul(t2, L->P, R->T);
    mpz_add(out->T, t1, t2);
    mpz_clears(t1,t2,NULL);
}

static void bs(long a, long b, PQT* out){
    if (b - a == 1) {
        if (a == 0) {
            mpz_set_ui(out->P, 1);
            mpz_set_ui(out->Q, 1);
            mpz_set_ui(out->T, 13591409u);
            return;
        }
        mpz_t t1; mpz_init(t1);
        mpz_set_si(out->P, (long)(6*a - 5));
        mpz_set_si(t1, (long)(2*a - 1)); mpz_mul(out->P, out->P, t1);
        mpz_set_si(t1, (long)(6*a - 1)); mpz_mul(out->P, out->P, t1);
        mpz_set_si(out->Q, a);
        mpz_mul(out->Q, out->Q, out->Q);
        mpz_mul(out->Q, out->Q, out->Q);
        mpz_mul(out->Q, out->Q, C3);
        mpz_set_si(t1, 545140134l);
        mpz_mul_si(t1, t1, a);
        mpz_add_ui(t1, t1, 13591409u);
        mpz_mul(out->T, out->P, t1);
        if (a & 1) mpz_neg(out->T, out->T);
        mpz_clear(t1);
        return;
    }
    long m = (a + b) / 2;
    PQT L, R; pqt_init(&L); pqt_init(&R);
    bs(a, m, &L);
    bs(m, b, &R);
    pqt_combine(out, &L, &R);
    pqt_clear(&L); pqt_clear(&R);
}

static int write_pi_to_file(mpf_t pi, unsigned long digits, const char* filename){
    long exp10 = 0;
    char* s = mpf_get_str(NULL, &exp10, 10, digits + 1, pi);
    if (!s) return -1;
    FILE* f = fopen(filename, "w");
    if (!f) { free(s); return -2; }
    size_t len = 0; while (s[len]) len++;
    if (len < digits + 1) {
        char* p = (char*)malloc(digits + 2);
        for (size_t i = 0; i < digits + 1; ++i) p[i] = (i < len) ? s[i] : '0';
        p[digits + 1] = '\0';
        free(s); s = p; len = digits + 1;
    }
    int neg = (s[0] == '-');
    size_t start = neg ? 1 : 0;
    if (neg) fputc('-', f);
    if (exp10 <= 0) {
        fputs("0.", f);
        for (long i = 0; i < -exp10; ++i) fputc('0', f);
        for (unsigned long i = 0; i < digits; ++i) {
            char c = s[start + i]; fputc(c ? c : '0', f);
        }
    } else {
        size_t pos = start;
        for (long i = 0; i < exp10; ++i) {
            char c = s[pos]; fputc(c ? c : '0', f); if (c) pos++;
        }
        fputc('.', f);
        for (unsigned long i = 0; i < digits; ++i) {
            char c = s[pos]; fputc(c ? c : '0', f); if (c) pos++;
        }
    }
    fputc('\n', f); fclose(f); free(s); return 0;
}

int main(int argc, char** argv){
    unsigned long digits = 100000ul;
    int save_file = 1;
    if (argc > 1) digits = strtoul(argv[1], NULL, 10);
    if (argc > 2 && strcmp(argv[2], "--nosave") == 0) save_file = 0;

#ifdef _OPENMP
    int T = omp_get_max_threads();
    printf("[OpenMP] max threads = %d\n", T);
#else
    int T = 1;
    printf("[OpenMP] OFF\n");
#endif

    const double DIGITS_PER_TERM = 14.1816474627;
    long n_terms = (long)(digits / DIGITS_PER_TERM) + 1;
    if (n_terms < 1) n_terms = 1;

    mpz_init(C3);
    mpz_ui_pow_ui(C3, 640320u, 3u);
    mpz_fdiv_q_ui(C3, C3, 24u);

    mp_bitcnt_t prec_bits = (mp_bitcnt_t)(digits * 3.3219280948873626) + 128;
    mpf_set_default_prec(prec_bits);

    double t0 = now();

    PQT* part = (PQT*)calloc(T, sizeof(PQT));
    for (int i = 0; i < T; ++i) pqt_init(&part[i]);
    long chunk = (n_terms + T - 1) / T;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < T; ++i) {
        long a = i * chunk;
        long b = a + chunk; if (b > n_terms) b = n_terms;
        if (a < b) {
            bs(a, b, &part[i]);
            printf("[Chunk %d/%d] done\n", i + 1, T);
        }
    }

    PQT sum; pqt_init(&sum);
    mpz_set_ui(sum.P, 1); mpz_set_ui(sum.Q, 1); mpz_set_ui(sum.T, 0);
    for (int i = 0; i < T; ++i) {
        PQT comb; pqt_init(&comb);
        pqt_combine(&comb, &sum, &part[i]);
        pqt_clear(&sum);
        pqt_clear(&part[i]);
        sum = comb;
    }
    free(part);

    mpf_t pi, fQ, fT, C; mpf_inits(pi, fQ, fT, C, NULL);
    mpf_set_ui(C, 10005u); mpf_sqrt(C, C); mpf_mul_ui(C, C, 426880u);
    mpf_set_z(fQ, sum.Q); mpf_set_z(fT, sum.T);
    mpf_mul(pi, fQ, C);
    mpf_div(pi, pi, fT);

    double t1 = now();

    gmp_printf("Preview (first 100 digits):\n%.102Ff\n", pi);
    printf("Digits=%lu | Terms=%ld | Threads=%d | Time=%.3f s\n",
           digits, n_terms, T, (t1 - t0));

    if (save_file) {
        if (write_pi_to_file(pi, digits, "pi.txt") == 0)
            printf("Saved to pi.txt\n");
        else
            printf("Failed to write pi.txt\n");
    } else {
        printf("Skipped saving to file (--nosave)\n");
    }

    mpf_clears(pi, fQ, fT, C, NULL);
    pqt_clear(&sum);
    mpz_clear(C3);
    return 0;
}
