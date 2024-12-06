#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void naive_mul(float *A, float *B, float *res) {}

void check(float *A, float *B, float *res, size_t dim) {
  float *exp = malloc(dim * dim * sizeof(float));

  naive_mul(A, B, exp);

  for (size_t i = 0; i < dim * dim; i++) {
    if (exp[i] != res[i]) {
      printf("Value at %lu differs: %f\n", i, exp[i] - res[i]);
    }
  }
}

int main() {
  size_t dim = 2048;

  float *A = malloc(dim * dim * sizeof(float));
  float *B = malloc(dim * dim * sizeof(float));
  float *C = malloc(dim * dim * sizeof(float));

  for (size_t i = 0; i < dim * dim; i++) {
    A[i] = rand() % 5;
    B[i] = rand() % 5;
  }

  // your code or function here

// you can activate check by adding -DCHECK_MUL to your command line
#ifdef CHECK_MUL
  check(A, B, C);
#endif

  free(A);
  free(B);
  free(C);

  return 0;
}
