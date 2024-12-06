#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void naive_mul(float *A, float *B, float *res, size_t dim) {
  for (size_t i = 0; i < dim; i++) {
    for (size_t j = 0; j < dim; j++) {
      res[i * dim + j] = 0;
      for (size_t k = 0; k < dim; k++) {
        res[i * dim + j] += A[i * dim + k] * B[k * dim + j];
      }
    }
  }
}

void check(float *A, float *B, float *res, size_t dim) {
  float *exp = malloc(dim * dim * sizeof(float));

  naive_mul(A, B, exp, dim);

  for (size_t i = 0; i < dim * dim; i++) {
    if (exp[i] != res[i]) {
      printf("Value at %lu differs: %f\n", i, exp[i] - res[i]);
    }
  }
}

int main() {
  //gcc -o mat matmul.c -DCHECK_MUL -O2

  size_t dim = 2048;

  float *A = malloc(dim * dim * sizeof(float));
  float *B = malloc(dim * dim * sizeof(float));
  float *C = malloc(dim * dim * sizeof(float));

  for (size_t i = 0; i < dim * dim; i++) {
    A[i] = rand() % 5;
    B[i] = rand() % 5;
  }

  // your code or function here
  naive_mul(A, B, C, dim);

// you can activate check by adding -DCHECK_MUL to your command line
#ifdef CHECK_MUL
  check(A, B, C, dim);
#endif

  free(A);
  free(B);
  free(C);

  return 0;
}
