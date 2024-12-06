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

void moins_naive_mul(float *A, float *B, float *res, size_t dim) {
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            res[i * dim + j] = 0;
        }
        for (size_t k = 0; k < dim; k++) {
            float premier = A[i * dim + k];
            for (size_t j = 0; j < dim; j++) {
                res[i * dim + j] += premier * B[k * dim + j];
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

  clock_t start = clock();
  naive_mul(A, B, C, dim);
  clock_t end = clock();
  double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
  printf("Temps d'execution de naive_mul: %f\n", time_spent);

  start = clock();
  moins_naive_mul(A, B, C, dim);
  end = clock();
  double time_spent2 = (double)(end - start) / CLOCKS_PER_SEC;
  printf("Temps d'execution de moins_naive_mul: %f\n", time_spent2);

// you can activate check by adding -DCHECK_MUL to your command line
#ifdef CHECK_MUL
  check(A, B, C, dim);
#endif

  free(A);
  free(B);
  free(C);

  return 0;
}
