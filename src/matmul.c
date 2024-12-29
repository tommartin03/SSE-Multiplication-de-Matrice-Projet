#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <xmmintrin.h>

void print_matrix(float *mat, size_t dim, const char *name) {
    printf("\nMatrice %s:\n", name);
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            printf("%6.2f ", mat[i * dim + j]);
        }
        printf("\n");
    }
}

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

void mul_parallel_sse_blocked(float *A, float *B, float *res, size_t dim, size_t block_size) {
    for (size_t i = 0; i < dim * dim; i++) {
        res[i] = 0;
    }

    for (size_t bi = 0; bi < dim; bi += block_size) {
        for (size_t bj = 0; bj < dim; bj += block_size) {
            for (size_t bk = 0; bk < dim; bk += block_size) {

                for (size_t i = bi; i < bi + block_size; i++) {
                    for (size_t j = bj; j < bj + block_size; j += 4) {
                        __m128 sum = _mm_load_ps(&res[i * dim + j]);

                        for (size_t k = bk; k < bk + block_size; k++) {
                            __m128 a_vals = _mm_set1_ps(A[i * dim + k]);
                            __m128 b_vals = _mm_load_ps(&B[k * dim + j]);
                            sum = _mm_add_ps(sum, _mm_mul_ps(a_vals, b_vals));
                        }

                        _mm_store_ps(&res[i * dim + j], sum);
                    }
                }
            }
        }
    }
}

void check(float *A, float *B, float *res, size_t dim) {
    float *exp = (float *)malloc(dim * dim * sizeof(float));
    naive_mul(A, B, exp, dim);
    float epsilon = 1e-3 * dim;
    for (size_t i = 0; i < dim * dim; i++) {
        if (fabs(exp[i] - res[i]) > epsilon) {
            printf("Erreur a l'indice %lu : attendu %f, obtenu %f\n", i, exp[i], res[i]);
            free(exp);
            exit(1);
        }
    }
    free(exp);
}

int main() {
    size_t dim = 2048;
    float *A, *B, *C_naive, *C_moins_naive, *C_parallel_sse;

    if (posix_memalign((void**)&A, 16, dim * dim * sizeof(float)) != 0 ||
        posix_memalign((void**)&B, 16, dim * dim * sizeof(float)) != 0 ||
        posix_memalign((void**)&C_naive, 16, dim * dim * sizeof(float)) != 0 ||
        posix_memalign((void**)&C_moins_naive, 16, dim * dim * sizeof(float)) != 0 ||
        posix_memalign((void**)&C_parallel_sse, 16, dim * dim * sizeof(float)) != 0) {
        perror("Erreur d'allocation memoire");
        free(A); free(B); free(C_naive); free(C_moins_naive); free(C_parallel_sse);
        return -1;
    }

    srand(time(NULL));
    for (size_t i = 0; i < dim * dim; i++) {
        A[i] = rand() % 5;
        B[i] = rand() % 5;
    }

    FILE *file = fopen("benchmark_results.txt", "w");
    if (!file) {
        perror("Erreur d'ouverture de fichier");
        free(A); free(B); free(C_naive); free(C_moins_naive); free(C_parallel_sse);
        return -1;
    }

    auto start = clock();
    naive_mul(A, B, C_naive, dim);
    auto end = clock();
    double time_naive = (double)(end - start) / CLOCKS_PER_SEC;
    fprintf(file, "Naive: %f s\n", time_naive);

    start = clock();
    moins_naive_mul(A, B, C_moins_naive, dim);
    end = clock();
    double time_moins_naive = (double)(end - start) / CLOCKS_PER_SEC;
    fprintf(file, "Moins naive: %f s\n", time_moins_naive);

    size_t block_sizes[] = {4, 8, 16};
    for (size_t b = 0; b < 3; b++) {
        size_t block_size = block_sizes[b];

        clock_t start = clock();
        mul_parallel_sse_blocked(A, B, C_parallel_sse, dim, block_size);
        clock_t end = clock();
        double time_blocked = (double)(end - start) / CLOCKS_PER_SEC;

        fprintf(file, "SSE avec blocs (%lu): %f s\n", block_size, time_blocked);
    }

    fclose(file);

    check(A, B, C_parallel_sse, dim);
    free(A); free(B); free(C_naive); free(C_moins_naive); free(C_parallel_sse);
    return 0;
}
