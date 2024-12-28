#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "matrix_multiply.h"  // Inclure l'en-tête des fonctions

void check(float *A, float *B, float *res, size_t dim) {
    float *exp = malloc(dim * dim * sizeof(float));

    naive_mul(A, B, exp, dim);

    for (size_t i = 0; i < dim * dim; i++) {
        if (fabs(exp[i] - res[i]) > 1e-4) {
            printf("Erreur à l'indice %lu : attendu %f, obtenu %f\n",
                   i, exp[i], res[i]);
            free(exp);
            exit(1); 
        }
    }
    free(exp);
}

int main() {
    size_t dim = 10; // Taille de la matrice

    float *A, *B, *C_naive, *C_moins_naive, *C_parallel;
    posix_memalign((void**)&A, 16, dim * dim * sizeof(float));
    posix_memalign((void**)&B, 16, dim * dim * sizeof(float));
    posix_memalign((void**)&C_naive, 16, dim * dim * sizeof(float));
    posix_memalign((void**)&C_moins_naive, 16, dim * dim * sizeof(float));
    posix_memalign((void**)&C_parallel, 16, dim * dim * sizeof(float));

    srand(time(NULL));
    for (size_t i = 0; i < dim * dim; i++) {
        A[i] = rand() % 5;
        B[i] = rand() % 5;
        C_naive[i] = 0;
        C_moins_naive[i] = 0;
        C_parallel[i] = 0;
    }

    // Ouvrir le fichier pour enregistrer les résultats
    FILE *file = fopen("benchmark_results.txt", "w");
    if (!file) {
        perror("Erreur d'ouverture de fichier");
        return -1;
    }

    // Effectuer le benchmark de naive_mul
    clock_t start_naive = clock();
    naive_mul(A, B, C_naive, dim);
    clock_t end_naive = clock();
    double time_naive = (double)(end_naive - start_naive) / CLOCKS_PER_SEC;
    fprintf(file, "%zu\t%f\tNaive\n", dim, time_naive);

    // Effectuer le benchmark de moins_naive_mul
    clock_t start_moins_naive = clock();
    moins_naive_mul(A, B, C_moins_naive, dim);
    clock_t end_moins_naive = clock();
    double time_moins_naive = (double)(end_moins_naive - start_moins_naive) / CLOCKS_PER_SEC;
    fprintf(file, "%zu\t%f\tMoins Naive\n", dim, time_moins_naive);

    // Effectuer le benchmark de mul_parallel
    clock_t start_parallel = clock();
    mul_parallel(A, B, C_parallel, dim);
    clock_t end_parallel = clock();
    double time_parallel = (double)(end_parallel - start_parallel) / CLOCKS_PER_SEC;
    fprintf(file, "%zu\t%f\tParallel\n", dim, time_parallel);

    // Vérification des résultats
    check(A, B, C_parallel, dim);

    // Fermer le fichier
    fclose(file);

    // Libération de la mémoire
    free(A);
    free(B);
    free(C_naive);
    free(C_moins_naive);
    free(C_parallel);
    return 0;
}
