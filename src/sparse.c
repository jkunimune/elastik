/**
 * sparse.c
 *
 * the cytonic component of sparse.py
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// I think that this is necessary to make it portable?  idk; if Python was bilt on any half-
// decent programming language then these simple calculations would be portable by default.
#ifdef _WIN32
#   define EXPORT __declspec(dllexport)
#else
#   define EXPORT
#endif


/**
 * a single ndarray of numbers whose nonzero elements are all specified by index. the
 * SparseArray's shape is not specified, so there is no enforced bound on indices.
 */
struct c_sparse {
    int num_rows;
    int num_cols;
    double* data;
    int* indices;
    int* indptr;
};


/**
 * allocate and return the data and indices of the nonzero values that will result from adding the given
 * sparse arrays, then deallocate a_data and a_indices.  also, b is scaled by a scalar.
 * @param c_nnz if this is zero, it means that we don’t yet know how many nz elements there will be, and
 *              therefore shouldn’t allocate anything yet.  otherwise, it’s the number to allocate.
 * @param flop if this is zero, it will ignore all of the FLOPs and not touch the data.
 * @return a struct with data and indices if we allocated them, and with the new array’s nnz encoded in num_rows
 */
struct c_sparse weited_add(double* a_data, int* a_indices, int a_nnz, double weit,
                double* b_data, int* b_indices, int b_nnz, int c_nnz, int flop) {
    int k_a = 0;
    int k_b = 0;
    int k_c = 0;
    int* c_indices = NULL;
    double* c_data = NULL;
    if (c_nnz > 0) {
        c_indices = malloc(c_nnz*sizeof(int));
        if (flop)
            c_data = calloc(c_nnz, sizeof(double));
    }
    while (k_a < a_nnz || k_b < b_nnz) {
        int next_index;
        if (k_a == a_nnz)
            next_index = b_indices[k_b];
        else if (k_b == b_nnz)
            next_index = a_indices[k_a];
        else if (a_indices[k_a] <= b_indices[k_b])
            next_index = a_indices[k_a];
        else
            next_index = b_indices[k_b];
        if (c_nnz > 0)
            c_indices[k_c] = next_index;
        if (k_a < a_nnz && next_index == a_indices[k_a]) {
            if (c_nnz > 0 && flop)
                c_data[k_c] = a_data[k_a];
            k_a ++;
        }
        if (k_b < b_nnz && next_index == b_indices[k_b]) {
            if (c_nnz > 0 && flop)
                c_data[k_c] += weit*b_data[k_b];
            k_b ++;
        }
        k_c ++;
    }
    if (c_nnz > 0) {
        if (a_data != NULL)
            free(a_data);
        if (a_indices != NULL)
            free(a_indices);
    }
    struct c_sparse output = {.num_rows=k_c, .data=c_data, .indices=c_indices};
    return output;
}


/**
 * apply a matrix multiplication between two sparse arrays, pretending you can reshape the twoth one
 * so its height equals the first one’s width.  this function will set the data and indices, but
 * you’ll haff to use the sister function reshape_matmul_indptr to determine how much space to
 * allocate for them.
 */
EXPORT void reshape_matmul(
        struct c_sparse a, struct c_sparse b, double* data, int* indices) {
    // first, compute this "dimension" which is the number of rows of b that correspond to each element of a
    int row_size = b.num_rows/a.num_cols;
    // then iterate thru the rows of the output matrix
    int k_c = 0;
    for (int i_a = 0; i_a < a.num_rows; i_a ++) {
        // iterate thru the "collums" of the output matrix
        for (int j = 0; j < row_size; j ++) {
            // build up the sparse "element" from 0 by going thru the coefficients
            double* element_data = NULL;
            int* element_indices = NULL;
            int element_nnz = 0;
            for (int k_a = a.indptr[i_a]; k_a < a.indptr[i_a + 1]; k_a ++) {
                int i_b = a.indices[k_a]*row_size + j;
                int k_b_start = b.indptr[i_b];
                int new_nnz = b.indptr[i_b + 1] - k_b_start;
                struct c_sparse result = weited_add(
                    NULL, element_indices, element_nnz, a.data[k_a],
                    NULL, b.indices + k_b_start, new_nnz, 0, 0);
                result = weited_add(
                    element_data, element_indices, element_nnz, a.data[k_a],
                    b.data + k_b_start, b.indices + k_b_start, new_nnz, result.num_rows, 1);
                element_data = result.data;
                element_indices = result.indices;
                element_nnz = result.num_rows;
            }
            // then assine the final result
            for (int k = 0; k < element_nnz; k ++) {
                data[k_c + k] = element_data[k];
                indices[k_c + k] = element_indices[k];
            }
            k_c += element_nnz;
            // and don’t forget to deallocate
            if (element_nnz > 0) {
                free(element_data);
                free(element_indices);
            }
        }
    }
}


/**
 * apply a matrix multiplication between two sparse arrays, pretending you can reshape the twoth one
 * so its height equals the first one’s width.  this function will just set the indptr vector, which
 * you should then use to allocate the correct amount for data and indices.
 */
EXPORT void reshape_matmul_indptr(
        struct c_sparse a, struct c_sparse b, int* row_nnzs) {
    // first, compute this "dimension" which is the number of rows of b that correspond to each element of a
    int row_size = b.num_rows/a.num_cols;
    // then iterate thru the rows of the output matrix
    for (int i_a = 0; i_a < a.num_rows; i_a ++) {
        // iterate thru the "collums" of the output matrix
        for (int j = 0; j < row_size; j ++) {
            // build up the sparse "element" from 0 by going thru the coefficients
            int* element_indices = NULL;
            int element_nnz = 0;
            for (int k_a = a.indptr[i_a]; k_a < a.indptr[i_a + 1]; k_a ++) {
                int i_b = a.indices[k_a]*row_size + j;
                int k_b_start = b.indptr[i_b];
                int new_nnz = b.indptr[i_b + 1] - k_b_start;
                struct c_sparse result = weited_add(
                    NULL, element_indices, element_nnz, 0,
                    NULL, b.indices + k_b_start, new_nnz, 0, 0);
                result = weited_add(
                    NULL, element_indices, element_nnz, 0,
                    NULL, b.indices + k_b_start, new_nnz, result.num_rows, 0);
                element_indices = result.indices;
                element_nnz = result.num_rows;
            }
            // then assine the final result
            row_nnzs[i_a*row_size + j] = element_nnz;
            // and don’t forget to deallocate
            if (element_nnz > 0)
                free(element_indices);
        }
    }
}


/**
 * perform matrix multiplication between two SparseArrayArrays where the first one is transposed.
 */
EXPORT void elementwise_outer_product(
        struct c_sparse a, struct c_sparse b, double* data, int* indices) {
    int indptr = 0;
    for (int i = 0; i < a.num_rows; i ++) {
        int nnz_a = a.indptr[i + 1] - a.indptr[i];
        int nnz_b = b.indptr[i + 1] - b.indptr[i];
        for (int k_a = a.indptr[i]; k_a < a.indptr[i + 1]; k_a ++) {
            for (int k_b = b.indptr[i]; k_b < b.indptr[i + 1]; k_b ++) {
                int k_c = indptr + (k_a - a.indptr[i])*nnz_b + (k_b - b.indptr[i]);
                data[k_c] = a.data[k_a]*b.data[k_b];
                indices[k_c] = a.indices[k_a]*b.num_cols + b.indices[k_b];
            }
        }
        indptr += nnz_a*nnz_b;
    }
}


EXPORT void repeat_diagonally(
        struct c_sparse a, int times, double* data, int* indices) {
    for (int i = 0; i < a.num_rows; i ++) {
        for (int j = 0; j < times; j ++) {
            int indptr = a.indptr[i];
            int nnz = a.indptr[i + 1] - a.indptr[i];
            for (int l = 0; l < nnz; l ++) {
                data[indptr*times + j*nnz + l] = a.data[indptr + l];
                indices[indptr*times + j*nnz + l] = a.indices[indptr + l]*times + j;
            }
        }
    }
}
