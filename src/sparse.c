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
 * a binary operator that works on array-likes
 */
enum Operator {
    ADD, SUBTRACT, MULTIPLY, DIVIDE, POWER
};

/**
 * a single ndarray of numbers whose nonzero elements are all specified by index. the
 * SparseArray's shape is not specified, so there is no enforced bound on indices.
 */
struct SparseArray {
    int ndim;
    int nitems;
    int* indices; // TODO: I can't help feel like it would be cleaner if these were 1d indices
    double* values;
};

/**
 * a dense ndarray of sparse arrays
 */
struct SparseArrayArray {
    int ndim;
    int size;
    int* shape;
    struct SparseArray* elements;
    int element_ndim;
};

/**
 * multiply a sequence together
 */
int product(const int values[], int length) {
    int product = 1;
    for (int k = 0; k < length; k ++)
        product *= values[k];
    return product;
}

/**
 * compare two arrays, taking the initial elements as most significant.
 * @return 1 if a > b, -1 if a < b, and 0 if a == b
 */
int compare(const int a[], const int b[], int length) {
    for (int k = 0; k < length; k ++) {
        if (a[k] > b[k])
            return 1;
        else if (a[k] < b[k])
            return -1;
        else
            continue;
    }
    return 0;
}

/**
 * compare two arrays for equivalence of all elements
 */
int array_equal(const int a[], const int b[], int length) {
    for (int k = 0; k < length; k ++)
        if (a[k] != b[k])
            return 0;
    return 1;
}

/**
 * free the memory allocated to a SparseArray
 * @param a
 */
void free_sa(struct SparseArray a) {
    if (a.nitems > 0) {
        free(a.indices);
        free(a.values);
    }
}

/**
 * free the memory allocated to a SparseArrayArray and all of its elements
 */
EXPORT void free_saa(
        struct SparseArrayArray a) {
    for (int i = 0; i < a.size; i ++)
        free_sa(a.elements[i]);
    if (a.size > 0)
        free(a.elements);
    free(a.shape);
}

/**
 * this is just the free function, but I want to have it in my own library
 */
EXPORT void free_char_p(char* data) {
	  free(data);
}

/**
 * concatenate two strings, automaticly allocating enough space for their combination,
 * and automaticly freeing the inputs.
 */
char* concatenate_and_free_strings(char* first, char* twoth) {
    char* output = malloc((strlen(first) + strlen(twoth) + 1)*sizeof(char));
    strcpy(output, first);
    free(first);
    strcat(output, twoth);
    free(twoth);
    return output;
}

/**
 * copy an int array and return a pointer to the copy
 */
int* copy_of_ia(const int original[], int length) {
    if (length > 0) {
        int *copy = malloc(length*sizeof(int));
        for (int k = 0; k < length; k ++)
            copy[k] = original[k];
        return copy;
    }
    else {
        return NULL;
    }
}

/**
 * copy a double array and return a pointer to the copy
 */
double* copy_of_da(const double original[], int length) {
    double* copy = malloc(length*sizeof(double));
    for (int k = 0; k < length; k ++)
        copy[k] = original[k];
    return copy;
}

/**
 * copy a SparseArray and return the new identical object
 */
struct SparseArray copy_of_sa(struct SparseArray original) {
    struct SparseArray copy = {.ndim=original.ndim, .nitems=original.nitems};
    if (copy.nitems > 0) {
        copy.indices = copy_of_ia(original.indices, original.nitems*original.ndim);
        copy.values = copy_of_da(original.values, original.nitems);
    }
    return copy;
}

/**
 * efficiently search a sorted int array to find the index of value.
 * if value does not exist in the array, return -1.
 * @param array the 2D array thru which to search
 * @param length the number of rows in the array
 * @param value the row that we want to locate in the array
 * @param the number of columns in the array
 * @return the index j such that array[j*ndim + k] == value[k] for all 0 <= k < ndim
 */
int binary_search(int array[], int length, int value[], int ndim) {
    int j_min = -1, j_max = length; // the current search bounds, exclusive
    while (j_max - j_min > 1) {
        int j_new = (j_min + j_max)/2;
        int comparison = compare(array + j_new*ndim, value, ndim);
        if (comparison < 0)
            j_min = j_new;
        else if (comparison > 0)
            j_max = j_new;
        else
            return j_new;
    }
    return -1;
}

/**
 * combine two SparseArrays into a new one created by applying the given operator to each
 * of their elements
 * @param operator the operation to perform on each pair of elements
 * @param a the first array
 * @param b the second array
 * @return the resulting array
 */
struct SparseArray elementwise_sa(enum Operator operator,
        struct SparseArray a, struct SparseArray b) {
    if (a.ndim != b.ndim) {
        printf("Error! SparseArrays have different numbers of dimensions\n");
        struct SparseArray null = {.ndim=-1};
        return null;
    }

    struct SparseArray c = {.ndim=a.ndim};

    if (a.nitems > 0 || b.nitems > 0) {
        // first, merge-sort the indices of a and b
        int* a_mapping = malloc((a.nitems + b.nitems)*sizeof(int));
        int* b_mapping = malloc((a.nitems + b.nitems)*sizeof(int));
        // they're both sorted, so iterate thru them simultaneously
        int j_a = 0, j_b = 0;
        int j_c = 0;
        while (j_a < a.nitems || j_b < b.nitems) {
            int *a_index = (j_a < a.nitems) ? a.indices + j_a*c.ndim : NULL;
            int *b_index = (j_b < b.nitems) ? b.indices + j_b*c.ndim : NULL;
            // each iteration, see which index comes next
            int comparison;
            if (a_index == NULL)
                comparison = 1; // 1: a b index comes next
            else if (b_index == NULL)
                comparison = -1; // -1: an a index comes next
            else
                comparison = compare(a_index, b_index, c.ndim);
            // if the next one should be a, assine it the next a index and mark b as nonparticipating
            if (comparison < 0) {
                a_mapping[j_c] = j_a;
                j_a ++;
                b_mapping[j_c] = - 1;
            }
                // if the next one should be b, assine it the next b index and mark a as nonparticipating
            else if (comparison > 0) {
                a_mapping[j_c] = - 1;
                b_mapping[j_c] = j_b;
                j_b ++;
            }
                // if they are both next, then they both get to participate!
            else {
                a_mapping[j_c] = j_a;
                j_a ++;
                b_mapping[j_c] = j_b;
                j_b ++;
            }
            // keep in mind that when multiplying, we only need to keep indices where both inputs have an element
            if (operator != MULTIPLY || (a_mapping[j_c] >= 0 && b_mapping[j_c] >= 0))
                j_c ++;
        }

        // now bild the new thing
        c.nitems = j_c;
        if (c.nitems > 0) {
            c.indices = malloc(c.nitems*c.ndim*sizeof(int));
            c.values = malloc(c.nitems*sizeof(double));
            for (j_c = 0; j_c < c.nitems; j_c ++) {
                j_a = a_mapping[j_c];
                j_b = b_mapping[j_c];
                // copy the indices from whencever they're defined
                for (int k = 0; k < c.ndim; k ++) {
                    if (j_a >= 0)
                        c.indices[j_c*c.ndim + k] = a.indices[j_a*c.ndim + k];
                    else
                        c.indices[j_c*c.ndim + k] = b.indices[j_b*c.ndim + k];
                }
                // and operate on the values
                if (j_a >= 0) {
                    if (j_b >= 0) {
                        if (operator == ADD)
                            c.values[j_c] = a.values[j_a] + b.values[j_b];
                        else if (operator == SUBTRACT)
                            c.values[j_c] = a.values[j_a] - b.values[j_b];
                        else if (operator == MULTIPLY)
                            c.values[j_c] = a.values[j_a]*b.values[j_b];
                        else {
                            printf("Error! %d is an illegal operation for SparseArrays.\n", operator);
                            struct SparseArray null = {.ndim=-1};
                            return null;
                        }
                    }
                    else
                        c.values[j_c] = a.values[j_a];
                }
                else {
                    if (operator == SUBTRACT) c.values[j_c] = - b.values[j_b];
                    else c.values[j_c] = b.values[j_b];
                }
            }
        }

        free(a_mapping);
        free(b_mapping);
    }
    else { // on the off-chance these are both empty
        c.nitems = 0; // don't allocate any memory
    }

    return c;
}

struct SparseArray add_sa(struct SparseArray a, struct SparseArray b) {
    return elementwise_sa(ADD, a, b);
}

struct SparseArray subtract_sa(struct SparseArray a, struct SparseArray b) {
    return elementwise_sa(SUBTRACT, a, b);
}

/**
 * multiply a SparseArray by a scalar
 */
struct SparseArray multiply_sa(struct SparseArray a, double factor) {
    struct SparseArray c = {.ndim=a.ndim, .nitems=a.nitems};
    if (a.nitems > 0) {
        c.indices = copy_of_ia(a.indices, a.ndim*a.nitems);
        c.values = malloc(c.nitems*sizeof(double));
        for (int j = 0; j < c.nitems; j ++)
            c.values[j] = a.values[j]*factor;
    }
    return c;
}

/**
 * multiply two SparseArrays and expand their dimensions to get al the combinations
 */
struct SparseArray outer_multiply_sa(struct SparseArray a, struct SparseArray b) {
    struct SparseArray c = {.ndim=a.ndim + b.ndim, .nitems=a.nitems*b.nitems};
    if (c.nitems > 0) {
        c.indices = malloc(c.nitems*c.ndim*sizeof(int));
        c.values = malloc(c.nitems*sizeof(double));
        for (int j_a = 0; j_a < a.nitems; j_a ++) {
            for (int j_b = 0; j_b < b.nitems; j_b ++) {
                int j_c = j_a*b.nitems + j_b;
                for (int k = 0; k < a.ndim; k ++)
                    c.indices[j_c*c.ndim + k] = a.indices[j_a*a.ndim + k];
                for (int k = a.ndim; k < c.ndim; k ++)
                    c.indices[j_c*c.ndim + k] = b.indices[j_b*b.ndim + (k - a.ndim)];
                c.values[j_c] = a.values[j_a]*b.values[j_b];
            }
        }
    }
    return c;
}

/**
 * compute the dot product two dense arrays
 */
double dot_product_nda(double* a, double* b, int size) {
    double product = 0;
    for (int i = 0; i < size; i ++)
        product += a[i]*b[i];
    return product;
}

/**
 * compute the dot product of a SparseArray with a dense array
 */
double dot_product_sa(struct SparseArray a, double* b, int* b_shape) {
    double product = 0;
    for (int j = 0; j < a.nitems; j ++) {
        int l = 0;
        int* index = a.indices + j*a.ndim;
        for (int k = 0; k < a.ndim; k ++) {
            l = l*b_shape[k] + index[k];
        }
        product += a.values[j]*b[l];
    }
    return product;
}

/**
 * combine two SparseArrayArrays into a new one created by applying the given operator to
 * each of their elements
 * @param operator the operation to perform on each pair of SparseArrays
 * @param a the first array
 * @param b the second array
 * @return the resulting array
 */
struct SparseArrayArray elementwise_saa(enum Operator operator,
        struct SparseArrayArray a, struct SparseArrayArray b) {
    // check their shapes breefly; no need to examine them in detail
    if (a.ndim != b.ndim || a.size != b.size || a.element_ndim != b.element_ndim) {
        printf("Error! the SparseArrayArray dimensions don't look rite.\n");
        struct SparseArrayArray null = {.ndim=-1};
        return null;
    }

    struct SparseArrayArray c = {.ndim=a.ndim, .size=a.size, .element_ndim=a.element_ndim};
    c.shape = copy_of_ia(a.shape, a.ndim);

    if (c.size > 0) {
        c.elements = malloc(c.size*sizeof(struct SparseArray));
        for (int i = 0; i < c.size; i ++)
            c.elements[i] = elementwise_sa(operator, a.elements[i], b.elements[i]);
    }

    return c;
}

EXPORT struct SparseArrayArray add_saa(
        struct SparseArrayArray a, struct SparseArrayArray b) {
    return elementwise_saa(ADD, a, b);
}

EXPORT struct SparseArrayArray subtract_saa(
        struct SparseArrayArray a, struct SparseArrayArray b) {
    return elementwise_saa(SUBTRACT, a, b);
}

EXPORT struct SparseArrayArray multiply_saa(
        struct SparseArrayArray a, struct SparseArrayArray b) {
    return elementwise_saa(MULTIPLY, a, b);
}

/**
 * perform matrix multiplication between two SparseArrayArrays.  the twoth will be dotted with
 * each SparseArray in the first, in such a way that its sparse structure is preserved.  thus,
 * the first must have the same sparse shape as the twoth has dense shape, and the result
 * will have the dense shape of the first and the sparse shape of the twoth.
 */
EXPORT struct SparseArrayArray matmul_saa(
        struct SparseArrayArray a, struct SparseArrayArray b) {
    // calculate what the two components of b's size would be if it were a compatible DenseSparseArray
    int dot_ndim = a.element_ndim; // these "dot" dimensions map to and dot with a's sparse dimensions
    if (dot_ndim > b.ndim) {
        printf("Error! the first matrix must have no more sparse dims than the twoth has dense ones.\n");
        struct SparseArrayArray null = {.ndim=-1};
        return null;
    }
    int row_ndim = b.ndim - dot_ndim; // these "row" dimensions do not participate in the matmul
    int row_size = 1;
    for (int k = dot_ndim; k < b.ndim; k ++)
        row_size *= b.shape[k];

    // define the output based on its known size and shape
    struct SparseArrayArray c = {.ndim=a.ndim + row_ndim,
                                 .size=a.size*row_size,
                                 .element_ndim=b.element_ndim};
    c.shape = malloc(c.ndim*sizeof(int));
    for (int k = 0; k < a.ndim; k ++)
        c.shape[k] = a.shape[k];
    for (int k = a.ndim; k < c.ndim; k ++)
        c.shape[k] = b.shape[k - a.ndim + dot_ndim];
    if (c.size > 0) {
        c.elements = malloc(c.size*sizeof(struct SparseArray));
        // then compute each row as a linear combination of rows from b
        for (int i_a = 0; i_a < a.size; i_a ++) {
            for (int l = 0; l < row_size; l ++) {
                struct SparseArray element = {.ndim=b.element_ndim, .nitems=0};
                for (int j = 0; j < a.elements[i_a].nitems; j ++) {
                    int i_b = 0;
                    for (int k = 0; k < dot_ndim; k ++)
                        i_b = i_b*b.shape[k] + a.elements[i_a].indices[j*dot_ndim + k];
                    struct SparseArray initial = element;
                    struct SparseArray change = multiply_sa(b.elements[i_b*row_size + l], a.elements[i_a].values[j]);
                    element = add_sa(initial, change);
                    free_sa(initial);
                    free_sa(change);
                }
                c.elements[i_a*row_size + l] = element;
            }
        }
    }

    return c;
}

/**
 * perform matrix multiplication between two SparseArrayArrays where the first one is transposed.
 */
EXPORT struct SparseArrayArray transpose_matmul_saa(
        struct SparseArrayArray a, const int* sparse_shape, int sparse_size, struct SparseArrayArray b) {
    // do some arithmetic with the number of dimensions
    int dot_ndim = a.ndim; // these "dot" dimensions map to and dot with a's dense dimensions
    if (dot_ndim > b.ndim) {
        printf("Error! the first matrix must have no more sparse dims than the twoth has dense ones.\n");
        struct SparseArrayArray null = {.ndim=-1};
        return null;
    }
    int row_ndim = b.ndim - dot_ndim; // these "row" dimensions do not participate in the matmul
    int row_size = 1;
    for (int k = dot_ndim; k < b.ndim; k ++)
        row_size *= b.shape[k];

    // define the output based on its known size and shape
    struct SparseArrayArray c = {.ndim=a.element_ndim + row_ndim,
                                 .size=sparse_size*row_size,
                                 .element_ndim=b.element_ndim};
    c.shape = malloc(c.ndim*sizeof(int));
    for (int k = 0; k < a.element_ndim; k ++)
        c.shape[k] = sparse_shape[k];
    for (int k = a.element_ndim; k < c.ndim; k ++)
        c.shape[k] = b.shape[k - a.element_ndim + dot_ndim];
    if (c.size > 0) {
        c.elements = malloc(c.size*sizeof(struct SparseArray));
        // instantiate it to zero arrays
        for (int i_c = 0; i_c < c.size; i_c ++) {
            struct SparseArray element = {.ndim=b.element_ndim, .nitems=0};
            c.elements[i_c] = element;
        }
        // then add up and bin the elements of b using the coefficients from a
        for (int i_a = 0; i_a < a.size; i_a ++) {
            for (int j = 0; j < a.elements[i_a].nitems; j ++) { // TODO I dislike this; I should just reimplement actual transposes
                int i_back = 0;
                for (int k = 0; k < a.element_ndim; k ++)
                    i_back = i_back*sparse_shape[k] + a.elements[i_a].indices[j*a.element_ndim + k];
                for (int l = 0; l < row_size; l ++) {
                    int i_c = i_back*row_size + l;
                    struct SparseArray initial = c.elements[i_c];
                    struct SparseArray change = multiply_sa(b.elements[i_a*row_size + l], a.elements[i_a].values[j]);
                    c.elements[i_c] = add_sa(initial, change);
                    free_sa(initial);
                    free_sa(change);
                }
            }
        }
    }

    return c;
}

/**
 * multiply two SparseArrayArrays elementwise and expand their sparse dimensions
 * @param a the first array
 * @param b the second array
 * @return the resulting array
 */
EXPORT struct SparseArrayArray outer_multiply_saa(
        struct SparseArrayArray a, struct SparseArrayArray b) {
    // check their shapes breefly; no need to examine them in detail
    if (a.ndim != b.ndim || a.size != b.size) {
        printf("Error! the SparseArrayArray dimensions don't look rite.\n");
        struct SparseArrayArray null = {.ndim=-1};
        return null;
    }

    struct SparseArrayArray c = {.ndim=a.ndim, .size=a.size};
    c.element_ndim = a.element_ndim + b.element_ndim;
    c.shape = copy_of_ia(a.shape, a.ndim);

    if (c.size > 0) {
        c.elements = malloc(c.size*sizeof(struct SparseArray));
        for (int i = 0; i < c.size; i ++)
            c.elements[i] = outer_multiply_sa(a.elements[i], b.elements[i]);
    }

    return c;
}

/**
 * create an empty SparseArrayArray
 * @param dense_shape
 * @param sparse_shape
 * @return
 */
EXPORT struct SparseArrayArray zeros(
        int dense_ndim, int dense_shape[], int sparse_ndim) {
    struct SparseArrayArray a = {.ndim=dense_ndim, .element_ndim=sparse_ndim};
    a.shape = copy_of_ia(dense_shape, dense_ndim);
    a.size = product(dense_shape, dense_ndim);
    if (a.size > 0) {
        a.elements = malloc(a.size*sizeof(struct SparseArray));
        for (int i = 0; i < a.size; i ++) {
            struct SparseArray element = {.ndim=sparse_ndim, .nitems=0};
            a.elements[i] = element;
        }
    }
    return a;
}

/**
 * create a SparseArrayArray where each element has a single 1 at an index matching its own index
 * @param ndim the number of dense dimensions and the number of sparse dimensions (half the total number of dimensions)
 * @param shape the shape of the SparseArrayArray, which is also the shape of each of its elements
 * @param add_zero if the array is 1d, this will add an extra zero element
 */
EXPORT struct SparseArrayArray identity(
        int ndim, const int shape[], _Bool add_zero) {
    int original_size = product(shape, ndim);

    struct SparseArrayArray a = {.ndim=ndim, .size=original_size, .element_ndim=ndim};
    a.shape = copy_of_ia(shape, ndim);

    if (add_zero) {
        if (ndim == 1) {
            a.shape[0] += 1;
            a.size += 1;
        }
        else
            printf("Error! add_zero can only be used on 1d inputs.\n");
    }

    // keep track of the index we're on as we iterate thru the SparseArrayArray
    if (a.size > 0) {
        a.elements = malloc(a.size*sizeof(struct SparseArray));
        int* index = calloc(a.ndim, sizeof(int));
        for (int i = 0; i < a.size; i ++) {
            if (i < original_size) {
                    // create the single-value SparseArray
                    struct SparseArray element = {.ndim=ndim, .nitems=1};
                    element.indices = copy_of_ia(index, a.ndim);
                    element.values = malloc(sizeof(double));
                    element.values[0] = 1.;
                    a.elements[i] = element;
            }
            else {
                // or create the null array if we're at that point
                struct SparseArray zero = {.ndim=ndim, .nitems=0};
                a.elements[i] = zero;
            }

            // increment the index
            for (int k = a.ndim - 1; k >= 0; k --) {
                index[k] += 1;
                if (index[k] >= shape[k]) {
                    index[k] = 0;
                    continue;
                }
                else
                    break;
            }
        }
        free(index);
    }

    return a;
}

/**
 * stack a series of 1dx1d SparseArrayArray verticly to create a new larger SparseArrayArray
 * @param ndim the number of dense dimensions and the number of sparse dimensions (half the total number of dimensions)
 * @param shape the shape of the SparseArrayArray, which is also the shape of each of its elements
 */
EXPORT struct SparseArrayArray concatenate(
        const struct SparseArrayArray* elements, int length) {
    if (length == 0) {
        printf("Error!  it would be nice if this worked for length==0 but I don't know how to infer element_ndim then.\n");
        struct SparseArrayArray null = {.ndim=-1};
        return null;
    }
    for (int j = 0; j < length; j ++) {
        if (elements[j].ndim != 1) {
            printf("Error!  concatenate only works with 1d SparseArrayArrays!\n");
            struct SparseArrayArray null = {.ndim=-1};
            return null;
        }
    }
    struct SparseArrayArray a = {.ndim=1, .element_ndim=elements[0].element_ndim};
    a.shape = malloc(sizeof(int));
    a.shape[0] = 0;
    for (int j = 0; j < length; j ++)
        a.shape[0] += elements[j].shape[0];
    a.size = product(a.shape, a.ndim);

    // keep track of the index we're on as we iterate thru the SparseArrayArray
    if (a.size > 0) {
        a.elements = malloc(a.size*sizeof(struct SparseArray));
        int i = 0;
        for (int j = 0; j < length; j ++) {
            for (int k = 0; k < elements[j].shape[0]; k ++) {
                  a.elements[i] = copy_of_sa(elements[j].elements[k]);
                  i ++;
            }
	    }
    }
    return a;
}

/**
 * take a SparseArray that may have zero elements and fix it
 */
struct SparseArray remove_zeros(struct SparseArray input, int free_input) {
    struct SparseArray output = {.ndim=input.ndim};
    output.nitems = 0;
    for (int j = 0; j < input.nitems; j ++)
        if (input.values[j] != 0)
            output.nitems += 1;
    if (output.nitems > 0) {
        output.indices = malloc(output.nitems*output.ndim*sizeof(int));
        output.values = malloc(output.nitems*sizeof(double));
        int j_out = 0;
        for (int j_in = 0; j_in < input.nitems; j_in ++) {
            if (input.values[j_in] != 0) {
                    for (int k = 0; k < output.ndim; k ++)
                        output.indices[j_out*output.ndim + k] = input.indices[j_in*output.ndim + k];
                    output.values[j_out] = input.values[j_in];
                    j_out ++;
                }
        }
    }
    if (free_input)
        free_sa(input);
    return output;
}

/**
 * take a SparseArray with its items out of order and put them in order
 */
struct SparseArray sort_in_place_sa(struct SparseArray a) {
    for (int j0 = 0; j0 < a.nitems; j0 ++) {
        int j_min = j0;
        for (int j1 = j0 + 1; j1 < a.nitems; j1 ++)
            if (compare(a.indices + j1*a.ndim, a.indices + j_min*a.ndim, a.ndim) < 0)
                j_min = j1;
        int* min_index = copy_of_ia(a.indices + j_min*a.ndim, a.ndim);
        double min_value = a.values[j_min];
        for (int k = 0; k < a.ndim; k ++)
            a.indices[j_min*a.ndim + k] = a.indices[j0*a.ndim + k];
        a.values[j_min] = a.values[j0];
        for (int k = 0; k < a.ndim; k ++)
            a.indices[j0*a.ndim + k] = min_index[k];
        a.values[j0] = min_value;
        free(min_index);
    }
    return a;
}

/**
 * take a SparseArray that may have adjacent duplicate elements and remove them
 */
struct SparseArray combine_duplicates(struct SparseArray input, int free_input) {
    int ndim = input.ndim;
    struct SparseArray output = {.ndim=ndim};
    output.nitems = 0;
    for (int j = 0; j < input.nitems; j ++)
        if (j == input.nitems - 1 || !array_equal(input.indices + j*ndim,
                                                  input.indices + (j+1)*ndim, ndim))
            output.nitems += 1;
    if (output.nitems > 0) {
        output.indices = malloc(output.nitems*output.ndim*sizeof(int));
        output.values = calloc(output.nitems, sizeof(double));
        int j_out = 0;
        for (int j_in = 0; j_in < input.nitems; j_in ++) {
            for (int k = 0; k < output.ndim; k ++)
                output.indices[j_out*output.ndim + k] = input.indices[j_in*output.ndim + k];
            output.values[j_out] += input.values[j_in];
            if (j_in == input.nitems - 1 || !array_equal(input.indices + j_in*ndim,
                                                         input.indices + (j_in+1)*ndim, ndim))
                j_out ++;
        }
    }
    if (free_input)
        free_sa(input);
    return output;
}

/**
 * create a SparseArrayArray based on the given indices and values.  they will be sorted.
 * @param nitems the number of items in each SparseArray
 * @param indices the indices of every nonzero item as an ndarray with shape dense_shape + nitems + sparse_ndim
 * @param values the value of every nonzero item as an ndarray with shape dense_shape + nitems
 */
EXPORT struct SparseArrayArray new_saa(
        int dense_ndim, const int dense_shape[], int nitems, int sparse_ndim,
        const int* indices, const double* values) {
    struct SparseArrayArray a = {.ndim=dense_ndim, .element_ndim=sparse_ndim};
    a.shape = copy_of_ia(dense_shape, dense_ndim);
    a.size = product(dense_shape, dense_ndim);
    if (a.size > 0) {
        a.elements = malloc(a.size*sizeof(struct SparseArray));
        for (int i = 0; i < a.size; i ++) {
            struct SparseArray element = {.ndim=sparse_ndim, .nitems=nitems};
            element.indices = copy_of_ia(indices + i*nitems*sparse_ndim, nitems*sparse_ndim);
            element.values = copy_of_da(values + i*nitems, nitems);
            element = remove_zeros(element, 1);
            element = sort_in_place_sa(element);
            element = combine_duplicates(element, 1);
            a.elements[i] = element;
        }
    }
    return a;
}

/**
 * take a linear index from an array of known shape and compute the indices along each axis.
 */
int* decompose_index(int unravelled, const int shape[], int ndim) {
    int* components = malloc(ndim*sizeof(int));
    for (int k = ndim - 1; k >= 0; k --) {
        components[k] = unravelled%shape[k];
        unravelled = unravelled/shape[k];
    }
    return components;
}

/**
 * take a 1d index on an nd-array of some shape and figure out what it would be if some of
 * the elements of shape were replaced with 1.  useful for summing as well as broadcasting.
 * @param old_index the original index, set to go thru an array in c-contiguus order
 * @param old_shape the shape used to produce the original index
 * @param new_shape a new shape that is the same as old_shape, except that some nonzero values may be replaced with 1.
 */
int broadcast_index(int old_index, const int old_shape[], const int new_shape[], int ndim) {
    int old_stride = product(old_shape, ndim);
    int new_index = 0;
    for (int k = 0; k < ndim; k ++) {
        old_stride /= old_shape[k];
        int index_k = old_index/old_stride;
        if (new_shape[k] != 1)
            new_index = new_index*new_shape[k] + index_k;
        old_index %= old_stride;
    }
    return new_index;
}

/**
 * create a new SparseArrayArray from the combination of an existing one with a
 * ndarray by way of elementwise operation on the Sparse one's nonzero indices.
 * @param operator the operation to perform on each pair of SparseArrays
 * @param a the first array
 * @param shape the shape of the numpy array (it mite have some 1s where the SparseArrayArray has other numbers)
 * @param b the numpy array
 * @return the resulting array
 */
struct SparseArrayArray elementwise_nda(enum Operator operator, struct SparseArrayArray a, const double* b, const int b_shape[]) {
    for (int k = 0; k < a.ndim; k ++) {
        if (b_shape[k] != a.shape[k] && b_shape[k] != 1) {
            printf("Error! these array dimensions don't match and can't be broadcast together.");
            struct SparseArrayArray null = {.ndim=-1};
            return null;
        }
    }

    struct SparseArrayArray c = {.ndim=a.ndim, .size=a.size, .element_ndim=a.element_ndim};
    c.shape = copy_of_ia(a.shape, a.ndim);

    // set each element of the new SparseArrayArray one at a time
    if (a.size > 0) {
        c.elements = malloc(a.size*sizeof(struct SparseArray));
        for (int i_a = 0; i_a < a.size; i_a ++) {
            struct SparseArray old = a.elements[i_a];
            struct SparseArray new = {.ndim=old.ndim, .nitems=old.nitems};
            if (new.nitems > 0) {
                new.indices = copy_of_ia(old.indices, old.nitems*old.ndim);
                new.values = malloc(old.nitems*sizeof(double));
                // first figure out how to broadcast a value from the ndarray to this spot
                int i_b = broadcast_index(i_a, a.shape, b_shape, a.ndim);
                // then perform the operation
                for (int j = 0; j < old.nitems; j ++) {
                    if (operator == MULTIPLY)
                        new.values[j] = old.values[j]*b[i_b];
                    else if (operator == DIVIDE)
                        new.values[j] = old.values[j]/b[i_b];
                    else {
                        printf("Error! %d is an illegal operation for a SparseArrayArray and a dense array.\n", operator);
                        struct SparseArrayArray null = {.ndim=-1};
                        return null;
                    }
                }
            }
            c.elements[i_a] = new;
        }
    }
    return c;
}

EXPORT struct SparseArrayArray multiply_nda(
        struct SparseArrayArray a, const double* b, const int shape[]) {
    return elementwise_nda(MULTIPLY, a, b, shape);
}

EXPORT struct SparseArrayArray divide_nda(
        struct SparseArrayArray a, const double* b, const int shape[]) {
    return elementwise_nda(DIVIDE, a, b, shape);
}

/**
 * perform matrix multiplication between a SparseArrayArray and a plain dense array.  the dense array
 * will be dotted with each SparseArray in the SparseArrayArray, and it will all also be densified.
 */
EXPORT void matmul_nda(
        struct SparseArrayArray a, const double* b, const int b_shape[], int b_ndim, double* result) {
    // calculate what the two components of b's size would be if it were a compatible DenseSparseArray
    int dot_ndim = a.element_ndim; // these dimensions map to and dot with a's sparse dimensions
    int row_size = 1; // these "row" dimensions do not participate in the matmul
    for (int k = dot_ndim; k < b_ndim; k ++)
        row_size *= b_shape[k];

    // then compute each row as a linear combination of rows from b
    for (int i_a = 0; i_a < a.size; i_a ++) {
        for (int l = 0; l < row_size; l ++)
            result[i_a*row_size + l] = 0;
        for (int j = 0; j < a.elements[i_a].nitems; j ++) {
            int i_b = 0;
            for (int k = 0; k < dot_ndim; k ++)
                i_b = i_b*b_shape[k] + a.elements[i_a].indices[j*dot_ndim + k];
            double coef = a.elements[i_a].values[j];
            for (int l = 0; l < row_size; l ++)
                result[i_a*row_size + l] += coef*b[i_b*row_size + l];
        }
    }
}

/**
 * perform matrix multiplication between a SparseArrayArray and a plain dense array, but where the
 * SparseArrayArray is transposed first (only works for 1 dense dim and 1 sparse dim).
 */
EXPORT void transpose_matmul_nda(
        struct SparseArrayArray a, int sparse_size, const double* b, const int b_shape[], int b_ndim, double* result) {
    if (a.ndim != 1) {
        printf("Error! this method only works when a is 1D.\n");
        return;
    }
    else if (a.size >= 1 && a.element_ndim != 1) {
        printf("Error! this method only works when a's elements are 1D.\n");
        return;
    }
    else if (b_ndim <= 0 || a.size != b_shape[0]) {
        printf("Error! these shapes are not matmul-compatible.\n");
        return;
    }

    // calculate what the two components of b's size would be if it were a compatible DenseSparseArray
    int row_size = 1; // these "row" dimensions do not participate in the matmul
    for (int k = 1; k < b_ndim; k ++)
        row_size *= b_shape[k];

    // initialize result to 0
    for (int i_c = 0; i_c < sparse_size; i_c ++)
        for (int l = 0; l < row_size; l ++)
            result[i_c*row_size + l] = 0;

    // then compute each row as a linear combination of rows from b
    for (int i_a = 0; i_a < a.size; i_a ++) {
        for (int j = 0; j < a.elements[i_a].nitems; j ++) {
            int i_c = a.elements[i_a].indices[j];
            double coef = a.elements[i_a].values[j];
            if (i_c < 0 || i_c >= sparse_size) {
                printf("Error! the index %d is outside of the specified sparse size %d.\n", i_c, sparse_size);
                return;
            }
            for (int l = 0; l < row_size; l ++)
                result[i_c*row_size + l] += coef*b[i_a*row_size + l];
        }
    }
}

/**
 * calculate an elementwise operation between the values of a SparseArrayArray and a scalar.
 */
struct SparseArrayArray elementwise_f(enum Operator operator, struct SparseArrayArray a, double b) {
    struct SparseArrayArray c = {.ndim=a.ndim, .size=a.size, .element_ndim=a.element_ndim};
    c.shape = copy_of_ia(a.shape, a.ndim);

    if (a.size > 0) {
        c.elements = malloc(a.size*sizeof(struct SparseArray));
        for (int i = 0; i < a.size; i ++) {
            struct SparseArray old = a.elements[i];
            struct SparseArray new = {.ndim=old.ndim, .nitems=old.nitems};
            if (new.nitems > 0) {
                new.indices = copy_of_ia(old.indices, old.nitems*old.ndim);

                new.values = malloc(old.nitems*sizeof(double));
                for (int j = 0; j < old.nitems; j ++) {
                    if (operator == MULTIPLY)
                        new.values[j] = old.values[j]*b;
                    else if (operator == DIVIDE)
                        new.values[j] = old.values[j]/b;
                    else if (operator == POWER)
                        new.values[j] = pow(old.values[j], b);
                    else {
                        printf("Error! %d is an illegal operation for a SparseArrayArray and a float.\n", operator);
                        struct SparseArrayArray null = {.ndim=-1};
                        return null;
                    }
                }
            }
            c.elements[i] = new;
        }
    }
    return c;
}

EXPORT struct SparseArrayArray multiply_f(
        struct SparseArrayArray a, double factor) {
    return elementwise_f(MULTIPLY, a, factor);
}

EXPORT struct SparseArrayArray divide_f(
        struct SparseArrayArray a, double divisor) {
    return elementwise_f(DIVIDE, a, divisor);
}

EXPORT struct SparseArrayArray power_f(
        struct SparseArrayArray a, double power) {
    return elementwise_f(POWER, a, power);
}

/**
 * perform an elementwise unary operation: the absolute value
 */
EXPORT struct SparseArrayArray abs_saa(struct SparseArrayArray a) {
    struct SparseArrayArray c = {.ndim=a.ndim, .size=a.size, .element_ndim=a.element_ndim};
    c.shape = copy_of_ia(a.shape, a.ndim);
    if (c.size > 0) {
        c.elements = malloc(c.size*sizeof(struct SparseArray));
        for (int i = 0; i < c.size; i ++) {
            struct SparseArray new = {.ndim=a.elements[i].ndim, .nitems=a.elements[i].nitems};
            if (new.nitems > 0) {
                new.indices = copy_of_ia(a.elements[i].indices, a.elements[i].nitems*a.elements[i].ndim);
                new.values = malloc(a.elements[i].nitems*sizeof(double));
                for (int j = 0; j < new.nitems; j ++) {
                    if (a.elements[i].values[j] >= 0)
                        new.values[j] = a.elements[i].values[j];
                    else
                        new.values[j] = -a.elements[i].values[j];
                }
            }
            c.elements[i] = new;
        }
    }
    return c;
}

/**
 * perform an elementwise reduction operation: the maximum
 */
EXPORT double min_saa(struct SparseArrayArray a) {
    double min = 0;
    for (int i = 0; i < a.size; i ++)
        for (int j = 0; j < a.elements[i].nitems; j ++)
            if (a.elements[i].values[j] < min)
                min = a.elements[i].values[j];
    return min;
}

/**
 * perform an elementwise reduction operation: the maximum
 */
EXPORT double max_saa(struct SparseArrayArray a) {
    double max = 0;
    for (int i = 0; i < a.size; i ++)
        for (int j = 0; j < a.elements[i].nitems; j ++)
            if (a.elements[i].values[j] > max)
                max = a.elements[i].values[j];
    return max;
}

/**
 * determine whether all of the eigenvalues of this array are positive
 */
EXPORT int is_positive_definite(struct SparseArrayArray a) {
    for (int size = 0; size < a.size; size ++) {
        double det = 1; // TODO: actually implement this
        if (det <= 0)
            return 0;
    }
    return 1;
}

/**
 * sum the squares of the elements of a vector
 */
double sqr_nda(double* array, int size) {
    double sum = 0;
    for (int i = 0; i < size; i ++)
        sum += array[i]*array[i];
    return sum;
}

/**
 * evaluate the magnitude of a vector
 */
double norm_nda(double* array, int size) {
    return sqrt(sqr_nda(array, size));
}

/**
 * find the max magnitude in this array
 */
int all_abs_lessequal(const double* array, int size, double threshold) {
    for (int i = 0; i < size; i ++)
        if (array[i] > threshold || -array[i] > threshold)
            return 0;
    return 1;
}

/**
 * iteratively invert a matrix and multiply that by the given vector,
 * using the conjugate gradients technique.  a must be square,
 * and b must have the same size and shape as a.
 */
EXPORT int inverse_matmul_nda(
        struct SparseArrayArray a, const double* b,
        double magnitude_tolerance, const double* guess, double* out) {
    // check if the solution is trivial, because that will break this algorithm
    if (all_abs_lessequal(b, a.size, 0.)) {
        for (int i = 0; i < a.size; i ++)
            out[i] = b[i];
        return 0;
    }

    // transfer guess into out and then stop reading guess
    for (int i = 0; i < a.size; i ++)
        out[i] = guess[i];

    // check the stop condition just in case the gess is correct (sometimes it is)
    double component_tolerance = magnitude_tolerance/sqrt(a.size);
    double* residue_old = malloc(a.size*sizeof(double));
    for (int i = 0; i < a.size; i ++)
        residue_old[i] = b[i] - dot_product_sa(a.elements[i], out, a.shape);
    if (all_abs_lessequal(residue_old, a.size, component_tolerance) ||
        norm_nda(residue_old, a.size) <= magnitude_tolerance) {
        free(residue_old);
        return 0;
    }
    // initialize the loop to step in the direction of steepest descent
    double* direction = malloc(a.size*sizeof(double));
    for (int i = 0; i < a.size; i ++)
        direction[i] = residue_old[i];
    double* Ad = malloc(a.size*sizeof(double));
    double* residue_new = malloc(a.size*sizeof(double));

    // do the iterations
    int num_iterations = 0;
    while (1) {
        // precompute this matrix product for later
        for (int i = 0; i < a.size; i ++)
            Ad[i] = dot_product_sa(a.elements[i], direction, a.shape);
        // take the step
        double alpha = sqr_nda(residue_old, a.size)/dot_product_nda(direction, Ad, a.size);
        for (int i = 0; i < a.size; i ++) {
            out[i] += alpha*direction[i];
            residue_new[i] = residue_old[i] - alpha*Ad[i];
        }
        // check the stop condition
        if (all_abs_lessequal(residue_new, a.size, component_tolerance) ||
            norm_nda(residue_new, a.size) <= magnitude_tolerance) {
            // make sure to double check the stop condition with the exact residue
            for (int i = 0; i < a.size; i ++)
                residue_new[i] = b[i] - dot_product_sa(a.elements[i], out, a.shape);
            if (all_abs_lessequal(residue_new, a.size, component_tolerance) ||
                norm_nda(residue_new, a.size) <= magnitude_tolerance) {
                free(residue_old);
                free(residue_new);
                free(direction);
                free(Ad);
                return 0;
            }
        }
        // update the step direction
        double beta = sqr_nda(residue_new, a.size)/sqr_nda(residue_old, a.size);
        for (int i = 0; i < a.size; i ++) {
            direction[i] = residue_new[i] + beta*direction[i];
            residue_old[i] = residue_new[i];
        }
        // check the backup stop condition
        num_iterations ++;
        if (num_iterations > 10*a.size) {
            printf("conjugate gradients did not converge; we may be in a saddle region.\n");
            free(residue_old);
            free(residue_new);
            free(direction);
            free(Ad);
            return 1;
        }
    }
}

/**
 * sum a SparseArrayArray along one of the dense axes.
 * @param a the first array
 * @param axis the axis of the array along wihch to perform the sum
 * @return the resulting array
 */
EXPORT struct SparseArrayArray sum_along_axis(
        struct SparseArrayArray a, int axis) {
    if (axis < 0 || axis >= a.ndim) {
        printf("Error! the specified axis (%d out of %d) does not exist.\n", axis, a.ndim);
        struct SparseArrayArray null = {.ndim=-1};
        return null;
    }

    // establish the new number of dimensions
    struct SparseArrayArray c = {.ndim=a.ndim - 1, .element_ndim=a.element_ndim};
    c.size = a.size/a.shape[axis];

    // set up the new shape
    int* b_shape = malloc(a.ndim*sizeof(int)); // a dummy shape that allows values to be broadcast from a to c
    for (int k = 0; k < a.ndim; k ++) {
        if (k == axis)
            b_shape[k] = 1;
        else
            b_shape[k] = a.shape[k];
    }
    c.shape = malloc(c.ndim*sizeof(int)); // the actual reduced shape of c
    for (int k = 0; k < c.ndim; k ++) {
        if (k < axis)
            c.shape[k] = a.shape[k];
        else
            c.shape[k] = a.shape[k + 1];
    }

    if (c.size > 0) {
        c.elements = malloc(c.size*sizeof(struct SparseArray));
        struct SparseArray zero = {.ndim=a.element_ndim, .nitems=0};
        for (int i = 0; i < c.size; i ++)
            c.elements[i] = zero;
    }

    // sum everything together
    for (int i_a = 0; i_a < a.size; i_a ++) {
        int i_c = broadcast_index(i_a, a.shape, b_shape, a.ndim);
        if (a.elements[i_a].nitems > 0) {
            struct SparseArray old_sum = c.elements[i_c];
            c.elements[i_c] = add_sa(c.elements[i_c], a.elements[i_a]);
            free_sa(old_sum);
        }
    }

    free(b_shape);

    return c;
}

/**
 * sum along all of the sparse axes of a SparseArrayArray
 */
EXPORT void sum_all_sparse(
        struct SparseArrayArray a, double* result) {
    // set up the ndarray
    for (int l = 0; l < a.size; l ++)
        result[l] = 0;
    // then do the sum for each SparseArray
    for (int i = 0; i < a.size; i ++)
        for (int j = 0; j < a.elements[i].nitems; j ++)
            result[i] += a.elements[i].values[j];
}

/**
 * sum along all of the dense axes of a SparseArrayArray
 */
EXPORT void sum_all_dense(
        struct SparseArrayArray a, const int shape[], double* result) {
    // calculate the size
    int size = product(shape, a.element_ndim);

    // finally, histogram the values
    for (int l = 0; l < size; l ++)
        result[l] = 0;
    for (int i = 0; i < a.size; i ++) {
        // for each element of each element
        for (int j = 0; j < a.elements[i].nitems; j ++) {
            int* index = a.elements[i].indices + j*a.elements[i].ndim;
            int l = 0;
            for (int k = 0; k < a.elements[i].ndim; k ++) {
                if (index[k] < 0 || index[k] >= shape[k]) {
                    printf("Error! a SparseArray had an index outside of the given shape (%d out of %d).\n", index[k], shape[k]);
                    return;
                }
                l = l*shape[k] + index[k]; // find the index
            }
            result[l] += a.elements[i].values[j]; // and add it there
        }
    }
}

/**
 * expand this by turning every element into a little identity matrix
 */
EXPORT struct SparseArrayArray repeat_diagonally(
        struct SparseArrayArray a, const int* new_shape, int new_ndim) {
//    printf("repeat_diagonally(a, [");
//    for (int k = 0; k < new_ndim; k ++)
//        printf("%d,", new_shape[k]);
//    printf("], %d)\n", new_ndim);
    int new_size = product(new_shape, new_ndim);
//    printf("new_size = %f\n", new_size);
    struct SparseArrayArray c = {.ndim=a.ndim + new_ndim, .size=a.size*new_size};
    c.element_ndim = c.ndim;
    c.shape = malloc(c.ndim);
    for (int k = 0; k < a.ndim; k ++)
        c.shape[k] = a.shape[k];
    for (int k = a.ndim; k < c.ndim; k ++)
        c.shape[k] = new_shape[k - a.ndim];
    if (c.size > 0) {
        c.elements = malloc(c.size*sizeof(struct SparseArray));
        for (int i_a = 0; i_a < a.size; i_a ++) {
            for (int l = 0; l < new_size; l ++) {
                int i_c = i_a*new_size + l;
                struct SparseArray element = {.ndim=c.ndim, .nitems=a.elements[i_a].nitems};
                if (element.nitems > 0) {
                    element.indices = malloc(element.nitems*element.ndim*sizeof(int));
                    element.values = malloc(element.nitems*sizeof(double));
                    for (int j = 0; j < element.nitems; j ++) {
                        element.values[j] = a.elements[i_a].values[j];
                        for (int k = 0; k < a.ndim; k ++)
                            element.indices[j*c.ndim + k] = a.elements[i_a].indices[j*a.ndim + k];
                        int compound_index = l;
                        for (int k = c.ndim - 1; k >= a.ndim; k --) {
                            element.indices[j*c.ndim + k] = compound_index%new_shape[k - a.ndim];
                            compound_index /= new_shape[k - a.ndim];
                        }
                    }
                }
                c.elements[i_c] = element;
            }
        }
    }
    return c;
}

/**
 * modify the shape of a SparseArrayArray by adding dimensions to the beginning of its
 * shape, without changing its size or altering its values.
 */
EXPORT struct SparseArrayArray expand_dims(
        struct SparseArrayArray a, int new_ndim) {
    struct SparseArrayArray c = {.ndim=a.ndim + new_ndim, .size=a.size, .element_ndim=a.element_ndim};

    c.shape = malloc(c.ndim*sizeof(int));
    for (int k = 0; k < new_ndim; k ++)
        c.shape[k] = 1;
    for (int k = new_ndim; k < c.ndim; k ++)
        c.shape[k] = a.shape[k - new_ndim];

    if (c.size > 0) {
        c.elements = malloc(c.size*sizeof(struct SparseArray));
        for (int i = 0; i < c.size; i ++)
            c.elements[i] = copy_of_sa(a.elements[i]);
    }

    return c;
}

/**
 * index the array on one axis, extracting a slice that is all of the elements where the
 * index on that axis matches what's given
 */
EXPORT struct SparseArrayArray get_slice_saa(
        struct SparseArrayArray a, int index, int axis) {
    if (axis < 0 || axis >= a.ndim || index < 0 || index >= a.shape[axis]) {
        printf("Error! the specified slice (%d on axis %d out of %d) is out of bounds.\n", index, axis, a.ndim);
        struct SparseArrayArray null = {.ndim=-1};
        return null;
    }

    // calculate the new shape and size
    struct SparseArrayArray c = {.ndim=a.ndim - 1, .element_ndim=a.element_ndim};
    c.size = a.size / a.shape[axis];
    c.shape = malloc(c.ndim*sizeof(int));
    for (int k = 0; k < c.ndim; k ++) {
        if (k >= axis)
            c.shape[k] = a.shape[k + 1];
        else
            c.shape[k] = a.shape[k];
    }

    // reassine the indices
    if (c.size > 0) {
        c.elements = malloc(c.size*sizeof(struct SparseArray));
        int* c_index = calloc(c.ndim, sizeof(int));

        for (int i_c = 0; i_c < c.size; i_c ++) {
            int i_a = 0;
            for (int k = 0; k < a.ndim; k ++) {
                int a_index_k;
                if (k < axis)
                    a_index_k = c_index[k];
                else if (k == axis)
                    a_index_k = index;
                else
                    a_index_k = c_index[k - 1];
                int a_shape_k = a.shape[k];
                i_a = i_a*a_shape_k + a_index_k;
            }
            c.elements[i_c] = copy_of_sa(a.elements[i_a]);
            // increment the index
            for (int k = c.ndim - 1; k >= 0; k --) {
                c_index[k] += 1;
                if (c_index[k] >= c.shape[k]) {
                    c_index[k] = 0;
                    continue;
                }
                else
                    break;
            }
        }

        free(c_index);
    }

    return c;
}

/**
 * index the array on one axis, extracting a slice that is all of the elements where the
 * index on that axis matches what's given
 */
EXPORT void get_diagonal_saa(struct SparseArrayArray a, double* result) {
    for (int i = 0; i < a.size; i ++) {
        if (a.elements[i].ndim != a.ndim) {
            printf("Error! arrays with sparse axes that don't match their dense axes don't have diagonals.");
            return;
        }
        int* index = decompose_index(i, a.shape, a.ndim);
        int j = binary_search(a.elements[i].indices, a.elements[i].nitems, index, a.elements[i].ndim);
        free(index);
        if (j >= 0)
            result[i] = a.elements[i].values[j];
        else
            result[i] = 0;
    }
}

/**
 * rearrange a SparseArrayArray along a particular axis using a 1D int array as an index
 * @param a the array to be reindexed
 * @param index the array of indices at which to evaluate the given axis
 * @param length the number of values in index
 * @param axis the values of index represent indices along this axis
 */
EXPORT struct SparseArrayArray get_reindex_saa(
        struct SparseArrayArray a, const int index[], int length, int axis) {
    if (axis < 0 || axis >= a.ndim) {
        printf("Error! the specified axis (%d out of %d) does not exist.\n", axis, a.ndim);
        struct SparseArrayArray null = {.ndim=-1};
        return null;
    }

    // calculate the new shape and size
    struct SparseArrayArray c = {.ndim=a.ndim, .element_ndim=a.element_ndim};
    c.size = a.size / a.shape[axis] * length;
    c.shape = malloc(c.ndim*sizeof(int));
    for (int k = 0; k < c.ndim; k ++) {
        if (k == axis)
            c.shape[k] = length;
        else
            c.shape[k] = a.shape[k];
    }

    // reassine the indices
    if (c.size > 0) {
        c.elements = malloc(c.size*sizeof(struct SparseArray));
        int* c_index = calloc(a.ndim, sizeof(int));

        for (int i_c = 0; i_c < c.size; i_c ++) {
            int i_a = 0;
            for (int k = 0; k < a.ndim; k ++) {
                int a_index_k;
                if (k == axis)
                    a_index_k = index[c_index[k]];
                else
                    a_index_k = c_index[k];
                i_a = i_a*a.shape[k] + a_index_k;
            }
            c.elements[i_c] = copy_of_sa(a.elements[i_a]);
            // keep track of where we are in c
            for (int k = a.ndim - 1; k >= 0; k --) {
                c_index[k] += 1;
                if (c_index[k] >= c.shape[k]) {
                    c_index[k] = 0;
                    continue;
                }
                else
                    break;
            }
        }

        free(c_index);
    }

    return c;
}

/**
 * convert the first few sparse axes of a DenseSparseArray to dense ones
 */
EXPORT struct SparseArrayArray densify_axes(
        struct SparseArrayArray a, const int new_shape[], int new_ndim) {
    // first, you must determine the shape
    struct SparseArrayArray c = {.ndim=a.ndim + new_ndim, .element_ndim=a.element_ndim - new_ndim};
    c.shape = malloc(c.ndim*sizeof(int));
    for (int k = 0; k < a.ndim; k ++)
        c.shape[k] = a.shape[k];
    c.size = a.size;
    for (int k = a.ndim; k < c.ndim; k ++) {
        c.shape[k] = new_shape[k - a.ndim];
        c.size *= c.shape[k];
    }
    int new_size = c.size/a.size;

    // allocate the main array
    if (c.size > 0) {
        c.elements = malloc(c.size*sizeof(struct SparseArray));

        // then, set the values by binning the floats from each SparseArray
        for (int i_a = 0; i_a < a.size; i_a ++) {
            struct SparseArray source = a.elements[i_a];
            struct SparseArray* destination = c.elements + i_a*new_size;
            // iterate thru the values and build up the new child SparseArrays as you go
            int j_source_start = 0;
            int num_completed = 0;
            for (int j_source = 0; j_source <= source.nitems; j_source ++) {
                // bin each value into the relevant i_new
                int i_new;
                if (j_source < source.nitems) {
                    i_new = 0;
                    for (int k = 0; k < new_ndim; k ++)
                        i_new = i_new*new_shape[k] + source.indices[j_source*source.ndim + k]; // find the newly dense part of the index
                }
                else {
                    i_new = new_size;
                }
                // each time i_new changes, fill in the previous new SparseArray
                while (num_completed < i_new) {
                    if (j_source > j_source_start) {
                        // transfer any pending values from source into children
                        struct SparseArray child = {.ndim=source.ndim - new_ndim,
                                                    .nitems=j_source - j_source_start};
                        if (child.nitems > 0) {
                            child.indices = malloc(child.nitems*child.ndim*sizeof(int));
                            child.values = malloc(child.nitems*sizeof(double));
                            for (int j_child = 0; j_child < child.nitems; j_child ++) {
                                for (int k = 0; k < child.ndim; k ++)
                                    child.indices[j_child*child.ndim + k] = source.indices[(j_source_start + j_child)*source.ndim + (new_ndim + k)];
                                child.values[j_child] = source.values[j_source_start + j_child];
                            }
                        }
                        destination[num_completed] = child;
                        j_source_start = j_source;
                    }
                    else {
                        // or if there are no values for this child, leave it empty
                        struct SparseArray child = {.ndim=source.ndim - new_ndim,
                                                    .nitems=0, .indices=NULL, .values=NULL};
                        destination[num_completed] = child;
                    }
                    num_completed ++;
                }
            }
        }
    }

    return c;
}

/**
 * convert a SparseArrayArray to a plain dense ndarray (flattend)
 */
EXPORT void to_dense(
        struct SparseArrayArray a, const int sparse_shape[], int sparse_ndim, double* result) {
    // first, you must determine the shape
    if (a.size > 0 && a.element_ndim != sparse_ndim) {
        printf("Error! the number of sparse dimensions in this SparseArrayArray is %d, but you seem to think it %d.\n",
               a.element_ndim, sparse_ndim);
        return;
    }
    int total_ndim = a.ndim + sparse_ndim;
    int* shape = malloc(total_ndim*sizeof(int));
    for (int k = 0; k < a.ndim; k ++)
        shape[k] = a.shape[k];
    for (int k = a.ndim; k < total_ndim; k ++)
        shape[k] = sparse_shape[k - a.ndim];

    // calculate the size
    int size = product(shape, total_ndim);

    // finally, set the values
    for (int l = 0; l < size; l ++)
        result[l] = 0;
    for (int i = 0; i < a.size; i ++) {
        struct SparseArray element = a.elements[i];
        // for each element of each element
        for (int j = 0; j < element.nitems; j ++) {
            int* index = element.indices + j*element.ndim;
            int l = i;
            for (int k = a.ndim; k < total_ndim; k ++) {
                if (index[k - a.ndim] < 0 || index[k - a.ndim] >= shape[k]) {
                    printf("Error! a SparseArray had an index outside of the given shape (%d out of %d).\n", index[k - a.ndim], shape[k]);
                    return;
                }
                l = l*shape[k] + index[k - a.ndim]; // find the index
            }
            result[l] += element.values[j]; // and add it there
        }
    }

    free(shape);
}

/**
 * return the number of nonzero values in this array
 */
EXPORT int count_items(struct SparseArrayArray a) {
    int nitems = 0;
    for (int i = 0; i < a.size; i ++)
        nitems += a.elements[i].nitems;
    return nitems;
}

/**
 * print out a SparseArrayArray (don't forget to free the output after you're done with it)
 */
EXPORT char* to_string(struct SparseArrayArray a) {
    char* output = calloc(1, sizeof(char));
    for (int i = 0; i < a.size; i ++) {
        for (int j = 0; j < a.elements[i].nitems; j ++) {
            for (int k = 0; k < a.elements[i].ndim; k ++) {
                int index = a.elements[i].indices[j*a.elements[i].ndim + k];
                char* new_string = malloc(9*sizeof(char));
                snprintf(new_string, 9, "%d,", index);
                output = concatenate_and_free_strings(output, new_string);
            }
            double value = a.elements[i].values[j];
            char* new_string = malloc(20*sizeof(char));
            snprintf(new_string, 20, ": %g,  ", value);
            output = concatenate_and_free_strings(output, new_string);
        }
        char* new_string = malloc(3*sizeof(char));
        strcpy(new_string, ";\n");
        output = concatenate_and_free_strings(output, new_string);
    }
    return output;
}
