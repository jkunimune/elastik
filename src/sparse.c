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
    int* indices;
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
 * compare two arrays, taking the initial elements as most significant
 */
int less_than(const int a[], const int b[], int length) {
    for (int k = 0; k < length; k ++) {
        if (a[k] > b[k])
            return 0;
        else if (a[k] < b[k])
            return 1;
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
__declspec(dllexport) void free_saa(
        struct SparseArrayArray a) {
    for (int i = 0; i < a.size; i ++)
        free_sa(a.elements[i]);
    free(a.elements);
    free(a.shape);
}

/**
 * free the memory allocated to an nd-array, given the location of its memory.
 * this is really just an alias for the existing free function; I only have it
 * here so that I only haff to import one dll file.
 */
__declspec(dllexport) void free_nda(double* a) {
    free(a);
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
            int comparison = 0; // default to 0: a combined a and b comes next
            if (a_index == NULL)
                comparison = 1; // 1: a b index comes next
            else if (b_index == NULL)
                comparison = - 1; // -1: an a index comes next
            else {
                for (int k = 0; k < c.ndim; k ++) {
                    if (a_index[k] < b_index[k])
                        comparison = - 1; // -1: an a index comes next
                    else if (a_index[k] > b_index[k])
                        comparison = 1; // 1: a b index comes next
                    if (comparison != 0)
                        break;
                }
            }
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
    c.indices = copy_of_ia(a.indices, a.ndim*a.nitems);
    c.values = malloc(c.nitems*sizeof(double));
    for (int j = 0; j < c.nitems; j ++)
        c.values[j] = a.values[j]*factor;
    return c;
}

/**
 * compute the dot product of two SparseArrays
 */
double dot_product_sa(struct SparseArray a, struct SparseArray b) {
    if (a.ndim != b.ndim) {
        printf("Error!  can't dot things whose shapes don't match.");
        return NAN;
    }

		double product = 0;

    // they're both sorted, so iterate thru them simultaneously
    int j_a = 0, j_b = 0;
    while (j_a < a.nitems || j_b < b.nitems) {
        int *a_index = (j_a < a.nitems) ? a.indices + j_a*a.ndim : NULL;
        int *b_index = (j_b < b.nitems) ? b.indices + j_b*b.ndim : NULL;
        // each iteration, see which index comes next
        int comparison = 0; // default to 0: the selected a and b elements are equal in index
        if (a_index == NULL)
            comparison = 1; // 1: the b index is sooner
        else if (b_index == NULL)
            comparison = - 1; // -1: the a index is sooner
        else {
            for (int k = 0; k < a.ndim; k ++) {
                if (a_index[k] < b_index[k])
                    comparison = - 1; // -1: an a index comes next
                else if (a_index[k] > b_index[k])
                    comparison = 1; // 1: a b index comes next
                if (comparison != 0)
                    break;
            }
        }

        if (comparison < 0) // if the a index is sooner, pass it
            j_a ++;
        else if (comparison > 0) // if the b index is sooner, pass it
            j_b ++;
        else { // if they are simultaneus, product them
            product += a.values[j_a]*b.values[j_b];
            j_a ++;
            j_b ++;
        }
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
    if (a.ndim != b.ndim || a.size != b.size) {
        printf("Error! the SparseArrayArray dimensions don't look rite.\n");
        struct SparseArrayArray null = {.ndim=-1};
        return null;
    }

    struct SparseArrayArray c = {.ndim=a.ndim, .size=a.size};
    c.shape = copy_of_ia(a.shape, a.ndim);

    c.elements = malloc(c.size*sizeof(struct SparseArray));
    for (int i = 0; i < c.size; i ++)
        c.elements[i] = elementwise_sa(operator, a.elements[i], b.elements[i]);

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
 * perform matrix multiplication between two 2d SparseArrayArrays
 * @param a a SparseArrayArray with a single sparse dimension
 * @param b a SparseArrayArray with a single dense dimension
 */
EXPORT struct SparseArrayArray matmul_saa(
        struct SparseArrayArray a, struct SparseArrayArray b) {
    if (b.ndim < 1) {
        printf("Error! the second matrix needs at least one dense dim that I can line up with a's sparse dim.\n");
        struct SparseArrayArray null = {.ndim=-1};
        return null;
    }

    // calculate this stride in case b is multidimensional
    int row_size = 1;
    for (int k = 1; k < b.ndim; k ++)
        row_size *= b.shape[k];
    int sparse_ndim = b.elements[0].ndim;

    // start by defining the output based on its known size and shape
    struct SparseArrayArray c = {.ndim=a.ndim + b.ndim - 1, .size=a.size*row_size};
    c.shape = malloc(c.ndim*sizeof(int));
    for (int k = 0; k < a.ndim; k ++)
        c.shape[k] = a.shape[k];
    for (int k = a.ndim; k < c.ndim; k ++)
        c.shape[k] = b.shape[k - a.ndim + 1];

    // then compute each element as the linear combination of rows from b
    c.elements = malloc(c.size*sizeof(struct SparseArray));
    for (int i = 0; i < a.size; i ++) {
        struct SparseArray vector = a.elements[i];
        if (vector.ndim != 1) {
            printf("Error! the first matrix must only have one sparse dim, to match the dense dim on b.\n");
            struct SparseArrayArray null = {.ndim=-1};
            return null;
        }

        for (int l = 0; l < row_size; l ++) {
            struct SparseArray element = {.ndim=sparse_ndim, .nitems=0};
            for (int j = 0; j < vector.nitems; j ++) {
                struct SparseArray initial = element;
                struct SparseArray change = multiply_sa(
                        b.elements[vector.indices[j]*row_size + l], vector.values[j]);
                element = add_sa(initial, change);
                free_sa(initial);
                free_sa(change);
            }
            c.elements[i*row_size + l] = element;
        }
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
    struct SparseArrayArray a = {.ndim=dense_ndim};
    a.shape = copy_of_ia(dense_shape, dense_ndim);
    a.size = product(dense_shape, dense_ndim);
    a.elements = malloc(a.size*sizeof(struct SparseArray));
    for (int i = 0; i < a.size; i ++) {
        struct SparseArray element = {.ndim=sparse_ndim, .nitems=0};
        a.elements[i] = element;
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

    struct SparseArrayArray a = {.ndim=ndim, .size=original_size};
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

    return a;
}

/**
 * stack a series of 1dx1d SparseArrayArray verticly to create a new larger SparseArrayArray
 * @param ndim the number of dense dimensions and the number of sparse dimensions (half the total number of dimensions)
 * @param shape the shape of the SparseArrayArray, which is also the shape of each of its elements
 */
EXPORT struct SparseArrayArray concatenate(
        const struct SparseArrayArray* elements, int length) {
    for (int j = 0; j < length; j ++) {
        if (elements[j].ndim != 1) {
            printf("Error!  concatenate only works with 1d SparseArrayArrays!\n");
            struct SparseArrayArray null = {.ndim=-1};
            return null;
        }
    }
    struct SparseArrayArray a = {.ndim=1};
    a.shape = malloc(sizeof(int));
    a.shape[0] = 0;
    for (int j = 0; j < length; j ++)
        a.shape[0] += elements[j].shape[0];
    a.size = product(a.shape, a.ndim);

    // keep track of the index we're on as we iterate thru the SparseArrayArray
    a.elements = malloc(a.size*sizeof(struct SparseArray));
    int i = 0;
    for (int j = 0; j < length; j ++) {
        for (int k = 0; k < elements[j].shape[0]; k ++) {
	          a.elements[i] = copy_of_sa(elements[j].elements[k]);
	          i ++;
	      }
	  }
    return a;
}

/**
 * take a SparseArray that may have zero elements and fix it
 */
EXPORT struct SparseArray remove_zeros_and_free_sa(struct SparseArray input) {
    struct SparseArray output = {.ndim=input.ndim};
    output.nitems = 0;
    for (int j = 0; j < input.nitems; j ++)
        if (input.values[j] != 0)
            output.nitems += 1;
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
    free_sa(input);
    return output;
}

/**
 * take a SparseArray with its items out of order and put them in order
 */
EXPORT struct SparseArray sort_in_place_sa(struct SparseArray a) {
    for (int j0 = 0; j0 < a.nitems; j0 ++) {
        int j_min = j0;
        for (int j1 = j0 + 1; j1 < a.nitems; j1 ++)
            if (less_than(a.indices + j1*a.ndim, a.indices + j_min*a.ndim, a.ndim))
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
EXPORT struct SparseArray combine_duplicates_and_free(struct SparseArray input) {
    int ndim = input.ndim;
    struct SparseArray output = {.ndim=ndim};
    output.nitems = 0;
    for (int j = 0; j < input.nitems; j ++)
        if (j == input.nitems - 1 || !array_equal(input.indices + j*ndim,
                                                  input.indices + (j+1)*ndim, ndim))
            output.nitems += 1;
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
    free_sa(input);
    return output;
}

/**
 * create a SparseArrayArray based on the given indices and values.  they will be sorted.
 * @praam nitems the number of items in each SparseArray
 * @param indices the indices of every nonzero item as an ndarray with shape dense_shape + nelements + sparse_ndim
 * @param values the value of every nonzero item as an ndarray with shape dense_shape + nelements
 */
EXPORT struct SparseArrayArray new_saa(
        int dense_ndim, const int dense_shape[], int nitems, int sparse_ndim,
        const int* indices, const double* values) {
    struct SparseArrayArray a = {.ndim=dense_ndim};
    a.shape = copy_of_ia(dense_shape, dense_ndim);
    a.size = product(dense_shape, dense_ndim);
    a.elements = malloc(a.size*sizeof(struct SparseArray));
    for (int i = 0; i < a.size; i ++) {
        struct SparseArray element = {.ndim=sparse_ndim, .nitems=nitems};
        element.indices = copy_of_ia(indices + i*nitems*sparse_ndim, nitems*sparse_ndim);
        element.values = copy_of_da(values + i*nitems, nitems);
        element = remove_zeros_and_free_sa(element);
        element = sort_in_place_sa(element);
        element = combine_duplicates_and_free(element);
        a.elements[i] = element;
    }
    return a;
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

    struct SparseArrayArray c = {.ndim=a.ndim, .size=a.size};
    c.shape = copy_of_ia(a.shape, a.ndim);

    // set each element of the new SparseArrayArray one at a time
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
 * perform matrix multiplication between a 2d SparseArrayArray and a plain dense array
 * @param a a SparseArrayArray with a single sparse dimension
 */
EXPORT double* matmul_nda(
        struct SparseArrayArray a, const double* b, const int b_shape[], int b_ndim) {
    // start by defining the output based on its known size and shape
    int c_ndim = a.ndim + b_ndim - 1;
    int* c_shape = malloc(c_ndim*sizeof(int));
    for (int k = 0; k < a.ndim; k ++)
        c_shape[k] = a.shape[k];
    for (int k = a.ndim; k < c_ndim; k ++)
        c_shape[k] = b_shape[k - a.ndim + 1];

    // calculate this one stride that we need
    int row_size = 1;
    for (int k = 1; k < b_ndim; k ++)
        row_size *= b_shape[k];
    int c_size = a.size*row_size;
    double* c = malloc(c_size*sizeof(double));

    free(c_shape);

    // then compute each row as a linear combination of rows from b
    for (int i = 0; i < a.size; i ++) {
        struct SparseArray vector = a.elements[i];
        if (vector.ndim != 1) {
            printf("Error! the first matrix must only have one sparse dim, to match the zeroth dim of b.\n");
            return NULL;
        }

        for (int l = 0; l < row_size; l ++) {
            c[i*row_size + l] = 0.;
            for (int j = 0; j < vector.nitems; j ++)
                c[i*row_size + l] += vector.values[j]*b[vector.indices[j]*row_size + l];
        }
    }

    return c;
}

///**
// * orthogonally project a dense vector into the subspace described by a SparseArrayArray.
// * @param a a SparseArrayArray with a single sparse dimension (it need not have orthogonal rows)
// */
//EXPORT double* project_nda(
//        struct SparseArrayArray a, const double* b, const int b_shape[], int b_ndim) {
//    // calculate some sizes
//    int row_size = 1;
//    for (int k = 1; k < b_ndim; k ++)
//        row_size *= b_shape[k];
//    int b_size = b_shape[0]*row_size;
//
//    // orthogonalize the rows of a
//    struct SparseArray* orthogonal_rows = malloc(a.size*sizeof(struct SparseArray));
//    for (int i0 = 0; i0 < a.size; i0 ++) {
//        struct SparseArray old_vector = copy_of_sa(a.elements[i0]);
//        for (int i1 = 0; i1 < i0; i1 ++) {
//            double coefficient = dot_product_sa(old_vector, orthogonal_rows[i1])/
//                                 dot_product_sa(orthogonal_rows[i1], orthogonal_rows[i1]);
//            struct SparseArray scaled = multiply_sa(orthogonal_rows[i1], coefficient);
//            struct SparseArray new_vector = subtract_sa(old_vector, scaled);
//            free_sa(scaled);
//            free_sa(old_vector);
//            old_vector = new_vector;
//        }
//        orthogonal_rows[i0] = old_vector;
//    }
//
//    // then bild up the projection one basis vector at a time
//    double* c = calloc(b_size, sizeof(double));
//    for (int i = 0; i < a.size; i ++) {
//        struct SparseArray row = orthogonal_rows[i];
//        if (row.ndim != 1) {
//            printf("Error! the first matrix must only have one sparse dim.\n");
//            return NULL;
//        }
//
//				for (int l = 0; l < row_size; l ++) {
//				    double v_dot_b = 0;
//				    double v_sqr = 0;
//				    for (int j = 0; j < row.nitems; j ++) {
//				        v_dot_b += row.values[j]*b[row.indices[j]*row_size + l];
//				        v_sqr += row.values[j]*row.values[j];
//				    }
//		        double coef = v_dot_b/v_sqr;
//            for (int j = 0; j < row.nitems; j ++)
//                c[row.indices[j]*row_size + l] += coef*row.values[j];
//        }
//    }
//
//    for (int i = 0; i < a.size; i ++)
//        free_sa(orthogonal_rows[i]);
//    free(orthogonal_rows);
//
//    return c;
//}

struct SparseArrayArray elementwise_f(enum Operator operator, struct SparseArrayArray a, double b) {
    struct SparseArrayArray c = {.ndim=a.ndim, .size=a.size};
    c.shape = copy_of_ia(a.shape, a.ndim);

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
    struct SparseArrayArray c = {.ndim=a.ndim - 1};
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

    c.elements = malloc(c.size*sizeof(struct SparseArray));
    struct SparseArray zero = {.ndim=a.elements[0].ndim, .nitems=0};
    for (int i = 0; i < c.size; i ++)
        c.elements[i] = zero;

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
EXPORT double* sum_all_sparse(
        struct SparseArrayArray a) {
    // set up the ndarray
    double* values = calloc(a.size, sizeof(double));
    // then do the sum for each SparseArray
    for (int i = 0; i < a.size; i ++)
        for (int j = 0; j < a.elements[i].nitems; j ++)
            values[i] += a.elements[i].values[j];
    return values;
}

/**
 * sum along all of the dense axes of a SparseArrayArray
 */
EXPORT double* sum_all_dense(
        struct SparseArrayArray a, const int shape[]) {
    // calculate the size
    int size = product(shape, a.elements[0].ndim);

    // finally, histogram the values
    double* values = calloc(size, sizeof(double));
    for (int i = 0; i < a.size; i ++) {
        // for each element of each element
        for (int j = 0; j < a.elements[i].nitems; j ++) {
            int* index = a.elements[i].indices + j*a.elements[i].ndim;
            int l = 0;
            for (int k = 0; k < a.elements[i].ndim; k ++) {
                if (index[k] < 0 || index[k] >= shape[k]) {
                    printf("Error! a SparseArray had an index outside of the given shape (%d out of %d).\n", index[k], shape[k]);
                    return NULL;
                }
                l = l*shape[k] + index[k]; // find the index
            }
            values[l] += a.elements[i].values[j]; // and add it there
        }
    }

    return values;
}

/**
 * reflect the axes of this array and switch the dense axis for the sparse one
 */
EXPORT struct SparseArrayArray transpose(struct SparseArrayArray a, int sparse_size) {
    if (a.ndim != 1) {
        printf("Error! I only can transpose SparseArrayArrays with exactly 1 dense axis.");
        struct SparseArrayArray null = {.ndim=-1};
        return null;
    }

    // first, iterate thru the input array to count how many elements are in each collum
    int* nitems = calloc(sparse_size, sizeof(int));
    for (int i = 0; i < a.size; i ++) {
        struct SparseArray row = a.elements[i];
        if (row.ndim != 1) {
            printf("Error! I can only transpose SparseArrayArrays with exactly 1 sparse axis.");
            struct SparseArrayArray null = {.ndim=-1};
            free(nitems);
            return null;
        }
        for (int j = 0; j < row.nitems; j ++) {
            if (row.indices[j] < 0 || row.indices[j] >= sparse_size) {
                printf("Error! the sparse array had indices out of bounds of the given sparse_size.");
                struct SparseArrayArray null = {.ndim=-1};
                free(nitems);
                return null;
            }
            nitems[row.indices[j]] ++;
        }
    }

    // then bild each row of the transpose, allocating however much memory was deemd necessary
    struct SparseArrayArray c = {.ndim=1, .size=sparse_size};
    c.shape = malloc(sizeof(int));
    c.shape[0] = sparse_size;
    c.elements = malloc(sparse_size*sizeof(struct SparseArray));
    for (int i = 0; i < c.size; i ++) {
        struct SparseArray row = {.ndim=1, .nitems=nitems[i]};
        row.indices = malloc(nitems[i]*sizeof(int));
        row.values = malloc(nitems[i]*sizeof(double));
        c.elements[i] = row;
    }
    free(nitems);

    // finally, iterate thru the old array to transfer the values
    int* nfilled = calloc(sparse_size, sizeof(int));
    for (int i_a = 0; i_a < a.size; i_a ++) {
        for (int j = 0; j < a.elements[i_a].nitems; j ++) {
            int i_c = a.elements[i_a].indices[j];
            c.elements[i_c].indices[nfilled[i_c]] = i_a;
            c.elements[i_c].values[nfilled[i_c]] = a.elements[i_a].values[j];
            nfilled[i_c] ++;
        }
    }
    free(nfilled);

    return c;
}

/**
 * modify the shape of a SparseArrayArray by adding dimensions to the beginning of its
 * shape, without changing its size or altering its values.
 */
EXPORT struct SparseArrayArray expand_dims(
        struct SparseArrayArray a, int new_ndim) {
    struct SparseArrayArray c = {.ndim = a.ndim + new_ndim, .size=a.size};

    c.shape = malloc(c.ndim*sizeof(int));
    for (int k = 0; k < new_ndim; k ++)
        c.shape[k] = 1;
    for (int k = new_ndim; k < c.ndim; k ++)
        c.shape[k] = a.shape[k - new_ndim];

    c.elements = malloc(c.size*sizeof(struct SparseArray));
    for (int i = 0; i < c.size; i ++)
        c.elements[i] = copy_of_sa(a.elements[i]);

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
    struct SparseArrayArray c = {.ndim=a.ndim - 1};
    c.size = a.size / a.shape[axis];
    c.shape = malloc(c.ndim*sizeof(int));
    for (int k = 0; k < c.ndim; k ++) {
        if (k >= axis)
            c.shape[k] = a.shape[k + 1];
        else
            c.shape[k] = a.shape[k];
    }

    // reassine the indices
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

    return c;
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
    struct SparseArrayArray c = {.ndim=a.ndim};
    c.size = a.size / a.shape[axis] * length;
    c.shape = malloc(c.ndim*sizeof(int));
    for (int k = 0; k < c.ndim; k ++) {
        if (k == axis)
            c.shape[k] = length;
        else
            c.shape[k] = a.shape[k];
    }

    // reassine the indices
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

    return c;
}

/**
 * convert a SparseArrayArray to a plain dense ndarray (flattend)
 */
EXPORT double* to_dense(
        struct SparseArrayArray a, const int sparse_shape[]) {
    // first, you must determine the shape
    int total_ndim = a.ndim + a.elements[0].ndim;
    int* shape = malloc(total_ndim*sizeof(int));
    for (int k = 0; k < a.ndim; k ++)
        shape[k] = a.shape[k];
    for (int k = a.ndim; k < total_ndim; k ++)
        shape[k] = sparse_shape[k - a.ndim];

    // calculate the size
    int size = product(shape, total_ndim);

    // finally, set the values
    double* values = calloc(size, sizeof(double));
    for (int i = 0; i < a.size; i ++) {
        struct SparseArray element = a.elements[i];
        // for each element of each element
        for (int j = 0; j < element.nitems; j ++) {
            int* index = element.indices + j*element.ndim;
            int l = i;
            for (int k = a.ndim; k < total_ndim; k ++) {
                if (index[k - a.ndim] < 0 || index[k - a.ndim] >= shape[k]) {
                    printf("Error! a SparseArray had an index outside of the given shape (%d out of %d).\n", index[k - a.ndim], shape[k]);
                    return NULL;
                }
                l = l*shape[k] + index[k - a.ndim]; // find the index
            }
            values[l] += element.values[j]; // and add it there
        }
    }

    free(shape);

    return values;
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
