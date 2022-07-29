/**
 * sparse.c
 *
 * the cytonic component of sparse.py
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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
__attribute__((unused)) __declspec(dllexport) void free_saa(
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
__attribute__((unused)) __declspec(dllexport) void free_nda(double* a) {
    free(a);
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
        struct SparseArray null = {};
        return null;
    }

    struct SparseArray c = {.ndim=a.ndim};

    if (a.nitems > 0 || b.nitems > 0) {
        // first, merge-sort the indices of a and b
        int a_mapping[a.nitems + b.nitems];
        int b_mapping[a.nitems + b.nitems];
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
                        struct SparseArray null = {};
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
    else { // on the off-chance these are both empty
        c.nitems = 0; // don't allocate any memory
    }

    return c;
}

struct SparseArray add_sa(struct SparseArray a, struct SparseArray b) {
    return elementwise_sa(ADD, a, b);
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
        struct SparseArrayArray null = {};
        return null;
    }

    struct SparseArrayArray c = {.ndim=a.ndim, .size=a.size};
    c.shape = copy_of_ia(a.shape, a.ndim);

    c.elements = malloc(c.size*sizeof(struct SparseArray));
    for (int i = 0; i < c.size; i ++)
        c.elements[i] = elementwise_sa(operator, a.elements[i], b.elements[i]);

    return c;
}

__attribute__((unused)) __declspec(dllexport) struct SparseArrayArray add_saa(
        struct SparseArrayArray a, struct SparseArrayArray b) {
    return elementwise_saa(ADD, a, b);
}

__attribute__((unused)) __declspec(dllexport) struct SparseArrayArray subtract_saa(
        struct SparseArrayArray a, struct SparseArrayArray b) {
    return elementwise_saa(SUBTRACT, a, b);
}

__attribute__((unused)) __declspec(dllexport) struct SparseArrayArray multiply_saa(
        struct SparseArrayArray a, struct SparseArrayArray b) {
    return elementwise_saa(MULTIPLY, a, b);
}

/**
 * create an empty SparseArrayArray
 * @param dense_shape
 * @param sparse_shape
 * @return
 */
__attribute__((unused)) __declspec(dllexport) struct SparseArrayArray zeros(
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
 */
__attribute__((unused)) __declspec(dllexport) struct SparseArrayArray identity(
        int ndim, const int shape[]) {
    struct SparseArrayArray a = {.ndim=ndim};
    a.shape = copy_of_ia(shape, ndim);
    a.size = product(shape, ndim);

    // keep track of the index we're on as we iterate thru the SparseArrayArray
    a.elements = malloc(a.size*sizeof(struct SparseArray));
    int* index = calloc(a.ndim, sizeof(int));
    for (int i = 0; i < a.size; i ++) {
        // create the single-value SparseArray
        struct SparseArray element = {.ndim=ndim, .nitems=1};
        element.indices = copy_of_ia(index, a.ndim);
        element.values = malloc(sizeof(double));
        element.values[0] = 1.;
        a.elements[i] = element;

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
 * create a SparseArrayArray where each element has a single 1 at an index matching its own index
 * @param ndim the number of dense dimensions and the number of sparse dimensions (half the total number of dimensions)
 * @param shape the shape of the SparseArrayArray, which is also the shape of each of its elements
 */
__attribute__((unused)) __declspec(dllexport) struct SparseArrayArray unit(
        int dense_ndim, const int dense_shape[], const int dense_index[],
        int sparse_ndim, const int sparse_index[], double value) {
    struct SparseArrayArray a = {.ndim=dense_ndim};
    a.shape = copy_of_ia(dense_shape, dense_ndim);
    a.size = product(dense_shape, dense_ndim);

    // find out which index we want to envalue
    int i_nonzero = 0;
    for (int k = 0; k < dense_ndim; k ++)
        i_nonzero = i_nonzero*dense_shape[k] + dense_index[k];

    // then bild the array
    a.elements = malloc(a.size*sizeof(struct SparseArray));
    for (int i = 0; i < a.size; i ++) {
        // create the single-value SparseArray
        struct SparseArray element = {.ndim=sparse_ndim};
        if (i == i_nonzero) {
            element.nitems = 1;
            element.indices = copy_of_ia(sparse_index, a.ndim);
            element.values = malloc(sizeof(double));
            *element.values = value;
        }
        else {
            element.nitems = 0;
            element.indices = NULL;
            element.values = NULL;
        }
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
            struct SparseArrayArray null = {};
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
                    struct SparseArrayArray null = {};
                    return null;
                }
            }
        }
        c.elements[i_a] = new;
    }
    return c;
}

__attribute__((unused)) __declspec(dllexport) struct SparseArrayArray multiply_nda(
        struct SparseArrayArray a, const double* b, const int shape[]) {
    return elementwise_nda(MULTIPLY, a, b, shape);
}

__attribute__((unused)) __declspec(dllexport) struct SparseArrayArray divide_nda(
        struct SparseArrayArray a, const double* b, const int shape[]) {
    return elementwise_nda(DIVIDE, a, b, shape);
}

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
                    struct SparseArrayArray null = {};
                    return null;
                }
            }
        }
        c.elements[i] = new;
    }
    return c;
}

__attribute__((unused)) __declspec(dllexport) struct SparseArrayArray multiply_f(
        struct SparseArrayArray a, double factor) {
    return elementwise_f(MULTIPLY, a, factor);
}

__attribute__((unused)) __declspec(dllexport) struct SparseArrayArray divide_f(
        struct SparseArrayArray a, double divisor) {
    return elementwise_f(DIVIDE, a, divisor);
}

__attribute__((unused)) __declspec(dllexport) struct SparseArrayArray power_f(
        struct SparseArrayArray a, double power) {
    return elementwise_f(POWER, a, power);
}

/**
 * sum a SparseArrayArray along one of the dense axes.
 * @param a the first array
 * @param axis the axis of the array along wihch to perform the sum
 * @return the resulting array
 */
__attribute__((unused)) __declspec(dllexport) struct SparseArrayArray sum_along_axis(
        struct SparseArrayArray a, int axis) {
    if (axis < 0 || axis >= a.ndim) {
        printf("Error! the specified axis (%d out of %d) does not exist.\n", axis, a.ndim);
        struct SparseArrayArray null = {};
        return null;
    }

    // establish the new number of dimensions
    struct SparseArrayArray c = {.ndim=a.ndim - 1};
    c.size = a.size/a.shape[axis];

    // set up the new shape
    int b_shape[a.ndim]; // a dummy shape that allows values to be broadcast from a to c
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

    return c;
}

/**
 * sum along all of the dense axes of a SparseArrayArray
 */
__attribute__((unused)) __declspec(dllexport) double* sum_all(
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
 * convert a SparseArrayArray to a plain dense ndarray (flattend)
 */
__attribute__((unused)) __declspec(dllexport) double* to_dense(
        struct SparseArrayArray a, const int sparse_shape[]) {
    // first, you must determine the shape
    int total_ndim = a.ndim + a.elements[0].ndim;
    int shape[total_ndim];
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

    return values;
}

/**
 * index the array on one axis, extracting a slice that is all of the elements where the
 * index on that axis matches what's given
 */
__attribute__((unused)) __declspec(dllexport) struct SparseArrayArray get_slice_saa(
        struct SparseArrayArray a, int index, int axis) {
    if (axis < 0 || axis >= a.ndim || index < 0 || index >= a.shape[axis]) {
        printf("Error! the specified slice (%d on axis %d out of %d) is out of bounds.\n", axis, a.ndim);
        struct SparseArrayArray null = {};
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
    int c_index[c.ndim];
    for (int k = 0; k < c.ndim; k ++)
        c_index[k] = 0;
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

    return c;
}

/**
 * rearrange a SparseArrayArray along a particular axis using a 1D int array as an index
 * @param a the array to be reindexed
 * @param index the array of indices at which to evaluate the given axis
 * @param length the number of values in index
 * @param axis the values of index represent indices along this axis
 */
__attribute__((unused)) __declspec(dllexport) struct SparseArrayArray get_reindex_saa(
        struct SparseArrayArray a, const int index[], int length, int axis) {
    if (axis < 0 || axis >= a.ndim) {
        printf("Error! the specified axis (%d out of %d) does not exist.\n", axis, a.ndim);
        struct SparseArrayArray null = {};
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
    int c_index[a.ndim];
    for (int k = 0; k < a.ndim; k ++)
        c_index[k] = 0;
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

    return c;
}
