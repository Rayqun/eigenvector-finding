/*
 *  THE CONTENTS OF THIS FILE WILL BE OVERWRITTEN WHILE GRADING.
 *  MODIFY AT YOUR OWN RISK.
 */

#ifndef BENCHMARK_H
#define BENCHMARK_H

/** Uses a batch variant of power iteration to compute eigenvectors of A.
 *  Each column of U is used as a seed for its own power iteration, and the result
 *  is placed in the corresponding column of V. Both U and V are clobbered by this 
 *  function. A, U, and V are all column-major NxN matrices, and the power iteration 
 *  terminates after ITERS iterations. */
void eig_naive(float *v, float *A, float *u, size_t n, unsigned iters);

/* A (hopefully) faster version of eig_naive for you to implement. */
void eig(float *v, float *A, float *u, size_t n, unsigned iters);


#endif
