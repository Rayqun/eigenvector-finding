/*
 *  THE CONTENTS OF THIS FILE WILL BE OVERWRITTEN WHILE GRADING.
 *  MODIFY AT YOUR OWN RISK.
 */
#include <string.h>
#include <math.h>
#include "benchmark.h"

void eig(float *v, float *A, float *u, size_t n, unsigned iters) {
    eig_naive(v, A, u, n, iters);
}
