#include <string.h>
#include <math.h>
#include "benchmark.h"
#include <emmintrin.h>
#include <pmmintrin.h>
#include <stdio.h>
#include <nmmintrin.h>

 void print_m(__m128 input) {
 float* a = (float*) malloc(sizeof(float)*4);
 _mm_storeu_ps(a,input);
 printf("%.1f , %.1f , %.1f , %.1f \n",a[0],a[1],a[2],a[3]);
 
 }
 
 
void eig(float *v, float *A, float *u, size_t n, unsigned iters) {
float *b;
int fringe =  n%4;
float *B = (float*) malloc(n*n* sizeof(float));
    int i, j, k=0, blocksize = 64, even = (n/4)*4;

    
// transpose

// Set a[3] = A_22
int a;
for (i=0; i<n; i++) {

    for (a=0; a<n/blocksize; a++) { 
        for (j=a*blocksize; j<(a+1)*blocksize; j+=4) {
            b = B+k;
            _mm_storeu_ps(b, _mm_set_ps(A[(j+3)*n+i],A[(j+2)*n+i],A[(j+1)*n+i],A[j*n+i]));
            k+=4;
            }
        }
        if (n%blocksize) {
            while (j<even) {
            b = B+k;
            _mm_storeu_ps(b, _mm_set_ps(A[(j+3)*n+i],A[(j+2)*n+i],A[(j+1)*n+i],A[j*n+i]));
            j+=4;
            k+=4;
            }
            while (j<n) {
            B[k]= A[j*n+i];
            j++;
            k++;
            }
        }

    }



for (int iter=0; iter<iters; iter++) {


    __m128 b0, c1, c2, c3, c4, v_abs, c_final;
    
    
#pragma omp parallel for  private(c1) private(c2) private(c3) private(c4) private(j) private(b0) private(v_abs) private(c_final) 
    for (int col=0; col<n; col++) {
    float mu = 0;
    int row; float* addr;


    v_abs = _mm_setzero_ps();
    float*p = (float*) malloc(4*sizeof(float));
    

    for (row=0; row<even; row+=4) {
    c1 = _mm_setzero_ps();
    c2 = _mm_setzero_ps();
    c3 = _mm_setzero_ps();
    c4 = _mm_setzero_ps();
    //c_final = _mm_setzero_ps();
int b; j=0;
        while (j<even){ 
            addr = u+j+col*n;
            b0 = _mm_loadu_ps(addr); 
                    addr = B+j+row*n;
                    c1 = _mm_add_ps(c1, _mm_mul_ps(_mm_loadu_ps(addr),  b0));
                    addr += n;
                    c2 = _mm_add_ps(c2, _mm_mul_ps(_mm_loadu_ps(addr), b0));
                    addr +=n;
                    c3 = _mm_add_ps(c3, _mm_mul_ps(_mm_loadu_ps(addr), b0));
                    addr +=n;
                    c4 = _mm_add_ps(c4, _mm_mul_ps(_mm_loadu_ps(addr), b0));    
                    j+=4;
                }
                
                
                if (fringe!=0) {
                memset(p,0,4*sizeof(float));
                                while (j<n){    
                                p[0] +=B[row*n+j]*u[j+col*n];
                                p[1] +=B[(row+1)*n+j]*u[j+col*n];
                                p[2] +=B[(row+2)*n+j]*u[j+col*n];
                                p[3] +=B[(row+3)*n+j]*u[j+col*n];
                                j++;

                        }
                    c1 = _mm_add_ps(c1, _mm_set_ss(p[0]));
                    c2 = _mm_add_ps(c2, _mm_set_ss(p[1]));
                    c3 = _mm_add_ps(c3, _mm_set_ss(p[2]));
                    c4 = _mm_add_ps(c4, _mm_set_ss(p[3]));                  
                }
                c_final = _mm_hadd_ps( _mm_hadd_ps(c1,c2), _mm_hadd_ps(c3,c4)); 
                v_abs = _mm_add_ps(v_abs, _mm_mul_ps(c_final,c_final));
                addr = v+col*n+row;
                _mm_storeu_ps(addr,c_final);
                }
                
                                        
    if (fringe!=0) {        
    c1 = _mm_setzero_ps();
    c2 = _mm_setzero_ps();
    c3 = _mm_setzero_ps();
    c4 = _mm_setzero_ps();          
    //c_final = _mm_setzero_ps();
    
                    for (j= 0; j<even;  j+=4) {
            addr = u+j+n*col;
            b0 = _mm_loadu_ps(addr);
/*      Depending on the # of leftover rows, assign to the appropriate vector the sum of products*/ 
        switch (fringe) {
    
    case 3: 
    addr = B+j+(row+2)*n;
    c3 = _mm_add_ps(c3, _mm_mul_ps(_mm_loadu_ps(addr), b0));
    case 2: 
    addr = B+j+(row+1)*n;
    c2 = _mm_add_ps(c2, _mm_mul_ps(_mm_loadu_ps(addr), b0));
    case 1:
    addr = B+j+row*n;
    c1 = _mm_add_ps(c1, _mm_mul_ps(_mm_loadu_ps(addr),  b0));
                    
                        }
                    }
                    
        memset(p,0,4*sizeof(float));
        while (j<n) {

            switch (fringe) {
                    case 3: p[2] +=B[(row+2)*n+j]*u[j+col*n];
                    case 2:     p[1] +=B[(row+1)*n+j]*u[j+col*n];
                    case 1: p[0] +=B[row*n+j]*u[j+col*n];
                                }
                                j++;

                        }
                        
                        
                c1 = _mm_add_ps(c1, _mm_set_ss(p[0]));
                c2 = _mm_add_ps(c2, _mm_set_ss(p[1]));
                c3 = _mm_add_ps(c3, _mm_set_ss(p[2]));                  
                c_final = _mm_hadd_ps( _mm_hadd_ps(c1,c2), _mm_hadd_ps(c3,c4));
                v_abs = _mm_add_ps(v_abs, _mm_mul_ps(c_final,c_final));
                
                _mm_store_ps(p, c_final);
                {
                    switch(fringe) {
                case 3: v[col*n+row+2] =    p[2]; 
                case 2: v[col*n+row+1] =    p[1]; 
                case 1: v[col*n+row] = p[0]; 
                }
                
                }
            
                

                }

                
    _mm_storeu_ps(p, v_abs);
    mu = p[0]+p[1]+p[2]+p[3];
    mu = sqrt(mu);
        for (j=0; j<even; j+=4) {
                addr = v+ col*n+j;
                float* dst  = u+ col*n+j;
            _mm_storeu_ps(dst, _mm_div_ps(_mm_loadu_ps(addr),_mm_set_ps1(mu)));
        }
                
        while (j<n) {
            u[col*n+j]  = v[col*n+j]/mu;
            j++;
            }
        free(p);    
        
    
    
    
    
    
    
    }

    }
free(B);

}