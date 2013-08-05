#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <nmmintrin.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define BLOCK 64

float *a; // global variable for A matrix
float *b; // global variable for B matrix
float *c; // global variable for C matrix
int m_b, m_b4, m_b16, m_b32; // m_b# represents the closest multiple of # less than m_b
int n_b; // global variable for n_a

void thread_api_failure(void) {
    printf("Pthreads API function returned nonzero status!\n");
    exit(-1);
}

/**
 *  Simple data structure to pass into the functions how much of the B matrix
 *  to traverse.
 **/

struct thread_data {
    int j_start;
    int j_end;
};

typedef struct thread_data data_t;

/**
 *  Our threads code is essentially the same as in openmp, except we manually
 *  create threads when before we would have used openmp pragmas. Because our
 *  original openmp code has 2 sections that are individually parallelize as
 *  well as a "leftovers" section, we need three different functions to allow
 *  multiple threads to run on each section of the matrices.
 *
 *  The main_thread divides up the work from i = [0,m_b256], jumping by 32
 *  The fringe_thread handles cases from i = [m_b32, m_b16], jumping by 16
 *  the leftovers_thread handles cases from i = [m_b16, m_b], jumping by 1
 *
 *  Each thread takes approximately 1/16 of B and traverses through all of A.
 **/

void *sgemm_main_thread(void *thereadarg) {
  data_t *myData = (data_t *) thereadarg;
  int j_s = myData->j_start;
  int j_e = myData->j_end;

  __m128 zero = _mm_setzero_ps(); // variable for sanity.

  __m128 tempA1, tempA2, tempA3, tempA4;
  __m128 tempA5, tempA6, tempA7, tempA8;
  __m128 tempB1;
  __m128 tempC1i1, tempC2i1, tempC3i1, tempC4i1;
  __m128 tempC5i1, tempC6i1, tempC7i1, tempC8i1;
  __m128 tempC9i1, tempC10i1, tempC11i1, tempC12i1;
  __m128 tempC13i1, tempC14i1, tempC15i1, tempC16i1;
  __m128 tempC1i2, tempC2i2, tempC3i2, tempC4i2;
  __m128 tempC5i2, tempC6i2, tempC7i2, tempC8i2;
  __m128 tempC9i2, tempC10i2, tempC11i2, tempC12i2;
  __m128 tempC13i2, tempC14i2, tempC15i2, tempC16i2;
  __m128 tempC1i3, tempC2i3, tempC3i3, tempC4i3;
  __m128 tempC5i3, tempC6i3, tempC7i3, tempC8i3;
  __m128 tempC9i3, tempC10i3, tempC11i3, tempC12i3;
  __m128 tempC13i3, tempC14i3, tempC15i3, tempC16i3;
  __m128 tempC1i4, tempC2i4, tempC3i4, tempC4i4;
  __m128 tempC5i4, tempC6i4, tempC7i4, tempC8i4;
  __m128 tempC9i4, tempC10i4, tempC11i4, tempC12i4;
  __m128 tempC13i4, tempC14i4, tempC15i4, tempC16i4;
  __m128 tempC1i5, tempC2i5, tempC3i5, tempC4i5;
  __m128 tempC5i5, tempC6i5, tempC7i5, tempC8i5;
  __m128 tempC9i5, tempC10i5, tempC11i5, tempC12i5;
  __m128 tempC13i5, tempC14i5, tempC15i5, tempC16i5;
  __m128 tempC1i6, tempC2i6, tempC3i6, tempC4i6;
  __m128 tempC5i6, tempC6i6, tempC7i6, tempC8i6;
  __m128 tempC9i6, tempC10i6, tempC11i6, tempC12i6;
  __m128 tempC13i6, tempC14i6, tempC15i6, tempC16i6;
  __m128 tempC1i7, tempC2i7, tempC3i7, tempC4i7;
  __m128 tempC5i7, tempC6i7, tempC7i7, tempC8i7;
  __m128 tempC9i7, tempC10i7, tempC11i7, tempC12i7;
  __m128 tempC13i7, tempC14i7, tempC15i7, tempC16i7;
  __m128 tempC1i8, tempC2i8, tempC3i8, tempC4i8;
  __m128 tempC5i8, tempC6i8, tempC7i8, tempC8i8;
  __m128 tempC9i8, tempC10i8, tempC11i8, tempC12i8;
  __m128 tempC13i8, tempC14i8, tempC15i8, tempC16i8;

  int j;

  for( int y = 0; y < m_b; y += BLOCK) {
    int e = MIN(m_b32, y+BLOCK);
    for( j = j_s; j < j_e; j += 16 ) {
      float *bsel = b + j;
      int j1 = j*m_b;
      int j2 = j1 + m_b;
      int j3 = j2 + m_b;
      int j4 = j3 + m_b;
      int j5 = j4 + m_b;
      int j6 = j5 + m_b;
      int j7 = j6 + m_b;
      int j8 = j7 + m_b;
      int j9 = j8 + m_b;
      int j10 = j9 + m_b;
      int j11 = j10 + m_b;
      int j12 = j11 + m_b;
      int j13 = j12 + m_b;
      int j14 = j13 + m_b;
      int j15 = j14 + m_b;
      int j16 = j15 + m_b;
      for( int i = y; i < e; i += 32 ) {
        float *ai = a + i;
        float *ai4 = ai + 4;
        float *ai8 = ai4 + 4;
        float *ai12 = ai8 + 4;
        float *ai16 = ai12 + 4;
        float *ai20 = ai16 + 4;
        float *ai24 = ai20 + 4;
        float *ai28 = ai24 + 4;

        tempC1i1 = zero; // C values begin at 0 anways, so there is no use in loading
        tempC2i1 = zero; // from C.
        tempC3i1 = zero;
        tempC4i1 = zero;
        tempC5i1 = zero;
        tempC6i1 = zero;
        tempC7i1 = zero;
        tempC8i1 = zero;
        tempC9i1 = zero;
        tempC10i1 = zero;
        tempC11i1 = zero;
        tempC12i1 = zero;
        tempC13i1 = zero;
        tempC14i1 = zero;
        tempC15i1 = zero;
        tempC16i1 = zero;

        tempC1i2 = zero;
        tempC2i2 = zero;
        tempC3i2 = zero;
        tempC4i2 = zero;
        tempC5i2 = zero;
        tempC6i2 = zero;
        tempC7i2 = zero;
        tempC8i2 = zero;
        tempC9i2 = zero;
        tempC10i2 = zero;
        tempC11i2 = zero;
        tempC12i2 = zero;
        tempC13i2 = zero;
        tempC14i2 = zero;
        tempC15i2 = zero;
        tempC16i2 = zero;

        tempC1i3 = zero;
        tempC2i3 = zero;
        tempC3i3 = zero;
        tempC4i3 = zero;
        tempC5i3 = zero;
        tempC6i3 = zero;
        tempC7i3 = zero;
        tempC8i3 = zero;
        tempC9i3 = zero;
        tempC10i3 = zero;
        tempC11i3 = zero;
        tempC12i3 = zero;
        tempC13i3 = zero;
        tempC14i3 = zero;
        tempC15i3 = zero;
        tempC16i3 = zero;

        tempC1i4 = zero;
        tempC2i4 = zero;
        tempC3i4 = zero;
        tempC4i4 = zero;
        tempC5i4 = zero;
        tempC6i4 = zero;
        tempC7i4 = zero;
        tempC8i4 = zero;
        tempC9i4 = zero;
        tempC10i4 = zero;
        tempC11i4 = zero;
        tempC12i4 = zero;
        tempC13i4 = zero;
        tempC14i4 = zero;
        tempC15i4 = zero;
        tempC16i4 = zero;

        tempC1i5 = zero;
        tempC2i5 = zero;
        tempC3i5 = zero;
        tempC4i5 = zero;
        tempC5i5 = zero;
        tempC6i5 = zero;
        tempC7i5 = zero;
        tempC8i5 = zero;
        tempC9i5 = zero;
        tempC10i5 = zero;
        tempC11i5 = zero;
        tempC12i5 = zero;
        tempC13i5 = zero;
        tempC14i5 = zero;
        tempC15i5 = zero;
        tempC16i5 = zero;

        tempC1i6 = zero;
        tempC2i6 = zero;
        tempC3i6 = zero;
        tempC4i6 = zero;
        tempC5i6 = zero;
        tempC6i6 = zero;
        tempC7i6 = zero;
        tempC8i6 = zero;
        tempC9i6 = zero;
        tempC10i6 = zero;
        tempC11i6 = zero;
        tempC12i6 = zero;
        tempC13i6 = zero;
        tempC14i6 = zero;
        tempC15i6 = zero;
        tempC16i6 = zero;

        tempC1i7 = zero;
        tempC2i7 = zero;
        tempC3i7 = zero;
        tempC4i7 = zero;
        tempC5i7 = zero;
        tempC6i7 = zero;
        tempC7i7 = zero;
        tempC8i7 = zero;
        tempC9i7 = zero;
        tempC10i7 = zero;
        tempC11i7 = zero;
        tempC12i7 = zero;
        tempC13i7 = zero;
        tempC14i7 = zero;
        tempC15i7 = zero;
        tempC16i7 = zero;

        tempC1i8 = zero;
        tempC2i8 = zero;
        tempC3i8 = zero;
        tempC4i8 = zero;
        tempC5i8 = zero;
        tempC6i8 = zero;
        tempC7i8 = zero;
        tempC8i8 = zero;
        tempC9i8 = zero;
        tempC10i8 = zero;
        tempC11i8 = zero;
        tempC12i8 = zero;
        tempC13i8 = zero;
        tempC14i8 = zero;
        tempC15i8 = zero;
        tempC16i8 = zero;

        for( int k = 0; k < n_b; k++ ) {
          int k0 = k*m_b ;

          tempA1 = _mm_loadu_ps(ai+k0);
          tempA2 = _mm_loadu_ps(ai4+k0);
          tempA3 = _mm_loadu_ps(ai8+k0);
          tempA4 = _mm_loadu_ps(ai12+k0);
          tempA5 = _mm_loadu_ps(ai16+k0);
          tempA6 = _mm_loadu_ps(ai20+k0);
          tempA7 = _mm_loadu_ps(ai24+k0);
          tempA8 = _mm_loadu_ps(ai28+k0);

          tempB1 = _mm_load1_ps(bsel+k0);

          tempC1i1 += _mm_mul_ps(tempA1, tempB1);
          tempC1i2 += _mm_mul_ps(tempA2, tempB1);
          tempC1i3 += _mm_mul_ps(tempA3, tempB1);
          tempC1i4 += _mm_mul_ps(tempA4, tempB1);
          tempC1i5 += _mm_mul_ps(tempA5, tempB1);
          tempC1i6 += _mm_mul_ps(tempA6, tempB1);
          tempC1i7 += _mm_mul_ps(tempA7, tempB1);
          tempC1i8 += _mm_mul_ps(tempA8, tempB1);

          tempB1 = _mm_load1_ps(bsel+1+k0);

          tempC2i1 += _mm_mul_ps(tempA1, tempB1);
          tempC2i2 += _mm_mul_ps(tempA2, tempB1);
          tempC2i3 += _mm_mul_ps(tempA3, tempB1);
          tempC2i4 += _mm_mul_ps(tempA4, tempB1);
          tempC2i5 += _mm_mul_ps(tempA5, tempB1);
          tempC2i6 += _mm_mul_ps(tempA6, tempB1);
          tempC2i7 += _mm_mul_ps(tempA7, tempB1);
          tempC2i8 += _mm_mul_ps(tempA8, tempB1);

          tempB1 = _mm_load1_ps(bsel+2+k0); 

          tempC3i1 += _mm_mul_ps(tempA1, tempB1);
          tempC3i2 += _mm_mul_ps(tempA2, tempB1);
          tempC3i3 += _mm_mul_ps(tempA3, tempB1);
          tempC3i4 += _mm_mul_ps(tempA4, tempB1);
          tempC3i5 += _mm_mul_ps(tempA5, tempB1);
          tempC3i6 += _mm_mul_ps(tempA6, tempB1);
          tempC3i7 += _mm_mul_ps(tempA7, tempB1);
          tempC3i8 += _mm_mul_ps(tempA8, tempB1);

          tempB1 = _mm_load1_ps(bsel+3+k0);

          tempC4i1 += _mm_mul_ps(tempA1, tempB1);
          tempC4i2 += _mm_mul_ps(tempA2, tempB1);
          tempC4i3 += _mm_mul_ps(tempA3, tempB1);
          tempC4i4 += _mm_mul_ps(tempA4, tempB1);
          tempC4i5 += _mm_mul_ps(tempA5, tempB1);
          tempC4i6 += _mm_mul_ps(tempA6, tempB1);
          tempC4i7 += _mm_mul_ps(tempA7, tempB1);
          tempC4i8 += _mm_mul_ps(tempA8, tempB1);

          tempB1 = _mm_load1_ps(bsel+4+k0);

          tempC5i1 += _mm_mul_ps(tempA1, tempB1);
          tempC5i2 += _mm_mul_ps(tempA2, tempB1);
          tempC5i3 += _mm_mul_ps(tempA3, tempB1);
          tempC5i4 += _mm_mul_ps(tempA4, tempB1);
          tempC5i5 += _mm_mul_ps(tempA5, tempB1);
          tempC5i6 += _mm_mul_ps(tempA6, tempB1);
          tempC5i7 += _mm_mul_ps(tempA7, tempB1);
          tempC5i8 += _mm_mul_ps(tempA8, tempB1);

          tempB1 = _mm_load1_ps(bsel+5+k0);

          tempC6i1 += _mm_mul_ps(tempA1, tempB1);
          tempC6i2 += _mm_mul_ps(tempA2, tempB1);
          tempC6i3 += _mm_mul_ps(tempA3, tempB1);
          tempC6i4 += _mm_mul_ps(tempA4, tempB1);
          tempC6i5 += _mm_mul_ps(tempA5, tempB1);
          tempC6i6 += _mm_mul_ps(tempA6, tempB1);
          tempC6i7 += _mm_mul_ps(tempA7, tempB1);
          tempC6i8 += _mm_mul_ps(tempA8, tempB1);

          tempB1 = _mm_load1_ps(bsel+6+k0);

          tempC7i1 += _mm_mul_ps(tempA1, tempB1);
          tempC7i2 += _mm_mul_ps(tempA2, tempB1);
          tempC7i3 += _mm_mul_ps(tempA3, tempB1);
          tempC7i4 += _mm_mul_ps(tempA4, tempB1);
          tempC7i5 += _mm_mul_ps(tempA5, tempB1);
          tempC7i6 += _mm_mul_ps(tempA6, tempB1);
          tempC7i7 += _mm_mul_ps(tempA7, tempB1);
          tempC7i8 += _mm_mul_ps(tempA8, tempB1);

          tempB1 = _mm_load1_ps(bsel+7+k0);

          tempC8i1 += _mm_mul_ps(tempA1, tempB1);
          tempC8i2 += _mm_mul_ps(tempA2, tempB1);
          tempC8i3 += _mm_mul_ps(tempA3, tempB1);
          tempC8i4 += _mm_mul_ps(tempA4, tempB1);
          tempC8i5 += _mm_mul_ps(tempA5, tempB1);
          tempC8i6 += _mm_mul_ps(tempA6, tempB1);
          tempC8i7 += _mm_mul_ps(tempA7, tempB1);
          tempC8i8 += _mm_mul_ps(tempA8, tempB1);

          tempB1 = _mm_load1_ps(bsel+8+k0);

          tempC9i1 += _mm_mul_ps(tempA1, tempB1);
          tempC9i2 += _mm_mul_ps(tempA2, tempB1);
          tempC9i3 += _mm_mul_ps(tempA3, tempB1);
          tempC9i4 += _mm_mul_ps(tempA4, tempB1);
          tempC9i5 += _mm_mul_ps(tempA5, tempB1);
          tempC9i6 += _mm_mul_ps(tempA6, tempB1);
          tempC9i7 += _mm_mul_ps(tempA7, tempB1);
          tempC9i8 += _mm_mul_ps(tempA8, tempB1);

          tempB1 = _mm_load1_ps(bsel+9+k0);

          tempC10i1 += _mm_mul_ps(tempA1, tempB1);
          tempC10i2 += _mm_mul_ps(tempA2, tempB1);
          tempC10i3 += _mm_mul_ps(tempA3, tempB1);
          tempC10i4 += _mm_mul_ps(tempA4, tempB1);
          tempC10i5 += _mm_mul_ps(tempA5, tempB1);
          tempC10i6 += _mm_mul_ps(tempA6, tempB1);
          tempC10i7 += _mm_mul_ps(tempA7, tempB1);
          tempC10i8 += _mm_mul_ps(tempA8, tempB1);

          tempB1 = _mm_load1_ps(bsel+10+k0);

          tempC11i1 += _mm_mul_ps(tempA1, tempB1);
          tempC11i2 += _mm_mul_ps(tempA2, tempB1);
          tempC11i3 += _mm_mul_ps(tempA3, tempB1);
          tempC11i4 += _mm_mul_ps(tempA4, tempB1);
          tempC11i5 += _mm_mul_ps(tempA5, tempB1);
          tempC11i6 += _mm_mul_ps(tempA6, tempB1);
          tempC11i7 += _mm_mul_ps(tempA7, tempB1);
          tempC11i8 += _mm_mul_ps(tempA8, tempB1);

          tempB1 = _mm_load1_ps(bsel+11+k0);

          tempC12i1 += _mm_mul_ps(tempA1, tempB1);
          tempC12i2 += _mm_mul_ps(tempA2, tempB1);
          tempC12i3 += _mm_mul_ps(tempA3, tempB1);
          tempC12i4 += _mm_mul_ps(tempA4, tempB1);
          tempC12i5 += _mm_mul_ps(tempA5, tempB1);
          tempC12i6 += _mm_mul_ps(tempA6, tempB1);
          tempC12i7 += _mm_mul_ps(tempA7, tempB1);
          tempC12i8 += _mm_mul_ps(tempA8, tempB1);

          tempB1 = _mm_load1_ps(bsel+12+k0);

          tempC13i1 += _mm_mul_ps(tempA1, tempB1);
          tempC13i2 += _mm_mul_ps(tempA2, tempB1);
          tempC13i3 += _mm_mul_ps(tempA3, tempB1);
          tempC13i4 += _mm_mul_ps(tempA4, tempB1);
          tempC13i5 += _mm_mul_ps(tempA5, tempB1);
          tempC13i6 += _mm_mul_ps(tempA6, tempB1);
          tempC13i7 += _mm_mul_ps(tempA7, tempB1);
          tempC13i8 += _mm_mul_ps(tempA8, tempB1);

          tempB1 = _mm_load1_ps(bsel+13+k0);

          tempC14i1 += _mm_mul_ps(tempA1, tempB1);
          tempC14i2 += _mm_mul_ps(tempA2, tempB1);
          tempC14i3 += _mm_mul_ps(tempA3, tempB1);
          tempC14i4 += _mm_mul_ps(tempA4, tempB1);
          tempC14i5 += _mm_mul_ps(tempA5, tempB1);
          tempC14i6 += _mm_mul_ps(tempA6, tempB1);
          tempC14i7 += _mm_mul_ps(tempA7, tempB1);
          tempC14i8 += _mm_mul_ps(tempA8, tempB1);

          tempB1 = _mm_load1_ps(bsel+14+k0);

          tempC15i1 += _mm_mul_ps(tempA1, tempB1);
          tempC15i2 += _mm_mul_ps(tempA2, tempB1);
          tempC15i3 += _mm_mul_ps(tempA3, tempB1);
          tempC15i4 += _mm_mul_ps(tempA4, tempB1);
          tempC15i5 += _mm_mul_ps(tempA5, tempB1);
          tempC15i6 += _mm_mul_ps(tempA6, tempB1);
          tempC15i7 += _mm_mul_ps(tempA7, tempB1);
          tempC15i8 += _mm_mul_ps(tempA8, tempB1);

          tempB1 = _mm_load1_ps(bsel+15+k0);

          tempC16i1 += _mm_mul_ps(tempA1, tempB1);
          tempC16i2 += _mm_mul_ps(tempA2, tempB1);
          tempC16i3 += _mm_mul_ps(tempA3, tempB1);
          tempC16i4 += _mm_mul_ps(tempA4, tempB1);
          tempC16i5 += _mm_mul_ps(tempA5, tempB1);
          tempC16i6 += _mm_mul_ps(tempA6, tempB1);
          tempC16i7 += _mm_mul_ps(tempA7, tempB1);
          tempC16i8 += _mm_mul_ps(tempA8, tempB1);
        }

        float *ci = c + i;

        _mm_storeu_ps(ci+j1, tempC1i1);
        _mm_storeu_ps(ci+j2, tempC2i1);
        _mm_storeu_ps(ci+j3, tempC3i1);
        _mm_storeu_ps(ci+j4, tempC4i1);
        _mm_storeu_ps(ci+j5, tempC5i1);
        _mm_storeu_ps(ci+j6, tempC6i1);
        _mm_storeu_ps(ci+j7, tempC7i1);
        _mm_storeu_ps(ci+j8, tempC8i1);
        _mm_storeu_ps(ci+j9, tempC9i1);
        _mm_storeu_ps(ci+j10, tempC10i1);
        _mm_storeu_ps(ci+j11, tempC11i1);
        _mm_storeu_ps(ci+j12, tempC12i1);
        _mm_storeu_ps(ci+j13, tempC13i1);
        _mm_storeu_ps(ci+j14, tempC14i1);
        _mm_storeu_ps(ci+j15, tempC15i1);
        _mm_storeu_ps(ci+j16, tempC16i1);


        ci += 4;

        _mm_storeu_ps(ci+j1, tempC1i2);
        _mm_storeu_ps(ci+j2, tempC2i2);
        _mm_storeu_ps(ci+j3, tempC3i2);
        _mm_storeu_ps(ci+j4, tempC4i2);
        _mm_storeu_ps(ci+j5, tempC5i2);
        _mm_storeu_ps(ci+j6, tempC6i2);
        _mm_storeu_ps(ci+j7, tempC7i2);
        _mm_storeu_ps(ci+j8, tempC8i2);
        _mm_storeu_ps(ci+j9, tempC9i2);
        _mm_storeu_ps(ci+j10, tempC10i2);
        _mm_storeu_ps(ci+j11, tempC11i2);
        _mm_storeu_ps(ci+j12, tempC12i2);
        _mm_storeu_ps(ci+j13, tempC13i2);
        _mm_storeu_ps(ci+j14, tempC14i2);
        _mm_storeu_ps(ci+j15, tempC15i2);
        _mm_storeu_ps(ci+j16, tempC16i2);

        ci += 4;

        _mm_storeu_ps(ci+j1, tempC1i3);
        _mm_storeu_ps(ci+j2, tempC2i3);
        _mm_storeu_ps(ci+j3, tempC3i3);
        _mm_storeu_ps(ci+j4, tempC4i3);
        _mm_storeu_ps(ci+j5, tempC5i3);
        _mm_storeu_ps(ci+j6, tempC6i3);
        _mm_storeu_ps(ci+j7, tempC7i3);
        _mm_storeu_ps(ci+j8, tempC8i3);
        _mm_storeu_ps(ci+j9, tempC9i3);
        _mm_storeu_ps(ci+j10, tempC10i3);
        _mm_storeu_ps(ci+j11, tempC11i3);
        _mm_storeu_ps(ci+j12, tempC12i3);
        _mm_storeu_ps(ci+j13, tempC13i3);
        _mm_storeu_ps(ci+j14, tempC14i3);
        _mm_storeu_ps(ci+j15, tempC15i3);
        _mm_storeu_ps(ci+j16, tempC16i3);

        ci += 4;

        _mm_storeu_ps(ci+j1, tempC1i4);
        _mm_storeu_ps(ci+j2, tempC2i4);
        _mm_storeu_ps(ci+j3, tempC3i4);
        _mm_storeu_ps(ci+j4, tempC4i4);
        _mm_storeu_ps(ci+j5, tempC5i4);
        _mm_storeu_ps(ci+j6, tempC6i4);
        _mm_storeu_ps(ci+j7, tempC7i4);
        _mm_storeu_ps(ci+j8, tempC8i4);
        _mm_storeu_ps(ci+j9, tempC9i4);
        _mm_storeu_ps(ci+j10, tempC10i4);
        _mm_storeu_ps(ci+j11, tempC11i4);
        _mm_storeu_ps(ci+j12, tempC12i4);
        _mm_storeu_ps(ci+j13, tempC13i4);
        _mm_storeu_ps(ci+j14, tempC14i4);
        _mm_storeu_ps(ci+j15, tempC15i4);
        _mm_storeu_ps(ci+j16, tempC16i4);

        ci += 4;

        _mm_storeu_ps(ci+j1, tempC1i5);
        _mm_storeu_ps(ci+j2, tempC2i5);
        _mm_storeu_ps(ci+j3, tempC3i5);
        _mm_storeu_ps(ci+j4, tempC4i5);
        _mm_storeu_ps(ci+j5, tempC5i5);
        _mm_storeu_ps(ci+j6, tempC6i5);
        _mm_storeu_ps(ci+j7, tempC7i5);
        _mm_storeu_ps(ci+j8, tempC8i5);
        _mm_storeu_ps(ci+j9, tempC9i5);
        _mm_storeu_ps(ci+j10, tempC10i5);
        _mm_storeu_ps(ci+j11, tempC11i5);
        _mm_storeu_ps(ci+j12, tempC12i5);
        _mm_storeu_ps(ci+j13, tempC13i5);
        _mm_storeu_ps(ci+j14, tempC14i5);
        _mm_storeu_ps(ci+j15, tempC15i5);
        _mm_storeu_ps(ci+j16, tempC16i5);

        ci += 4;

        _mm_storeu_ps(ci+j1, tempC1i6);
        _mm_storeu_ps(ci+j2, tempC2i6);
        _mm_storeu_ps(ci+j3, tempC3i6);
        _mm_storeu_ps(ci+j4, tempC4i6);
        _mm_storeu_ps(ci+j5, tempC5i6);
        _mm_storeu_ps(ci+j6, tempC6i6);
        _mm_storeu_ps(ci+j7, tempC7i6);
        _mm_storeu_ps(ci+j8, tempC8i6);
        _mm_storeu_ps(ci+j9, tempC9i6);
        _mm_storeu_ps(ci+j10, tempC10i6);
        _mm_storeu_ps(ci+j11, tempC11i6);
        _mm_storeu_ps(ci+j12, tempC12i6);
        _mm_storeu_ps(ci+j13, tempC13i6);
        _mm_storeu_ps(ci+j14, tempC14i6);
        _mm_storeu_ps(ci+j15, tempC15i6);
        _mm_storeu_ps(ci+j16, tempC16i6);

        ci += 4;

        _mm_storeu_ps(ci+j1, tempC1i7);
        _mm_storeu_ps(ci+j2, tempC2i7);
        _mm_storeu_ps(ci+j3, tempC3i7);
        _mm_storeu_ps(ci+j4, tempC4i7);
        _mm_storeu_ps(ci+j5, tempC5i7);
        _mm_storeu_ps(ci+j6, tempC6i7);
        _mm_storeu_ps(ci+j7, tempC7i7);
        _mm_storeu_ps(ci+j8, tempC8i7);
        _mm_storeu_ps(ci+j9, tempC9i7);
        _mm_storeu_ps(ci+j10, tempC10i7);
        _mm_storeu_ps(ci+j11, tempC11i7);
        _mm_storeu_ps(ci+j12, tempC12i7);
        _mm_storeu_ps(ci+j13, tempC13i7);
        _mm_storeu_ps(ci+j14, tempC14i7);
        _mm_storeu_ps(ci+j15, tempC15i7);
        _mm_storeu_ps(ci+j16, tempC16i7);

        ci += 4;

        _mm_storeu_ps(ci+j1, tempC1i8);
        _mm_storeu_ps(ci+j2, tempC2i8);
        _mm_storeu_ps(ci+j3, tempC3i8);
        _mm_storeu_ps(ci+j4, tempC4i8);
        _mm_storeu_ps(ci+j5, tempC5i8);
        _mm_storeu_ps(ci+j6, tempC6i8);
        _mm_storeu_ps(ci+j7, tempC7i8);
        _mm_storeu_ps(ci+j8, tempC8i8);
        _mm_storeu_ps(ci+j9, tempC9i8);
        _mm_storeu_ps(ci+j10, tempC10i8);
        _mm_storeu_ps(ci+j11, tempC11i8);
        _mm_storeu_ps(ci+j12, tempC12i8);
        _mm_storeu_ps(ci+j13, tempC13i8);
        _mm_storeu_ps(ci+j14, tempC14i8);
        _mm_storeu_ps(ci+j15, tempC15i8);
        _mm_storeu_ps(ci+j16, tempC16i8);
      }
    }
    if(j == m_b16) {
      for(int j = m_b16; j < m_b4; j += 4) {
        float *bsel = b + j;
        int j1 = j*m_b;
        int j2 = j1 + m_b;
        int j3 = j2 + m_b;
        int j4 = j3 + m_b;
        for( int i = y; i < e; i += 32 ) {
          float *ai = a + i;
          float *ai4 = ai + 4;
          float *ai8 = ai4 + 4;
          float *ai12 = ai8 + 4;
          float *ai16 = ai12 + 4;
          float *ai20 = ai16 + 4;
          float *ai24 = ai20 + 4;
          float *ai28 = ai24 + 4;

          tempC1i1 = zero; // C values begin at 0 anways, so there is no use in loading from C.
          tempC2i1 = zero;
          tempC3i1 = zero;
          tempC4i1 = zero;

          tempC1i2 = zero;
          tempC2i2 = zero;
          tempC3i2 = zero;
          tempC4i2 = zero;

          tempC1i3 = zero;
          tempC2i3 = zero;
          tempC3i3 = zero;
          tempC4i3 = zero;

          tempC1i4 = zero;
          tempC2i4 = zero;
          tempC3i4 = zero;
          tempC4i4 = zero;

          tempC1i5 = zero;
          tempC2i5 = zero;
          tempC3i5 = zero;
          tempC4i5 = zero;

          tempC1i6 = zero;
          tempC2i6 = zero;
          tempC3i6 = zero;
          tempC4i6 = zero;

          tempC1i7 = zero;
          tempC2i7 = zero;
          tempC3i7 = zero;
          tempC4i7 = zero;

          tempC1i8 = zero;
          tempC2i8 = zero;
          tempC3i8 = zero;
          tempC4i8 = zero;

          for( int k = 0; k < n_b; k++ ) {
            int k0 = k*m_b ;

            tempA1 = _mm_loadu_ps(ai+k0);
            tempA2 = _mm_loadu_ps(ai4+k0);
            tempA3 = _mm_loadu_ps(ai8+k0);
            tempA4 = _mm_loadu_ps(ai12+k0);
            tempA5 = _mm_loadu_ps(ai16+k0);
            tempA6 = _mm_loadu_ps(ai20+k0);
            tempA7 = _mm_loadu_ps(ai24+k0);
            tempA8 = _mm_loadu_ps(ai28+k0);

            tempB1 = _mm_load1_ps(bsel+k0);

            tempC1i1 += _mm_mul_ps(tempA1, tempB1);
            tempC1i2 += _mm_mul_ps(tempA2, tempB1);
            tempC1i3 += _mm_mul_ps(tempA3, tempB1);
            tempC1i4 += _mm_mul_ps(tempA4, tempB1);
            tempC1i5 += _mm_mul_ps(tempA5, tempB1);
            tempC1i6 += _mm_mul_ps(tempA6, tempB1);
            tempC1i7 += _mm_mul_ps(tempA7, tempB1);
            tempC1i8 += _mm_mul_ps(tempA8, tempB1);

            tempB1 = _mm_load1_ps(bsel+1+k0); 

            tempC2i1 += _mm_mul_ps(tempA1, tempB1);
            tempC2i2 += _mm_mul_ps(tempA2, tempB1);
            tempC2i3 += _mm_mul_ps(tempA3, tempB1);
            tempC2i4 += _mm_mul_ps(tempA4, tempB1);
            tempC2i5 += _mm_mul_ps(tempA5, tempB1);
            tempC2i6 += _mm_mul_ps(tempA6, tempB1);
            tempC2i7 += _mm_mul_ps(tempA7, tempB1);
            tempC2i8 += _mm_mul_ps(tempA8, tempB1);

            tempB1 = _mm_load1_ps(bsel+2+k0); 

            tempC3i1 += _mm_mul_ps(tempA1, tempB1);
            tempC3i2 += _mm_mul_ps(tempA2, tempB1);
            tempC3i3 += _mm_mul_ps(tempA3, tempB1);
            tempC3i4 += _mm_mul_ps(tempA4, tempB1);
            tempC3i5 += _mm_mul_ps(tempA5, tempB1);
            tempC3i6 += _mm_mul_ps(tempA6, tempB1);
            tempC3i7 += _mm_mul_ps(tempA7, tempB1);
            tempC3i8 += _mm_mul_ps(tempA8, tempB1);

            tempB1 = _mm_load1_ps(bsel+3+k0); 

            tempC4i1 += _mm_mul_ps(tempA1, tempB1);
            tempC4i2 += _mm_mul_ps(tempA2, tempB1);
            tempC4i3 += _mm_mul_ps(tempA3, tempB1);
            tempC4i4 += _mm_mul_ps(tempA4, tempB1);
            tempC4i5 += _mm_mul_ps(tempA5, tempB1);
            tempC4i6 += _mm_mul_ps(tempA6, tempB1);
            tempC4i7 += _mm_mul_ps(tempA7, tempB1);
            tempC4i8 += _mm_mul_ps(tempA8, tempB1);
          }

          float *ci = c + i;

          _mm_storeu_ps(ci+j1, tempC1i1);
          _mm_storeu_ps(ci+j2, tempC2i1);
          _mm_storeu_ps(ci+j3, tempC3i1);
          _mm_storeu_ps(ci+j4, tempC4i1);

          ci += 4;

          _mm_storeu_ps(ci+j1, tempC1i2);
          _mm_storeu_ps(ci+j2, tempC2i2);
          _mm_storeu_ps(ci+j3, tempC3i2);
          _mm_storeu_ps(ci+j4, tempC4i2);

          ci += 4;

          _mm_storeu_ps(ci+j1, tempC1i3);
          _mm_storeu_ps(ci+j2, tempC2i3);
          _mm_storeu_ps(ci+j3, tempC3i3);
          _mm_storeu_ps(ci+j4, tempC4i3);

          ci += 4;

          _mm_storeu_ps(ci+j1, tempC1i4);
          _mm_storeu_ps(ci+j2, tempC2i4);
          _mm_storeu_ps(ci+j3, tempC3i4);
          _mm_storeu_ps(ci+j4, tempC4i4);

          ci += 4;

          _mm_storeu_ps(ci+j1, tempC1i5);
          _mm_storeu_ps(ci+j2, tempC2i5);
          _mm_storeu_ps(ci+j3, tempC3i5);
          _mm_storeu_ps(ci+j4, tempC4i5);

          ci += 4;

          _mm_storeu_ps(ci+j1, tempC1i6);
          _mm_storeu_ps(ci+j2, tempC2i6);
          _mm_storeu_ps(ci+j3, tempC3i6);
          _mm_storeu_ps(ci+j4, tempC4i6);

          ci += 4;

          _mm_storeu_ps(ci+j1, tempC1i7);
          _mm_storeu_ps(ci+j2, tempC2i7);
          _mm_storeu_ps(ci+j3, tempC3i7);
          _mm_storeu_ps(ci+j4, tempC4i7);

          ci += 4;

          _mm_storeu_ps(ci+j1, tempC1i8);
          _mm_storeu_ps(ci+j2, tempC2i8);
          _mm_storeu_ps(ci+j3, tempC3i8);
          _mm_storeu_ps(ci+j4, tempC4i8);
        }
      }

      for(int j = m_b4; j < m_b; j++) {
        float *bsel = b + j;
        int j1 = j*m_b;
        for( int i = y; i < e; i += 32 ) {
          float *ai = a + i;
          float *ai4 = ai + 4;
          float *ai8 = ai4 + 4;
          float *ai12 = ai8 + 4;
          float *ai16 = ai12 + 4;
          float *ai20 = ai16 + 4;
          float *ai24 = ai20 + 4;
          float *ai28 = ai24 + 4;

          tempC1i1 = zero; // C values begin at 0 anways, so there is no use in loading from C.

          tempC1i2 = zero;

          tempC1i3 = zero;

          tempC1i4 = zero;

          tempC1i5 = zero;

          tempC1i6 = zero;

          tempC1i7 = zero;

          tempC1i8 = zero;

          for( int k = 0; k < n_b; k++ ) {
            int k0 = k*m_b ;

            tempA1 = _mm_loadu_ps(ai+k0);
            tempA2 = _mm_loadu_ps(ai4+k0);
            tempA3 = _mm_loadu_ps(ai8+k0);
            tempA4 = _mm_loadu_ps(ai12+k0);
            tempA5 = _mm_loadu_ps(ai16+k0);
            tempA6 = _mm_loadu_ps(ai20+k0);
            tempA7 = _mm_loadu_ps(ai24+k0);
            tempA8 = _mm_loadu_ps(ai28+k0);

            tempB1 = _mm_load1_ps(bsel+k0);

            tempC1i1 += _mm_mul_ps(tempA1, tempB1);
            tempC1i2 += _mm_mul_ps(tempA2, tempB1);
            tempC1i3 += _mm_mul_ps(tempA3, tempB1);
            tempC1i4 += _mm_mul_ps(tempA4, tempB1);
            tempC1i5 += _mm_mul_ps(tempA5, tempB1);
            tempC1i6 += _mm_mul_ps(tempA6, tempB1);
            tempC1i7 += _mm_mul_ps(tempA7, tempB1);
            tempC1i8 += _mm_mul_ps(tempA8, tempB1);
          }

          float *ci = c + i;

          _mm_storeu_ps(ci+j1, tempC1i1);

          ci += 4;

          _mm_storeu_ps(ci+j1, tempC1i2);

          ci += 4;

          _mm_storeu_ps(ci+j1, tempC1i3);

          ci += 4;

          _mm_storeu_ps(ci+j1, tempC1i4);

          ci += 4;

          _mm_storeu_ps(ci+j1, tempC1i5);

          ci += 4;

          _mm_storeu_ps(ci+j1, tempC1i6);

          ci += 4;

          _mm_storeu_ps(ci+j1, tempC1i7);

          ci += 4;

          _mm_storeu_ps(ci+j1, tempC1i8);
        }
      }
    }
  }
}

void *sgemm_fringe_thread(void *thereadarg) {
  data_t *myData = (data_t *) thereadarg;
  int j_s = myData->j_start;
  int j_e = myData->j_end;

  __m128 zero = _mm_setzero_ps();

  for( int j = j_s; j < j_e; j += 4) {
    float *bsel = b+j;
    int j1 = j*m_b;
    int j2 = j1 + m_b;
    int j3 = j2 + m_b;
    int j4 = j3 + m_b;

    __m128 tempA1, tempA2, tempA3, tempA4;
    __m128 tempB1;
    __m128 tempC1i1, tempC2i1, tempC3i1, tempC4i1;
    __m128 tempC1i2, tempC2i2, tempC3i2, tempC4i2;
    __m128 tempC1i3, tempC2i3, tempC3i3, tempC4i3;
    __m128 tempC1i4, tempC2i4, tempC3i4, tempC4i4;

    for( int i = m_b32; i < m_b16; i += 16) {
      float *ai = a+i;
      float *ai4 = ai + 4;
      float *ai8 = ai4 + 4;
      float *ai12 = ai8 + 4;

      tempC1i1 = zero; // C values begin at 0 anways, so there is no use in loading
      tempC2i1 = zero; // from C.
      tempC3i1 = zero;
      tempC4i1 = zero;

      tempC1i2 = zero;
      tempC2i2 = zero;
      tempC3i2 = zero;
      tempC4i2 = zero;

      tempC1i3 = zero;
      tempC2i3 = zero;
      tempC3i3 = zero;
      tempC4i3 = zero;

      tempC1i4 = zero;
      tempC2i4 = zero;
      tempC3i4 = zero;
      tempC4i4 = zero;
      for( int k = 0; k < n_b; k++) {
        int k0 = k*m_b;

        tempA1 = _mm_loadu_ps(ai+k0);
        tempA2 = _mm_loadu_ps(ai4+k0);
        tempA3 = _mm_loadu_ps(ai8+k0);
        tempA4 = _mm_loadu_ps(ai12+k0);

        tempB1 = _mm_load1_ps(bsel+k0);

        tempC1i1 += _mm_mul_ps(tempA1, tempB1);
        tempC1i2 += _mm_mul_ps(tempA2, tempB1);
        tempC1i3 += _mm_mul_ps(tempA3, tempB1);
        tempC1i4 += _mm_mul_ps(tempA4, tempB1);

        tempB1 = _mm_load1_ps(bsel+1+k0);

        tempC2i1 += _mm_mul_ps(tempA1, tempB1);
        tempC2i2 += _mm_mul_ps(tempA2, tempB1);
        tempC2i3 += _mm_mul_ps(tempA3, tempB1);
        tempC2i4 += _mm_mul_ps(tempA4, tempB1);

        tempB1 = _mm_load1_ps(bsel+2+k0);

        tempC3i1 += _mm_mul_ps(tempA1, tempB1);
        tempC3i2 += _mm_mul_ps(tempA2, tempB1);
        tempC3i3 += _mm_mul_ps(tempA3, tempB1);
        tempC3i4 += _mm_mul_ps(tempA4, tempB1);

        tempB1 = _mm_load1_ps(bsel+3+k0);

        tempC4i1 += _mm_mul_ps(tempA1, tempB1);
        tempC4i2 += _mm_mul_ps(tempA2, tempB1);
        tempC4i3 += _mm_mul_ps(tempA3, tempB1);
        tempC4i4 += _mm_mul_ps(tempA4, tempB1);
      }
      float *ci = c+i;

      _mm_storeu_ps(ci+j1, tempC1i1);
      _mm_storeu_ps(ci+j2, tempC2i1);
      _mm_storeu_ps(ci+j3, tempC3i1);
      _mm_storeu_ps(ci+j4, tempC4i1);

      ci += 4;

      _mm_storeu_ps(ci+j1, tempC1i2);
      _mm_storeu_ps(ci+j2, tempC2i2);
      _mm_storeu_ps(ci+j3, tempC3i2);
      _mm_storeu_ps(ci+j4, tempC4i2);

      ci += 4;

      _mm_storeu_ps(ci+j1, tempC1i3);
      _mm_storeu_ps(ci+j2, tempC2i3);
      _mm_storeu_ps(ci+j3, tempC3i3);
      _mm_storeu_ps(ci+j4, tempC4i3);

      ci += 4;

      _mm_storeu_ps(ci+j1, tempC1i4);
      _mm_storeu_ps(ci+j2, tempC2i4);
      _mm_storeu_ps(ci+j3, tempC3i4);
      _mm_storeu_ps(ci+j4, tempC4i4);

    }
    for( int i = m_b16; i < m_b4; i += 4) {
      float *ai = a+i;

      tempC1i1 = zero;
      tempC2i1 = zero;
      tempC3i1 = zero;
      tempC4i1 = zero;
      for( int k = 0; k < n_b; k++) {
        int k0 = k*m_b;

        tempA1 = _mm_loadu_ps(ai+k0);
        tempB1 = _mm_load1_ps(bsel+k0);
        tempC1i1 += _mm_mul_ps(tempA1, tempB1);

        tempB1 = _mm_load1_ps(bsel+1+k0);
        tempC2i1 += _mm_mul_ps(tempA1, tempB1);

        tempB1 = _mm_load1_ps(bsel+2+k0);
        tempC3i1 += _mm_mul_ps(tempA1, tempB1);

        tempB1 = _mm_load1_ps(bsel+3+k0);
        tempC4i1 += _mm_mul_ps(tempA1, tempB1);
      }
      _mm_storeu_ps(c+i+j1, tempC1i1);
      _mm_storeu_ps(c+i+j2, tempC2i1);
      _mm_storeu_ps(c+i+j3, tempC3i1);
      _mm_storeu_ps(c+i+j4, tempC4i1);
    }
    for( int i = m_b4; i < m_b; i++) {
      float *ai = a+i;

      tempC1i1 = zero;
      tempC2i1 = zero;
      tempC3i1 = zero;
      tempC4i1 = zero;
      for( int k = 0; k < n_b; k++) {
        int k0 = k*m_b;

        tempA1 = _mm_loadu_ps(ai+k0);
        tempB1 = _mm_load1_ps(bsel+k0);
        tempC1i1 += _mm_mul_ps(tempA1, tempB1);

        tempB1 = _mm_load1_ps(bsel+1+k0);
        tempC2i1 += _mm_mul_ps(tempA1, tempB1);

        tempB1 = _mm_load1_ps(bsel+2+k0);
        tempC3i1 += _mm_mul_ps(tempA1, tempB1);

        tempB1 = _mm_load1_ps(bsel+3+k0);
        tempC4i1 += _mm_mul_ps(tempA1, tempB1);
      }
      _mm_store_ss(c+i+j1, tempC1i1);
      _mm_store_ss(c+i+j2, tempC2i1);
      _mm_store_ss(c+i+j3, tempC3i1);
      _mm_store_ss(c+i+j4, tempC4i1);
    }
  }
}

void *sgemm_leftover_thread(void *thereadarg) {
  data_t *myData = (data_t *) thereadarg;
  int j_s = myData->j_start;
  int j_e = myData->j_end;
  //printf("leftover start:%d  end:%d  diff:%d\n",j_s,j_e,j_e-j_s);

  __m128 zero = _mm_setzero_ps();

  for( int j = j_s; j < j_e; j++) {
    float *bsel = b+j;
    int j1 = j*m_b;

    __m128 tempA1;
    __m128 tempB1;
    __m128 tempC1i1, tempC2i1, tempC3i1, tempC4i1;

    for( int i = m_b32; i < m_b16; i += 16) {
      float *ai = a+i;
      float *ai4 = ai + 4;
      float *ai8 = ai4 + 4;
      float *ai12 = ai8 + 4;

      tempC1i1 = zero; // C values begin at 0 anways, so there is no use in loading
      tempC2i1 = zero; // from C.
      tempC3i1 = zero;
      tempC4i1 = zero;
      for( int k = 0; k < n_b; k++) {
        int k0 = k*m_b;

        tempB1 = _mm_load1_ps(bsel+k0);
        tempA1 = _mm_loadu_ps(ai+k0);
        tempC1i1 += _mm_mul_ps(tempA1, tempB1);

        tempA1 = _mm_loadu_ps(ai+4+k0);
        tempC2i1 += _mm_mul_ps(tempA1, tempB1);

        tempA1 = _mm_loadu_ps(ai+8+k0);
        tempC3i1 += _mm_mul_ps(tempA1, tempB1);

        tempA1 = _mm_loadu_ps(ai+12+k0);
        tempC4i1 += _mm_mul_ps(tempA1, tempB1);
      }
      float *ci = c+i;

      _mm_storeu_ps(ci+j1, tempC1i1);
      _mm_storeu_ps(ci+4+j1, tempC2i1);
      _mm_storeu_ps(ci+8+j1, tempC3i1);
      _mm_storeu_ps(ci+12+j1, tempC4i1);
    }
    for( int i = m_b16; i < m_b4; i += 4) {
      float *ai = a+i;

      tempC1i1 = zero;
      for( int k = 0; k < n_b; k++) {
        int k0 = k*m_b;

        tempA1 = _mm_loadu_ps(ai+k0);
        tempB1 = _mm_load1_ps(bsel+k0);
        tempC1i1 += _mm_mul_ps(tempA1, tempB1);
      }
      _mm_storeu_ps(c+i+j1, tempC1i1);
    }
    for( int i = m_b4; i < m_b; i++) {
      float *ai = a+i;

      tempC1i1 = zero;
      for( int k = 0; k < n_b; k++) {
        int k0 = k*m_b;

        tempA1 = _mm_loadu_ps(ai+k0);
        tempB1 = _mm_load1_ps(bsel+k0);
        tempC1i1 += _mm_mul_ps(tempA1, tempB1);
      }
      _mm_store_ss(c+i+j1, tempC1i1);
    }
  }
}

/**
 *  We create 16 threads for each of the chunks of code which used to have
 *  pragmas, as well as one thread for the leftovers case, making a total of
 *  33 threads.
 **/

void sgemm( int m_a, int n_a, float *A, float *B, float *C ) {
  pthread_t mythread1;
  pthread_t mythread2;
  pthread_t mythread3;
  pthread_t mythread4;
  pthread_t mythread5;
  pthread_t mythread6;
  pthread_t mythread7;
  pthread_t mythread8;
  pthread_t mythread9;
  pthread_t mythread10;
  pthread_t mythread11;
  pthread_t mythread12;
  pthread_t mythread13;
  pthread_t mythread14;
  pthread_t mythread15;
  pthread_t mythread16;
  pthread_t mythread17;
  pthread_t mythread18;
  pthread_t mythread19;
  pthread_t mythread20;
  pthread_t mythread21;
  pthread_t mythread22;
  pthread_t mythread23;
  pthread_t mythread24;
  pthread_t mythread25;
  pthread_t mythread26;
  pthread_t mythread27;
  pthread_t mythread28;
  pthread_t mythread29;
  pthread_t mythread30;
  pthread_t mythread31;
  pthread_t mythread32;
  pthread_t mythread33;

  n_b = n_a;
  m_b = m_a;
  m_b4 = m_a/4*4;
  m_b16 = m_a/16*16;
  m_b32 = m_a/32*32;
  int m_b64 = m_a/64*64;
  int m_b256 = m_b/256*256;

  a = A;
  b = B;
  c = C;

  int division = m_b256/16; // Splits B matrix into 16 different sections
  int counter1 = (m_b16-m_b256)/16; // Counter that distributes parts of B<m_b16 but greater than m_b256
  int counter2 = (m_b16-m_b64)/4; // Does the same thing for m_b16

  int progression = 0; // Counter that shows how much of the B matrix has been assigned to a thread already.

  data_t *myData1 = malloc(sizeof(data_t));
  myData1->j_start = progression;
  myData1->j_end = progression + division;
  if(counter1){ // if counter is not 0, then you need to assign more of the matrix to the thread in order
    myData1->j_end += 16; //to traverse through all of B by the end.
    counter1--;
    progression += 16;
  }


  if(pthread_create(&mythread1, NULL, sgemm_main_thread, myData1)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData2 = malloc(sizeof(data_t));
  myData2->j_start = progression;
  myData2->j_end = progression + division;
  if(counter1){
    myData2->j_end += 16;
    counter1--;
    progression += 16;
  }


  if(pthread_create(&mythread2, NULL, sgemm_main_thread, myData2)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData3 = malloc(sizeof(data_t));
  myData3->j_start = progression;
  myData3->j_end = progression + division;
  if(counter1){
    myData3->j_end += 16;
    counter1--;
    progression += 16;
  }


  if(pthread_create(&mythread3, NULL, sgemm_main_thread, myData3)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData4 = malloc(sizeof(data_t));
  myData4->j_start = progression;
  myData4->j_end = progression + division;
  if(counter1){
    myData4->j_end += 16;
    counter1--;
    progression += 16;
  }


  if(pthread_create(&mythread4, NULL, sgemm_main_thread, myData4)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData5 = malloc(sizeof(data_t));
  myData5->j_start = progression;
  myData5->j_end = progression + division;
  if(counter1){
    myData5->j_end += 16;
    counter1--;
    progression += 16;
  }


  if(pthread_create(&mythread5, NULL, sgemm_main_thread, myData5)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData6 = malloc(sizeof(data_t));
  myData6->j_start = progression;
  myData6->j_end = progression + division;
  if(counter1){
    myData6->j_end += 16;
    counter1--;
    progression += 16;
  }


  if(pthread_create(&mythread6, NULL, sgemm_main_thread, myData6)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData7 = malloc(sizeof(data_t));
  myData7->j_start = progression;
  myData7->j_end = progression + division;
  if(counter1){
    myData7->j_end += 16;
    counter1--;
    progression += 16;
  }


  if(pthread_create(&mythread7, NULL, sgemm_main_thread, myData7)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData8 = malloc(sizeof(data_t));
  myData8->j_start = progression;
  myData8->j_end = progression + division;
  if(counter1){
    myData8->j_end += 16;
    counter1--;
    progression += 16;
  }


  if(pthread_create(&mythread8, NULL, sgemm_main_thread, myData8)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData9 = malloc(sizeof(data_t));
  myData9->j_start = progression;
  myData9->j_end = progression + division;
  if(counter1){
    myData9->j_end += 16;
    counter1--;
    progression += 16;
  }


  if(pthread_create(&mythread9, NULL, sgemm_main_thread, myData9)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData10 = malloc(sizeof(data_t));
  myData10->j_start = progression;
  myData10->j_end = progression + division;
  if(counter1){
    myData10->j_end += 16;
    counter1--;
    progression += 16;
  }


  if(pthread_create(&mythread10, NULL, sgemm_main_thread, myData10)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData11 = malloc(sizeof(data_t));
  myData11->j_start = progression;
  myData11->j_end = progression + division;
  if(counter1){
    myData11->j_end += 16;
    counter1--;
    progression += 16;
  }


  if(pthread_create(&mythread11, NULL, sgemm_main_thread, myData11)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData12 = malloc(sizeof(data_t));
  myData12->j_start = progression;
  myData12->j_end = progression + division;
  if(counter1){
    myData12->j_end += 16;
    counter1--;
    progression += 16;
  }

  if(pthread_create(&mythread12, NULL, sgemm_main_thread, myData12)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData13 = malloc(sizeof(data_t));
  myData13->j_start = progression;
  myData13->j_end = progression + division;
  if(counter1){
    myData13->j_end += 16;
    counter1--;
    progression += 16;
  }


  if(pthread_create(&mythread13, NULL, sgemm_main_thread, myData13)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData14 = malloc(sizeof(data_t));
  myData14->j_start = progression;
  myData14->j_end = progression + division;
  if(counter1){
    myData14->j_end += 16;
    counter1--;
    progression += 16;
  }


  if(pthread_create(&mythread14, NULL, sgemm_main_thread, myData14)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData15 = malloc(sizeof(data_t));
  myData15->j_start = progression;
  myData15->j_end = progression + division;
  if(counter1){
    myData15->j_end += 16;
    counter1--;
    progression += 16;
  }


  if(pthread_create(&mythread15, NULL, sgemm_main_thread, myData15)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData16 = malloc(sizeof(data_t));
  myData16->j_start = progression;
  myData16->j_end = progression + division;
  if(counter1){
    myData16->j_end += 16;
    counter1--;
    progression += 16;
  }

  if(pthread_create(&mythread16, NULL, sgemm_main_thread, myData16)) {
    thread_api_failure();
  }

  progression = 0;

  data_t *myData17 = malloc(sizeof(data_t));
  myData17->j_start = progression;
  myData17->j_end = progression + division;
  if(counter2){
    myData17->j_end += 4;
    counter2--;
    progression += 4;
  }

  if(pthread_create(&mythread17, NULL, sgemm_fringe_thread, myData17)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData18 = malloc(sizeof(data_t));
  myData18->j_start = progression;
  myData18->j_end = progression + division;
  if(counter2){
    myData18->j_end += 4;
    counter2--;
    progression += 4;
  }

  if(pthread_create(&mythread18, NULL, sgemm_fringe_thread, myData18)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData19 = malloc(sizeof(data_t));
  myData19->j_start = progression;
  myData19->j_end = progression + division;
  if(counter2){
    myData19->j_end += 4;
    counter2--;
    progression += 4;
  }

  if(pthread_create(&mythread19, NULL, sgemm_fringe_thread, myData19)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData20 = malloc(sizeof(data_t));
  myData20->j_start = progression;
  myData20->j_end = progression + division;
  if(counter2){
    myData20->j_end += 4;
    counter2--;
    progression += 4;
  }

  if(pthread_create(&mythread20, NULL, sgemm_fringe_thread, myData20)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData21 = malloc(sizeof(data_t));
  myData21->j_start = progression;
  myData21->j_end = progression + division;
  if(counter2){
    myData21->j_end += 4;
    counter2--;
    progression += 4;
  }

  if(pthread_create(&mythread21, NULL, sgemm_fringe_thread, myData21)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData22 = malloc(sizeof(data_t));
  myData22->j_start = progression;
  myData22->j_end = progression + division;
  if(counter2){
    myData22->j_end += 4;
    counter2--;
    progression += 4;
  }

  if(pthread_create(&mythread22, NULL, sgemm_fringe_thread, myData22)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData23 = malloc(sizeof(data_t));
  myData23->j_start = progression;
  myData23->j_end = progression + division;
  if(counter2){
    myData23->j_end += 4;
    counter2--;
    progression += 4;
  }

  if(pthread_create(&mythread23, NULL, sgemm_fringe_thread, myData23)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData24 = malloc(sizeof(data_t));
  myData24->j_start = progression;
  myData24->j_end = progression + division;
  if(counter2){
    myData24->j_end += 4;
    counter2--;
    progression += 4;
  }

  if(pthread_create(&mythread24, NULL, sgemm_fringe_thread, myData24)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData25 = malloc(sizeof(data_t));
  myData25->j_start = progression;
  myData25->j_end = progression + division;
  if(counter2){
    myData25->j_end += 4;
    counter2--;
    progression += 4;
  }

  if(pthread_create(&mythread25, NULL, sgemm_fringe_thread, myData25)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData26 = malloc(sizeof(data_t));
  myData26->j_start = progression;
  myData26->j_end = progression + division;
  if(counter2){
    myData26->j_end += 4;
    counter2--;
    progression += 4;
  }

  if(pthread_create(&mythread26, NULL, sgemm_fringe_thread, myData26)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData27 = malloc(sizeof(data_t));
  myData27->j_start = progression;
  myData27->j_end = progression + division;
  if(counter2){
    myData27->j_end += 4;
    counter2--;
    progression += 4;
  }

  if(pthread_create(&mythread27, NULL, sgemm_fringe_thread, myData27)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData28 = malloc(sizeof(data_t));
  myData28->j_start = progression;
  myData28->j_end = progression + division;
  if(counter2){
    myData28->j_end += 4;
    counter2--;
    progression += 4;
  }

  if(pthread_create(&mythread28, NULL, sgemm_fringe_thread, myData28)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData29 = malloc(sizeof(data_t));
  myData29->j_start = progression;
  myData29->j_end = progression + division;
  if(counter2){
    myData29->j_end += 4;
    counter2--;
    progression += 4;
  }

  if(pthread_create(&mythread29, NULL, sgemm_fringe_thread, myData29)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData30 = malloc(sizeof(data_t));
  myData30->j_start = progression;
  myData30->j_end = progression + division;
  if(counter2){
    myData30->j_end += 4;
    counter2--;
    progression += 4;
  }

  if(pthread_create(&mythread30, NULL, sgemm_fringe_thread, myData30)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData31 = malloc(sizeof(data_t));
  myData31->j_start = progression;
  myData31->j_end = progression + division;
  if(counter2){
    myData31->j_end += 4;
    counter2--;
    progression += 4;
  }

  if(pthread_create(&mythread31, NULL, sgemm_fringe_thread, myData31)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData32 = malloc(sizeof(data_t));
  myData32->j_start = progression;
  myData32->j_end = progression + division;
  if(counter2){
    myData32->j_end += 4;
    counter2--;
    progression += 4;
  }

  if(pthread_create(&mythread32, NULL, sgemm_fringe_thread, myData32)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData33 = malloc(sizeof(data_t));
  myData33->j_start = progression;
  myData33->j_end = m_b;

  if(pthread_create(&mythread33, NULL, sgemm_leftover_thread, myData33)) {
    thread_api_failure();
  }


  if ( pthread_join ( mythread1, NULL ) ) {
    thread_api_failure();
  }

  if ( pthread_join ( mythread2, NULL ) ) {
    thread_api_failure();
  }

  if ( pthread_join ( mythread3, NULL ) ) {
    thread_api_failure();
  }

  if ( pthread_join ( mythread4, NULL ) ) {
    thread_api_failure();
  }

  if ( pthread_join ( mythread5, NULL ) ) {
    thread_api_failure();
  }

  if ( pthread_join ( mythread6, NULL ) ) {
    thread_api_failure();
  }

  if ( pthread_join ( mythread7, NULL ) ) {
    thread_api_failure();
  }

  if ( pthread_join ( mythread8, NULL ) ) {
    thread_api_failure();
  }

  if ( pthread_join ( mythread9, NULL ) ) {
    thread_api_failure();
  }

  if ( pthread_join ( mythread10, NULL ) ) {
    thread_api_failure();
  }

  if ( pthread_join ( mythread11, NULL ) ) {
    thread_api_failure();
  }

  if ( pthread_join ( mythread12, NULL ) ) {
    thread_api_failure();
  }

  if ( pthread_join ( mythread13, NULL ) ) {
    thread_api_failure();
  }

  if ( pthread_join ( mythread14, NULL ) ) {
    thread_api_failure();
  }

  if ( pthread_join ( mythread15, NULL ) ) {
    thread_api_failure();
  }

  if ( pthread_join ( mythread16, NULL ) ) {
    thread_api_failure();
  }

  if ( pthread_join ( mythread17, NULL ) ) {
    thread_api_failure();
  }

  if ( pthread_join ( mythread18, NULL ) ) {
    thread_api_failure();
  }

  if ( pthread_join ( mythread19, NULL ) ) {
    thread_api_failure();
  }

  if ( pthread_join ( mythread20, NULL ) ) {
    thread_api_failure();
  }

  if ( pthread_join ( mythread21, NULL ) ) {
    thread_api_failure();
  }

  if ( pthread_join ( mythread22, NULL ) ) {
    thread_api_failure();
  }

  if ( pthread_join ( mythread23, NULL ) ) {
    thread_api_failure();
  }

  if ( pthread_join ( mythread24, NULL ) ) {
    thread_api_failure();
  }

  if ( pthread_join ( mythread25, NULL ) ) {
    thread_api_failure();
  }

  if ( pthread_join ( mythread26, NULL ) ) {
    thread_api_failure();
  }

  if ( pthread_join ( mythread27, NULL ) ) {
    thread_api_failure();
  }

  if ( pthread_join ( mythread28, NULL ) ) {
    thread_api_failure();
  }

  if ( pthread_join ( mythread29, NULL ) ) {
    thread_api_failure();
  }

  if ( pthread_join ( mythread30, NULL ) ) {
    thread_api_failure();
  }

  if ( pthread_join ( mythread31, NULL ) ) {
    thread_api_failure();
  }

  if ( pthread_join ( mythread32, NULL ) ) {
    thread_api_failure();
  }

  if ( pthread_join ( mythread33, NULL ) ) {
    thread_api_failure();
  }

  //Clean up
  free(myData1);
  free(myData2);
  free(myData3);
  free(myData4);
  free(myData5);
  free(myData6);
  free(myData7);
  free(myData8);
  free(myData9);
  free(myData10);
  free(myData11);
  free(myData12);
  free(myData13);
  free(myData14);
  free(myData15);
  free(myData16);
  free(myData17);
  free(myData18);
  free(myData19);
  free(myData20);
  free(myData21);
  free(myData22);
  free(myData23);
  free(myData24);
  free(myData25);
  free(myData26);
  free(myData27);
  free(myData28);
  free(myData29);
  free(myData30);
  free(myData31);
  free(myData32);
  free(myData33);

  return;
}
