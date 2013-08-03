#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <nmmintrin.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define BLOCKSIZE 64

float *a;
float *b;
float *c;
int m_b;
int n_b;

void thread_api_failure(void) {
    printf("Pthreads API function returned nonzero status!\n");
    exit(-1);
}

struct thread_data {
    int j_start;
    int j_end;
};

typedef struct thread_data data_t;

void *thread_function(void *thereadarg) {
  data_t *myData = (data_t *) thereadarg;
  int j_s = myData->j_start;
  int j_e = myData->j_end;
  printf("start:%d  end:%d  diff:%d\n",j_s,j_e,j_e-j_s);

  for( int y = 0; y < m_b; y += BLOCKSIZE) {
    int e = MIN(m_b, y+BLOCKSIZE);
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
    for(int j = j_s; j < j_e; j += 16) {
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

        tempC1i1 = _mm_setzero_ps(); // C values begin at 0 anways, so there is no use in loading
        tempC2i1 = _mm_setzero_ps(); // from C.
        tempC3i1 = _mm_setzero_ps(); 
        tempC4i1 = _mm_setzero_ps(); 
        tempC5i1 = _mm_setzero_ps(); 
        tempC6i1 = _mm_setzero_ps(); 
        tempC7i1 = _mm_setzero_ps(); 
        tempC8i1 = _mm_setzero_ps(); 
        tempC9i1 = _mm_setzero_ps(); 
        tempC10i1 = _mm_setzero_ps(); 
        tempC11i1 = _mm_setzero_ps(); 
        tempC12i1 = _mm_setzero_ps(); 
        tempC13i1 = _mm_setzero_ps(); 
        tempC14i1 = _mm_setzero_ps(); 
        tempC15i1 = _mm_setzero_ps(); 
        tempC16i1 = _mm_setzero_ps(); 

        tempC1i2 = _mm_setzero_ps(); 
        tempC2i2 = _mm_setzero_ps(); 
        tempC3i2 = _mm_setzero_ps(); 
        tempC4i2 = _mm_setzero_ps(); 
        tempC5i2 = _mm_setzero_ps(); 
        tempC6i2 = _mm_setzero_ps(); 
        tempC7i2 = _mm_setzero_ps(); 
        tempC8i2 = _mm_setzero_ps(); 
        tempC9i2 = _mm_setzero_ps(); 
        tempC10i2 = _mm_setzero_ps(); 
        tempC11i2 = _mm_setzero_ps(); 
        tempC12i2 = _mm_setzero_ps(); 
        tempC13i2 = _mm_setzero_ps(); 
        tempC14i2 = _mm_setzero_ps(); 
        tempC15i2 = _mm_setzero_ps(); 
        tempC16i2 = _mm_setzero_ps(); 

        tempC1i3 = _mm_setzero_ps(); 
        tempC2i3 = _mm_setzero_ps(); 
        tempC3i3 = _mm_setzero_ps(); 
        tempC4i3 = _mm_setzero_ps(); 
        tempC5i3 = _mm_setzero_ps(); 
        tempC6i3 = _mm_setzero_ps(); 
        tempC7i3 = _mm_setzero_ps(); 
        tempC8i3 = _mm_setzero_ps(); 
        tempC9i3 = _mm_setzero_ps(); 
        tempC10i3 = _mm_setzero_ps(); 
        tempC11i3 = _mm_setzero_ps(); 
        tempC12i3 = _mm_setzero_ps(); 
        tempC13i3 = _mm_setzero_ps(); 
        tempC14i3 = _mm_setzero_ps(); 
        tempC15i3 = _mm_setzero_ps(); 
        tempC16i3 = _mm_setzero_ps(); 

        tempC1i4 = _mm_setzero_ps(); 
        tempC2i4 = _mm_setzero_ps(); 
        tempC3i4 = _mm_setzero_ps(); 
        tempC4i4 = _mm_setzero_ps(); 
        tempC5i4 = _mm_setzero_ps(); 
        tempC6i4 = _mm_setzero_ps(); 
        tempC7i4 = _mm_setzero_ps(); 
        tempC8i4 = _mm_setzero_ps(); 
        tempC9i4 = _mm_setzero_ps(); 
        tempC10i4 = _mm_setzero_ps(); 
        tempC11i4 = _mm_setzero_ps(); 
        tempC12i4 = _mm_setzero_ps(); 
        tempC13i4 = _mm_setzero_ps(); 
        tempC14i4 = _mm_setzero_ps(); 
        tempC15i4 = _mm_setzero_ps(); 
        tempC16i4 = _mm_setzero_ps(); 

        tempC1i5 = _mm_setzero_ps(); 
        tempC2i5 = _mm_setzero_ps(); 
        tempC3i5 = _mm_setzero_ps(); 
        tempC4i5 = _mm_setzero_ps(); 
        tempC5i5 = _mm_setzero_ps(); 
        tempC6i5 = _mm_setzero_ps(); 
        tempC7i5 = _mm_setzero_ps(); 
        tempC8i5 = _mm_setzero_ps(); 
        tempC9i5 = _mm_setzero_ps(); 
        tempC10i5 = _mm_setzero_ps(); 
        tempC11i5 = _mm_setzero_ps(); 
        tempC12i5 = _mm_setzero_ps(); 
        tempC13i5 = _mm_setzero_ps(); 
        tempC14i5 = _mm_setzero_ps(); 
        tempC15i5 = _mm_setzero_ps(); 
        tempC16i5 = _mm_setzero_ps(); 

        tempC1i6 = _mm_setzero_ps(); 
        tempC2i6 = _mm_setzero_ps(); 
        tempC3i6 = _mm_setzero_ps(); 
        tempC4i6 = _mm_setzero_ps(); 
        tempC5i6 = _mm_setzero_ps(); 
        tempC6i6 = _mm_setzero_ps(); 
        tempC7i6 = _mm_setzero_ps(); 
        tempC8i6 = _mm_setzero_ps(); 
        tempC9i6 = _mm_setzero_ps(); 
        tempC10i6 = _mm_setzero_ps(); 
        tempC11i6 = _mm_setzero_ps(); 
        tempC12i6 = _mm_setzero_ps(); 
        tempC13i6 = _mm_setzero_ps(); 
        tempC14i6 = _mm_setzero_ps(); 
        tempC15i6 = _mm_setzero_ps(); 
        tempC16i6 = _mm_setzero_ps(); 

        tempC1i7 = _mm_setzero_ps(); 
        tempC2i7 = _mm_setzero_ps(); 
        tempC3i7 = _mm_setzero_ps(); 
        tempC4i7 = _mm_setzero_ps(); 
        tempC5i7 = _mm_setzero_ps(); 
        tempC6i7 = _mm_setzero_ps(); 
        tempC7i7 = _mm_setzero_ps(); 
        tempC8i7 = _mm_setzero_ps(); 
        tempC9i7 = _mm_setzero_ps(); 
        tempC10i7 = _mm_setzero_ps(); 
        tempC11i7 = _mm_setzero_ps(); 
        tempC12i7 = _mm_setzero_ps(); 
        tempC13i7 = _mm_setzero_ps(); 
        tempC14i7 = _mm_setzero_ps(); 
        tempC15i7 = _mm_setzero_ps(); 
        tempC16i7 = _mm_setzero_ps(); 
        
        tempC1i8 = _mm_setzero_ps(); 
        tempC2i8 = _mm_setzero_ps(); 
        tempC3i8 = _mm_setzero_ps(); 
        tempC4i8 = _mm_setzero_ps(); 
        tempC5i8 = _mm_setzero_ps(); 
        tempC6i8 = _mm_setzero_ps(); 
        tempC7i8 = _mm_setzero_ps(); 
        tempC8i8 = _mm_setzero_ps(); 
        tempC9i8 = _mm_setzero_ps(); 
        tempC10i8 = _mm_setzero_ps(); 
        tempC11i8 = _mm_setzero_ps(); 
        tempC12i8 = _mm_setzero_ps(); 
        tempC13i8 = _mm_setzero_ps(); 
        tempC14i8 = _mm_setzero_ps(); 
        tempC15i8 = _mm_setzero_ps(); 
        tempC16i8 = _mm_setzero_ps(); 

        for( int k = 0; k < n_b; k++ ) {
          int k0 = k*m_b;

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
  }
}

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

  int m_a32 = m_a/32*32, m_diff = 0;
  int m_a256 = m_a/256*256;

  if( m_a32 != m_a256 ){ // if matrices need padding, m_a32 becomes the
    m_a32 += 32;       // lowest multiple of 16 greater than m_a
    m_diff = m_a32 - m_a;

    a = calloc(m_a32*n_a, sizeof(float));
    b = calloc(m_a32*n_a, sizeof(float));
    c = malloc(m_a32*m_a32 * sizeof(float));

    for( int i = 0; i < n_a; i++ ){ // moves the values of A and B into a and b
      for( int j = 0; j < m_a; j++ ){
        *(a + i*m_a32 + j) = *(A + i*m_a + j);
        *(b + i*m_a32 + j) = *(B + i*m_a + j);
      }
    }
  }else{
    a = A;
    b = B;
    c = C;
  }

  m_b = m_a32;
  n_b = n_a;
  int division = m_a256/16;
  int progression = 0;
  data_t *myData1 = malloc(sizeof(data_t));
  myData1->j_start = progression;
  myData1->j_end = progression + division;

  if(pthread_create(&mythread1, NULL, thread_function, myData1)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData2 = malloc(sizeof(data_t));
  myData2->j_start = progression;
  myData2->j_end = progression + division;

  if(pthread_create(&mythread2, NULL, thread_function, myData2)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData3 = malloc(sizeof(data_t));
  myData3->j_start = progression;
  myData3->j_end = progression + division;

  if(pthread_create(&mythread3, NULL, thread_function, myData3)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData4 = malloc(sizeof(data_t));
  myData4->j_start = progression;
  myData4->j_end = progression + division;

  if(pthread_create(&mythread4, NULL, thread_function, myData4)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData5 = malloc(sizeof(data_t));
  myData5->j_start = progression;
  myData5->j_end = progression + division;

  if(pthread_create(&mythread5, NULL, thread_function, myData5)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData6 = malloc(sizeof(data_t));
  myData6->j_start = progression;
  myData6->j_end = progression + division;

  if(pthread_create(&mythread6, NULL, thread_function, myData6)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData7 = malloc(sizeof(data_t));
  myData7->j_start = progression;
  myData7->j_end = progression + division;

  if(pthread_create(&mythread7, NULL, thread_function, myData7)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData8 = malloc(sizeof(data_t));
  myData8->j_start = progression;
  myData8->j_end = progression + division;

  if(pthread_create(&mythread8, NULL, thread_function, myData8)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData9 = malloc(sizeof(data_t));
  myData9->j_start = progression;
  myData9->j_end = progression + division;

  if(pthread_create(&mythread9, NULL, thread_function, myData9)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData10 = malloc(sizeof(data_t));
  myData10->j_start = progression;
  myData10->j_end = progression + division;

  if(pthread_create(&mythread10, NULL, thread_function, myData10)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData11 = malloc(sizeof(data_t));
  myData11->j_start = progression;
  myData11->j_end = progression + division;

  if(pthread_create(&mythread11, NULL, thread_function, myData11)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData12 = malloc(sizeof(data_t));
  myData12->j_start = progression;
  myData12->j_end = progression + division;

  if(pthread_create(&mythread12, NULL, thread_function, myData12)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData13 = malloc(sizeof(data_t));
  myData13->j_start = progression;
  myData13->j_end = progression + division;

  if(pthread_create(&mythread13, NULL, thread_function, myData13)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData14 = malloc(sizeof(data_t));
  myData14->j_start = progression;
  myData14->j_end = progression + division;

  if(pthread_create(&mythread14, NULL, thread_function, myData14)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData15 = malloc(sizeof(data_t));
  myData15->j_start = progression;
  myData15->j_end = progression + division;

  if(pthread_create(&mythread15, NULL, thread_function, myData15)) {
    thread_api_failure();
  }

  progression += division;

  data_t *myData16 = malloc(sizeof(data_t));
  myData16->j_start = progression;
  myData16->j_end = progression + division;

  if(pthread_create(&mythread16, NULL, thread_function, myData16)) {
    thread_api_failure();
  }

  data_t *myData17;
  if( m_diff ){
    myData17 = malloc(sizeof(data_t));
    myData17->j_start = progression + division;
    myData17->j_end = m_a32;

    if(pthread_create(&mythread17, NULL, thread_function, myData17)) {
      thread_api_failure();
    }
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

  if( m_diff ){
    if ( pthread_join ( mythread17, NULL ) ) {
      thread_api_failure();
    }
  }

  if( m_diff ){
    free(a); // frees allocated matrices
    free(b); // a and b don't change, so no need to move it back
    for(int i = 0; i < m_a; i++){
      for(int j = 0; j < m_a; j++){
        *(C + i*m_a + j) = *(c + i*m_a32 + j);
      }
    }
    free(c);
    free(myData17);
  }

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

  return;
}
