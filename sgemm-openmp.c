#include <nmmintrin.h> /* where intrinsics are defined */
#include <stdio.h>
#include <omp.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))

void sgemm( int m_a, int n_a, float *A, float *B, float *C ) {

  float *a, *b, *c;
  a = A;
  b = B;
  c = C;

  int m_a4 = m_a/4*4;
  int m_a16 = m_a/16*16;
  int m_a32 = m_a/32*32;
//  int m_a64 = m_a/64*64;

  // e, y
  // j, b, bsel
  // j0, j1, j2, j3
  // i, a, ai, c, ci
  // k, k0, k2
  // n_a, m_a4, m_a32

  int blocksize = 64;

  #pragma omp parallel for
  for( int y = 0; y < m_a; y += blocksize) {
    int e = MIN(m_a32, y+blocksize);
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
    for(int j = 0; j < m_a16; j += 16) {
      float *bsel = b + j;
      int j1 = j*m_a;
      int j2 = j1 + m_a;
      int j3 = j2 + m_a;
      int j4 = j3 + m_a;
      int j5 = j4 + m_a;
      int j6 = j5 + m_a;
      int j7 = j6 + m_a;
      int j8 = j7 + m_a;
      int j9 = j8 + m_a;
      int j10 = j9 + m_a;
      int j11 = j10 + m_a;
      int j12 = j11 + m_a;
      int j13 = j12 + m_a;
      int j14 = j13 + m_a;
      int j15 = j14 + m_a;
      int j16 = j15 + m_a;
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

        for( int k = 0; k < n_a; k++ ) {
          int k0 = k*m_a ;

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

      if(e == m_a32){
        if(m_a32 != m_a16){
          float *ai = a+m_a32;
          float *ai4 = ai + 4;
          float *ai8 = ai4 + 4;
          float *ai12 = ai8 + 4;

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

          for( int k = 0; k < n_a; k++){
            int k0 = k*m_a;

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

            tempB1 = _mm_load1_ps(bsel+4+k0);

            tempC5i1 += _mm_mul_ps(tempA1, tempB1);
            tempC5i2 += _mm_mul_ps(tempA2, tempB1);
            tempC5i3 += _mm_mul_ps(tempA3, tempB1);
            tempC5i4 += _mm_mul_ps(tempA4, tempB1);

            tempB1 = _mm_load1_ps(bsel+5+k0);

            tempC6i1 += _mm_mul_ps(tempA1, tempB1);
            tempC6i2 += _mm_mul_ps(tempA2, tempB1);
            tempC6i3 += _mm_mul_ps(tempA3, tempB1);
            tempC6i4 += _mm_mul_ps(tempA4, tempB1);

            tempB1 = _mm_load1_ps(bsel+6+k0);

            tempC7i1 += _mm_mul_ps(tempA1, tempB1);
            tempC7i2 += _mm_mul_ps(tempA2, tempB1);
            tempC7i3 += _mm_mul_ps(tempA3, tempB1);
            tempC7i4 += _mm_mul_ps(tempA4, tempB1);

            tempB1 = _mm_load1_ps(bsel+7+k0);

            tempC8i1 += _mm_mul_ps(tempA1, tempB1);
            tempC8i2 += _mm_mul_ps(tempA2, tempB1);
            tempC8i3 += _mm_mul_ps(tempA3, tempB1);
            tempC8i4 += _mm_mul_ps(tempA4, tempB1);

            tempB1 = _mm_load1_ps(bsel+8+k0);

            tempC9i1 += _mm_mul_ps(tempA1, tempB1);
            tempC9i2 += _mm_mul_ps(tempA2, tempB1);
            tempC9i3 += _mm_mul_ps(tempA3, tempB1);
            tempC9i4 += _mm_mul_ps(tempA4, tempB1);

            tempB1 = _mm_load1_ps(bsel+9+k0);

            tempC10i1 += _mm_mul_ps(tempA1, tempB1);
            tempC10i2 += _mm_mul_ps(tempA2, tempB1);
            tempC10i3 += _mm_mul_ps(tempA3, tempB1);
            tempC10i4 += _mm_mul_ps(tempA4, tempB1);

            tempB1 = _mm_load1_ps(bsel+10+k0);

            tempC11i1 += _mm_mul_ps(tempA1, tempB1);
            tempC11i2 += _mm_mul_ps(tempA2, tempB1);
            tempC11i3 += _mm_mul_ps(tempA3, tempB1);
            tempC11i4 += _mm_mul_ps(tempA4, tempB1);

            tempB1 = _mm_load1_ps(bsel+11+k0);

            tempC12i1 += _mm_mul_ps(tempA1, tempB1);
            tempC12i2 += _mm_mul_ps(tempA2, tempB1);
            tempC12i3 += _mm_mul_ps(tempA3, tempB1);
            tempC12i4 += _mm_mul_ps(tempA4, tempB1);

            tempB1 = _mm_load1_ps(bsel+12+k0);

            tempC13i1 += _mm_mul_ps(tempA1, tempB1);
            tempC13i2 += _mm_mul_ps(tempA2, tempB1);
            tempC13i3 += _mm_mul_ps(tempA3, tempB1);
            tempC13i4 += _mm_mul_ps(tempA4, tempB1);

            tempB1 = _mm_load1_ps(bsel+13+k0);

            tempC14i1 += _mm_mul_ps(tempA1, tempB1);
            tempC14i2 += _mm_mul_ps(tempA2, tempB1);
            tempC14i3 += _mm_mul_ps(tempA3, tempB1);
            tempC14i4 += _mm_mul_ps(tempA4, tempB1);

            tempB1 = _mm_load1_ps(bsel+14+k0);

            tempC15i1 += _mm_mul_ps(tempA1, tempB1);
            tempC15i2 += _mm_mul_ps(tempA2, tempB1);
            tempC15i3 += _mm_mul_ps(tempA3, tempB1);
            tempC15i4 += _mm_mul_ps(tempA4, tempB1);

            tempB1 = _mm_load1_ps(bsel+15+k0);

            tempC16i1 += _mm_mul_ps(tempA1, tempB1);
            tempC16i2 += _mm_mul_ps(tempA2, tempB1);
            tempC16i3 += _mm_mul_ps(tempA3, tempB1);
            tempC16i4 += _mm_mul_ps(tempA4, tempB1);
          }

          float *ci = c+m_a32;

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
        }

        for( int i = m_a16; i < m_a4; i += 4){
          float *ai = a+i;

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

          for( int k = 0; k < n_a; k++){
            int k0 = k*m_a;

            tempA1 = _mm_loadu_ps(ai+k0);

            tempB1 = _mm_load1_ps(bsel+k0);
            tempC1i1 += _mm_mul_ps(tempA1, tempB1);

            tempB1 = _mm_load1_ps(bsel+1+k0);
            tempC2i1 += _mm_mul_ps(tempA1, tempB1);

            tempB1 = _mm_load1_ps(bsel+2+k0);
            tempC3i1 += _mm_mul_ps(tempA1, tempB1);

            tempB1 = _mm_load1_ps(bsel+3+k0);
            tempC4i1 += _mm_mul_ps(tempA1, tempB1);

            tempB1 = _mm_load1_ps(bsel+4+k0);
            tempC5i1 += _mm_mul_ps(tempA1, tempB1);

            tempB1 = _mm_load1_ps(bsel+5+k0);
            tempC6i1 += _mm_mul_ps(tempA1, tempB1);

            tempB1 = _mm_load1_ps(bsel+6+k0);
            tempC7i1 += _mm_mul_ps(tempA1, tempB1);

            tempB1 = _mm_load1_ps(bsel+7+k0);
            tempC8i1 += _mm_mul_ps(tempA1, tempB1);

            tempB1 = _mm_load1_ps(bsel+8+k0);
            tempC9i1 += _mm_mul_ps(tempA1, tempB1);

            tempB1 = _mm_load1_ps(bsel+9+k0);
            tempC10i1 += _mm_mul_ps(tempA1, tempB1);

            tempB1 = _mm_load1_ps(bsel+10+k0);
            tempC11i1 += _mm_mul_ps(tempA1, tempB1);

            tempB1 = _mm_load1_ps(bsel+11+k0);
            tempC12i1 += _mm_mul_ps(tempA1, tempB1);

            tempB1 = _mm_load1_ps(bsel+12+k0);
            tempC13i1 += _mm_mul_ps(tempA1, tempB1);

            tempB1 = _mm_load1_ps(bsel+13+k0);
            tempC14i1 += _mm_mul_ps(tempA1, tempB1);

            tempB1 = _mm_load1_ps(bsel+14+k0);
            tempC15i1 += _mm_mul_ps(tempA1, tempB1);

            tempB1 = _mm_load1_ps(bsel+15+k0);
            tempC16i1 += _mm_mul_ps(tempA1, tempB1);
          }
          float *ci = c+i;

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
        }

        for( int i = m_a4; i < m_a; i++ ) {
          float *ai = a+i;

          tempC1i1 = _mm_setzero_ps(); // C values begin at 0 anways, so there is no use in loading
          tempC2i1 = _mm_setzero_ps(); // from C.
          tempC3i1 = _mm_setzero_ps();
          tempC4i1 = _mm_setzero_ps();

          for( int k = 0; k < n_a; k++ ) {
            int k0 = k*m_a;

            tempA1 = _mm_load1_ps(ai + k0);

            tempB1 = _mm_loadu_ps(bsel + k0);
            tempC1i1 += _mm_mul_ps(tempA1, tempB1);

            tempB1 = _mm_loadu_ps(bsel + 4 + k0);
            tempC2i1 += _mm_mul_ps(tempA1, tempB1);

            tempB1 = _mm_loadu_ps(bsel + 8 + k0);
            tempC3i1 += _mm_mul_ps(tempA1, tempB1);

            tempB1 = _mm_loadu_ps(bsel + 12 + k0);
            tempC4i1 += _mm_mul_ps(tempA1, tempB1);

          }
          float *ci = c+i;

          _mm_store_ss(ci+j1, tempC1i1);
          tempC1i1 = _mm_shuffle_ps(tempC1i1, tempC1i1, 0x39);
          _mm_store_ss(ci+j2, tempC1i1);
          tempC1i1 = _mm_shuffle_ps(tempC1i1, tempC1i1, 0x39);
          _mm_store_ss(ci+j3, tempC1i1);
          tempC1i1 = _mm_shuffle_ps(tempC1i1, tempC1i1, 0x39);
          _mm_store_ss(ci+j4, tempC1i1);

          _mm_store_ss(ci+j5, tempC2i1);
          tempC2i1 = _mm_shuffle_ps(tempC2i1, tempC2i1, 0x39);
          _mm_store_ss(ci+j6, tempC2i1);
          tempC2i1 = _mm_shuffle_ps(tempC2i1, tempC2i1, 0x39);
          _mm_store_ss(ci+j7, tempC2i1);
          tempC2i1 = _mm_shuffle_ps(tempC2i1, tempC2i1, 0x39);
          _mm_store_ss(ci+j8, tempC2i1);

          _mm_store_ss(ci+j9, tempC3i1);
          tempC3i1 = _mm_shuffle_ps(tempC3i1, tempC3i1, 0x39);
          _mm_store_ss(ci+j10, tempC3i1);
          tempC3i1 = _mm_shuffle_ps(tempC3i1, tempC3i1, 0x39);
          _mm_store_ss(ci+j11, tempC3i1);
          tempC3i1 = _mm_shuffle_ps(tempC3i1, tempC3i1, 0x39);
          _mm_store_ss(ci+j12, tempC3i1);

          _mm_store_ss(ci+j13, tempC4i1);
          tempC4i1 = _mm_shuffle_ps(tempC4i1, tempC4i1, 0x39);
          _mm_store_ss(ci+j14, tempC4i1);
          tempC4i1 = _mm_shuffle_ps(tempC4i1, tempC4i1, 0x39);
          _mm_store_ss(ci+j15, tempC4i1);
          tempC4i1 = _mm_shuffle_ps(tempC4i1, tempC4i1, 0x39);
          _mm_store_ss(ci+j16, tempC4i1);
        }
      }
    }
  }

#pragma omp parallel for
  for( int j = m_a16; j < m_a; j++) {
    float *bsel = b+j;
    int j1 = j*m_a;

    __m128 tempA1, tempA2, tempA3, tempA4;
    __m128 tempA5, tempA6, tempA7, tempA8;
    __m128 tempB1;
    __m128 tempC1i1, tempC2i1, tempC3i1, tempC4i1;
    __m128 tempC5i1, tempC6i1, tempC7i1, tempC8i1;

    for( int i = 0; i < m_a32; i += 32 ) {
      float *ai = a+i;
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

      for( int k = 0; k < n_a; k++ ) {
        int k0 = k*m_a;
        tempB1 = _mm_load1_ps(bsel+k0);
        tempA1 = _mm_loadu_ps(ai+k0);
        tempC1i1 += _mm_mul_ps(tempA1, tempB1);

        tempA1 = _mm_loadu_ps(ai4+k0);
        tempC2i1 += _mm_mul_ps(tempA1, tempB1);

        tempA1 = _mm_loadu_ps(ai8+k0);
        tempC3i1 += _mm_mul_ps(tempA1, tempB1);

        tempA1 = _mm_loadu_ps(ai12+k0);
        tempC4i1 += _mm_mul_ps(tempA1, tempB1);

        tempA1 = _mm_loadu_ps(ai16+k0);
        tempC5i1 += _mm_mul_ps(tempA1, tempB1);

        tempA1 = _mm_loadu_ps(ai20+k0);
        tempC6i1 += _mm_mul_ps(tempA1, tempB1);

        tempA1 = _mm_loadu_ps(ai24+k0);
        tempC7i1 += _mm_mul_ps(tempA1, tempB1);

        tempA1 = _mm_loadu_ps(ai28+k0);
        tempC8i1 += _mm_mul_ps(tempA1, tempB1);

      }
      float *ci = c+i;

      _mm_storeu_ps(ci+j1, tempC1i1);
      _mm_storeu_ps(ci+4+j1, tempC2i1);
      _mm_storeu_ps(ci+8+j1, tempC3i1);
      _mm_storeu_ps(ci+12+j1, tempC4i1);
      _mm_storeu_ps(ci+16+j1, tempC5i1);
      _mm_storeu_ps(ci+20+j1, tempC6i1);
      _mm_storeu_ps(ci+24+j1, tempC7i1);
      _mm_storeu_ps(ci+28+j1, tempC8i1);
    }
    for( int i = m_a32; i < m_a16; i += 16) {
      float *ai = a+i;
      float *ai4 = ai + 4;
      float *ai8 = ai4 + 4;
      float *ai12 = ai8 + 4;

      tempC1i1 = _mm_setzero_ps(); // C values begin at 0 anways, so there is no use in loading
      tempC2i1 = _mm_setzero_ps(); // from C.
      tempC3i1 = _mm_setzero_ps();
      tempC4i1 = _mm_setzero_ps();
      for( int k = 0; k < n_a; k++) {
        int k0 = k*m_a;

        tempB1 = _mm_load1_ps(bsel+k0);
        tempA1 = _mm_loadu_ps(ai+k0);
        tempC1i1 += _mm_mul_ps(tempA1, tempB1);

        tempA1 = _mm_loadu_ps(ai4+k0);
        tempC2i1 += _mm_mul_ps(tempA1, tempB1);

        tempA1 = _mm_loadu_ps(ai8+k0);
        tempC3i1 += _mm_mul_ps(tempA1, tempB1);

        tempA1 = _mm_loadu_ps(ai12+k0);
        tempC4i1 += _mm_mul_ps(tempA1, tempB1);
      }
      float *ci = c+i;

      _mm_storeu_ps(ci+j1, tempC1i1);
      _mm_storeu_ps(ci+4+j1, tempC2i1);
      _mm_storeu_ps(ci+8+j1, tempC3i1);
      _mm_storeu_ps(ci+12+j1, tempC4i1);
    }
    for( int i = m_a16; i < m_a4; i += 4) {
      float *ai = a+i;

      tempC1i1 = _mm_setzero_ps();
      for( int k = 0; k < n_a; k++) {
        int k0 = k*m_a;

        tempA1 = _mm_loadu_ps(ai+k0);
        tempB1 = _mm_load1_ps(bsel+k0);
        tempC1i1 += _mm_mul_ps(tempA1, tempB1);
      }
      _mm_storeu_ps(c+i+j1, tempC1i1);
    }
    for( int i = m_a4; i < m_a; i++) {
      float *ai = a+i;

      tempC1i1 = _mm_setzero_ps();
      for( int k = 0; k < n_a; k++) {
        int k0 = k*m_a;

        tempA1 = _mm_loadu_ps(ai+k0);
        tempB1 = _mm_load1_ps(bsel+k0);
        tempC1i1 += _mm_mul_ps(tempA1, tempB1);
      }
      _mm_store_ss(c+i+j1, tempC1i1);
    }
  }
}
