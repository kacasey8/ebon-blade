#include <nmmintrin.h> /* where intrinsics are defined */
#include <stdio.h>
#include <omp.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b)) // A cool function that returns the minimum value given two values
#define BLOCK 64 // blocking i by 64
#define BLOCK2 16 // blocking j by 16, mainly to properly parallelize


void sgemm( int m_a, int n_a, float *A, float *B, float *C ) {

  float *a = A, *b = B, *c = C; // A fragment of a memory from when we attempted padding.

  int m_a4 = m_a/4*4;     // quick references to certain truncated values of m_a
  int m_a16 = m_a/16*16;
  int m_a32 = m_a/32*32;

  __m128 zero = _mm_setzero_ps(); // a reference to the 0 value, appears to copy by value when used
  #pragma omp parallel for
  for( int x = 0; x < m_a; x += BLOCK2) {
    int d = MIN(m_a16, x + BLOCK2); // grabs the size of the block to use, either a block of 16 or up to the end of the array
    int j;
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
    for( int y = 0; y < m_a; y += BLOCK) {
      int e = MIN(m_a32, y+BLOCK); // grabs the size of the block to use, either a block of 32 or up to the end of the array
      for(j = x; j < d; j += 16) { // traverse j by 16
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
        for( int i = y; i < e; i += 32 ) { // traverse i by 32 at a time
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

          int k0 = 0;

          for( int k = 0; k < n_a; k++ ) {
            tempA1 = _mm_loadu_ps(ai+k0);   // load up the 8 A values we will be using
            tempA2 = _mm_loadu_ps(ai4+k0);  // these registers will abuse register blocking
            tempA3 = _mm_loadu_ps(ai8+k0); 
            tempA4 = _mm_loadu_ps(ai12+k0); 
            tempA5 = _mm_loadu_ps(ai16+k0); 
            tempA6 = _mm_loadu_ps(ai20+k0); 
            tempA7 = _mm_loadu_ps(ai24+k0); 
            tempA8 = _mm_loadu_ps(ai28+k0); 

            tempB1 = _mm_load1_ps(bsel+k0);  // load up one B value we will use with each A

            tempC1i1 += _mm_mul_ps(tempA1, tempB1); // store the set up multiplying
            tempC1i2 += _mm_mul_ps(tempA2, tempB1);
            tempC1i3 += _mm_mul_ps(tempA3, tempB1);
            tempC1i4 += _mm_mul_ps(tempA4, tempB1);
            tempC1i5 += _mm_mul_ps(tempA5, tempB1);
            tempC1i6 += _mm_mul_ps(tempA6, tempB1);
            tempC1i7 += _mm_mul_ps(tempA7, tempB1);
            tempC1i8 += _mm_mul_ps(tempA8, tempB1);

            tempB1 = _mm_load1_ps(bsel+1+k0); // use a new B value

            tempC2i1 += _mm_mul_ps(tempA1, tempB1); // and store into another level of tempC
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

            k0 += m_a; // setup k0 for next iteration
          }

          float *ci = c + i;

          _mm_storeu_ps(ci+j1, tempC1i1); // store each level
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


          ci += 4; // modify the pointer for easy storing

          _mm_storeu_ps(ci+j1, tempC1i2); // store next level
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
      if(j == m_a16) { // edge case, covers when m_a is not a multiple of 16 for the j loop, calculates up to 15 right columns of C
        for(int j = m_a16; j < m_a4; j += 4) { // same logic as before, traverse by only 4 for j this time.
          float *bsel = b + j;
          int j1 = j*m_a;
          int j2 = j1 + m_a;
          int j3 = j2 + m_a;
          int j4 = j3 + m_a;
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

        for(int j = m_a4; j < m_a; j++) { // same logic as before, traverse j by only 1 this time, hit last columns of C
          float *bsel = b + j;
          int j1 = j*m_a;
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

  // edge case, covers when m_a is not a multiple of 32 for the i loop, calculates up to 31 bottom rows of C
  #pragma omp parallel for
  for( int j = 0; j < m_a4; j += 4) { // traverse j at a smaller rate, to avoid massive fringe casing
    float *bsel = b+j;
    int j1 = j*m_a;
    int j2 = j1 + m_a;
    int j3 = j2 + m_a;
    int j4 = j3 + m_a;

    __m128 tempA1, tempA2, tempA3, tempA4;
    __m128 tempB1;
    __m128 tempC1i1, tempC2i1, tempC3i1, tempC4i1;
    __m128 tempC1i2, tempC2i2, tempC3i2, tempC4i2;
    __m128 tempC1i3, tempC2i3, tempC3i3, tempC4i3;
    __m128 tempC1i4, tempC2i4, tempC3i4, tempC4i4;

    for( int i = m_a32; i < m_a16; i += 16) { // try hitting the last 16 rows if not calcualted yet.
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
      for( int k = 0; k < n_a; k++) {
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
    for( int i = m_a16; i < m_a4; i += 4) { // try hitting the last groups of 4 rows of C if they aren't hit yet
      float *ai = a+i;

      tempC1i1 = zero;
      tempC2i1 = zero;
      tempC3i1 = zero;
      tempC4i1 = zero;
      for( int k = 0; k < n_a; k++) {
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
      }
      _mm_storeu_ps(c+i+j1, tempC1i1);
      _mm_storeu_ps(c+i+j2, tempC2i1);
      _mm_storeu_ps(c+i+j3, tempC3i1);
      _mm_storeu_ps(c+i+j4, tempC4i1);
    }
    for( int i = m_a4; i < m_a; i++) { // hit up to last 3 rows of C if they aren't hit yet
      float *ai = a+i;

      tempC1i1 = zero;
      tempC2i1 = zero;
      tempC3i1 = zero;
      tempC4i1 = zero;
      for( int k = 0; k < n_a; k++) {
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
      }
      _mm_store_ss(c+i+j1, tempC1i1);
      _mm_store_ss(c+i+j2, tempC2i1);
      _mm_store_ss(c+i+j3, tempC3i1);
      _mm_store_ss(c+i+j4, tempC4i1);
    }
  }
  for( int j = m_a4; j < m_a; j++) { // traverse j by 1. This allows us to get the bottom right corner of C calculated
    float *bsel = b+j;
    int j1 = j*m_a;

    __m128 tempA1;
    __m128 tempB1;
    __m128 tempC1i1, tempC2i1, tempC3i1, tempC4i1;

    for( int i = m_a32; i < m_a16; i += 16) { // try hitting the last 16 rows if not calcualted yet.
      float *ai = a+i;
      float *ai4 = ai + 4;
      float *ai8 = ai4 + 4;
      float *ai12 = ai8 + 4;

      tempC1i1 = zero; // C values begin at 0 anways, so there is no use in loading
      tempC2i1 = zero; // from C.
      tempC3i1 = zero;
      tempC4i1 = zero;
      for( int k = 0; k < n_a; k++) {
        int k0 = k*m_a;

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
    for( int i = m_a16; i < m_a4; i += 4) { // try hitting the last 4 rows if not calcualted yet.
      float *ai = a+i;

      tempC1i1 = zero;
      for( int k = 0; k < n_a; k++) {
        int k0 = k*m_a;

        tempA1 = _mm_loadu_ps(ai+k0);
        tempB1 = _mm_load1_ps(bsel+k0);
        tempC1i1 += _mm_mul_ps(tempA1, tempB1);
      }
      _mm_storeu_ps(c+i+j1, tempC1i1);
    }
    for( int i = m_a4; i < m_a; i++) { // hit the very last corner...looks like array is completed
      float *ai = a+i;

      tempC1i1 = zero;
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
