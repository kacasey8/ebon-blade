#include <nmmintrin.h> /* where intrinsics are defined */
#include <stdio.h>
#include <omp.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))

void sgemm( int m_a, int n_a, float *A, float *B, float *C ) {

  float *a, *b, *c;

  int m_a4 = m_a/4*4, m_diff = 0; 
  int n_a4 = n_a/4*4, n_diff = 0;

  if( m_a4 != m_a ){ // if matrices need padding, m_a4 and n_a4 become the
    m_a4 += 4;       // lowest multiple of 4 greater than m_a and n_a
    m_diff = m_a4 - m_a;
  }
  if( n_a4 != n_a ){
    n_a4 += 4;
    n_diff = n_a4 - n_a;
  }

  if( m_diff || n_diff ){
    a = calloc(m_a4*n_a4, sizeof(float));
    b = calloc(m_a4*n_a4, sizeof(float));
    if( m_diff ){ // c only needs to be padded if m_a isn't a multiple of 4
      c = malloc(m_a4*m_a4*sizeof(float));
    }else {
      c = C;
    }

    for( int i = 0; i < n_a; i++ ){ // moves the values of A and B into a and b
      for( int j = 0; j < m_a; j++ ){
        *(a + i*m_a4 + j) = *(A + i*m_a + j);
        *(b + i*m_a4 + j) = *(B + i*m_a + j);
      }
    }
  } else {
    a = A;
    b = B;
    c = C;
  }

  // e, y
  // j, b, bsel
  // j0, j1, j2, j3
  // i, a, ai, c, ci
  // k, k0, k2
  // n_a4, m_a4

  int blocksize = 64;

  #pragma omp parallel for
  for( int y = 0; y < m_a4; y += blocksize) {
    int e = MIN(m_a4, y+blocksize);
    __m128 tempA1, tempA2, tempA3, tempA4;
    __m128 tempB1;
    __m128 tempC1i1, tempC2i1, tempC3i1, tempC4i1;
    __m128 tempC5i1, tempC6i1, tempC7i1, tempC8i1;
    __m128 tempC1i2, tempC2i2, tempC3i2, tempC4i2;
    __m128 tempC5i2, tempC6i2, tempC7i2, tempC8i2;
    __m128 tempC1i3, tempC2i3, tempC3i3, tempC4i3;
    __m128 tempC5i3, tempC6i3, tempC7i3, tempC8i3;
    __m128 tempC1i4, tempC2i4, tempC3i4, tempC4i4;
    __m128 tempC5i4, tempC6i4, tempC7i4, tempC8i4;
    for(int j = 0; j < m_a4; j += 8) {
      float *bsel = b + j;
      int j1 = j*m_a4;
      int j2 = j1 + m_a4;
      int j3 = j2 + m_a4;
      int j4 = j3 + m_a4;
      int j5 = j4 + m_a4;
      int j6 = j5 + m_a4;
      int j7 = j6 + m_a4;
      int j8 = j7 + m_a4;
      for( int i = y; i < e; i += 16 ) {
        float *ai = a + i;
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
        tempC1i2 = _mm_setzero_ps(); 
        tempC2i2 = _mm_setzero_ps(); 
        tempC3i2 = _mm_setzero_ps(); 
        tempC4i2 = _mm_setzero_ps(); 
        tempC5i2 = _mm_setzero_ps(); 
        tempC6i2 = _mm_setzero_ps(); 
        tempC7i2 = _mm_setzero_ps(); 
        tempC8i2 = _mm_setzero_ps(); 
        tempC1i3 = _mm_setzero_ps(); 
        tempC2i3 = _mm_setzero_ps(); 
        tempC3i3 = _mm_setzero_ps(); 
        tempC4i3 = _mm_setzero_ps(); 
        tempC5i3 = _mm_setzero_ps(); 
        tempC6i3 = _mm_setzero_ps(); 
        tempC7i3 = _mm_setzero_ps(); 
        tempC8i3 = _mm_setzero_ps(); 
        tempC1i4 = _mm_setzero_ps(); 
        tempC2i4 = _mm_setzero_ps(); 
        tempC3i4 = _mm_setzero_ps(); 
        tempC4i4 = _mm_setzero_ps(); 
        tempC5i4 = _mm_setzero_ps(); 
        tempC6i4 = _mm_setzero_ps(); 
        tempC7i4 = _mm_setzero_ps(); 
        tempC8i4 = _mm_setzero_ps(); 

        for( int k = 0; k < n_a4; k++ ) {
          int k0 = k*m_a4;

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


        ci += 4;

        _mm_storeu_ps(ci+j1, tempC1i2);
        _mm_storeu_ps(ci+j2, tempC2i2);
        _mm_storeu_ps(ci+j3, tempC3i2);
        _mm_storeu_ps(ci+j4, tempC4i2);
        _mm_storeu_ps(ci+j5, tempC5i2);
        _mm_storeu_ps(ci+j6, tempC6i2);
        _mm_storeu_ps(ci+j7, tempC7i2);
        _mm_storeu_ps(ci+j8, tempC8i2);

        ci += 4;

        _mm_storeu_ps(ci+j1, tempC1i3);
        _mm_storeu_ps(ci+j2, tempC2i3);
        _mm_storeu_ps(ci+j3, tempC3i3);
        _mm_storeu_ps(ci+j4, tempC4i3);
        _mm_storeu_ps(ci+j5, tempC5i3);
        _mm_storeu_ps(ci+j6, tempC6i3);
        _mm_storeu_ps(ci+j7, tempC7i3);
        _mm_storeu_ps(ci+j8, tempC8i3);

        ci += 4;

        _mm_storeu_ps(ci+j1, tempC1i4);
        _mm_storeu_ps(ci+j2, tempC2i4);
        _mm_storeu_ps(ci+j3, tempC3i4);
        _mm_storeu_ps(ci+j4, tempC4i4);
        _mm_storeu_ps(ci+j5, tempC5i4);
        _mm_storeu_ps(ci+j6, tempC6i4);
        _mm_storeu_ps(ci+j7, tempC7i4);
        _mm_storeu_ps(ci+j8, tempC8i4);
      }
    }
  }
  //C[i+j*m_a] += A[i+k*m_a] * B[j+k*m_a];

  if( m_diff || n_diff ){
    free(a); // frees allocated matrices
    free(b); // a and b don't change, so no need to move it back
    if( m_diff ){ // unpads c
      for(int i = 0; i < m_a; i++){
        for(int j = 0; j < m_a; j++){
          *(C + i*m_a + j) = *(c + i*m_a4 + j);
        }
      }
      free(c);
    }
  }

}