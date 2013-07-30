#include <nmmintrin.h> /* where intrinsics are defined */
#include <stdio.h>
#include <omp.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))

void sgemm( int m_a, int n_a, float *A, float *B, float *C ) {
  /*__m128 tempA1, tempA2, tempA3, tempA4;
  __m128 tempB1, tempB2, tempB3, tempB4;
  __m128 tempC1, tempC2, tempC3, tempC4;*/

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
      c = calloc(m_a4*m_a4, sizeof(float));
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

  int blocksize = 64;

  #pragma omp parallel for
  for( int z = 0; z < m_a4; z += blocksize) {
    __m128 tempA1, tempA2, tempA3, tempA4;
    __m128 tempB1, tempB2, tempB3, tempB4;
    __m128 tempC1, tempC2, tempC3, tempC4;
    for(int y = 0; y < m_a4; y += blocksize) {
      int d = MIN(m_a4, y+blocksize);
      int e = MIN(m_a4, z+blocksize);
      for( int j = y; j < d; j += 4 ) {
        for( int i = z; i < e; i += 4 ) {
          tempC1 = _mm_setzero_ps(); // C values begin at 0 anways, so there is no use in loading
          tempC2 = _mm_setzero_ps(); // from C.
          tempC3 = _mm_setzero_ps();
          tempC4 = _mm_setzero_ps();

          for( int k = 0; k < n_a4; k += 4 ) {
            tempA1 = _mm_loadu_ps(a+i+k*m_a4); // load the 16 values for A we will be using in this loop
            tempA2 = _mm_loadu_ps(a+i+(k+1)*m_a4);
            tempA3 = _mm_loadu_ps(a+i+(k+2)*m_a4);
            tempA4 = _mm_loadu_ps(a+i+(k+3)*m_a4);

            tempB1 = _mm_load1_ps(b+j+k*m_a4); // load 4 B values [1, 1, 1, 1]
            tempB2 = _mm_load1_ps(b+j+(k+1)*m_a4); // [5, 5, 5, 5]
            tempB3 = _mm_load1_ps(b+j+(k+2)*m_a4); // [9, 9, 9, 9]
            tempB4 = _mm_load1_ps(b+j+(k+3)*m_a4); // [13, 13, 13, 13]

            tempC1 += _mm_mul_ps(tempA1, tempB1); // use the 16 A values, 4 B values
            tempC1 += _mm_mul_ps(tempA2, tempB2);
            tempC1 += _mm_mul_ps(tempA3, tempB3);
            tempC1 += _mm_mul_ps(tempA4, tempB4);

            tempB1 = _mm_load1_ps(b+j+1+k*m_a4); // load 4 new B values [2, 2, 2, 2]
            tempB2 = _mm_load1_ps(b+j+1+(k+1)*m_a4); // [6, 6, 6, 6]
            tempB3 = _mm_load1_ps(b+j+1+(k+2)*m_a4); // [10, 10, 10, 10]
            tempB4 = _mm_load1_ps(b+j+1+(k+3)*m_a4); // [14, 14, 14, 14]

            tempC2 += _mm_mul_ps(tempA1, tempB1);
            tempC2 += _mm_mul_ps(tempA2, tempB2);
            tempC2 += _mm_mul_ps(tempA3, tempB3);
            tempC2 += _mm_mul_ps(tempA4, tempB4);

            tempB1 = _mm_load1_ps(b+j+2+k*m_a4); // load 4 new B values [3, 3, 3, 3]
            tempB2 = _mm_load1_ps(b+j+2+(k+1)*m_a4); // [7, 7, 7, 7]
            tempB3 = _mm_load1_ps(b+j+2+(k+2)*m_a4); // [11, 11, 11, 11]
            tempB4 = _mm_load1_ps(b+j+2+(k+3)*m_a4); // ]15, 15, 15, 15]

            /*tempC3 += _mm_mul_ps(tempA1, tempB1);
            tempC3 += _mm_mul_ps(tempA2, tempB2);
            tempC3 += _mm_mul_ps(tempA3, tempB3);
            tempC3 += _mm_mul_ps(tempA4, tempB4);*/

            tempC3 =        // a nested function call, does what
            _mm_add_ps(     // 4 lines commented code above does
              tempC3,       // seems to generate amazing speed
              _mm_add_ps(   // increase (1 Gflop/s)
                _mm_mul_ps(tempA1, tempB1),
                _mm_add_ps(
                  _mm_mul_ps(tempA2, tempB2),
                  _mm_add_ps(
                    _mm_mul_ps(tempA3, tempB3),
                    _mm_mul_ps(tempA4, tempB4)))));

            tempB1 = _mm_load1_ps(b+j+3+k*m_a4); // load 4 new B values [4, 4, 4, 4]
            tempB2 = _mm_load1_ps(b+j+3+(k+1)*m_a4); // [8, 8, 8, 8]
            tempB3 = _mm_load1_ps(b+j+3+(k+2)*m_a4); // [12, 12, 12, 12]
            tempB4 = _mm_load1_ps(b+j+3+(k+3)*m_a4); // [16, 16, 16, 16]

            tempC4 += _mm_mul_ps(tempA1, tempB1);
            tempC4 += _mm_mul_ps(tempA2, tempB2);
            tempC4 += _mm_mul_ps(tempA3, tempB3);
            tempC4 += _mm_mul_ps(tempA4, tempB4);
          }
          _mm_storeu_ps(c+i+j*m_a4, tempC1); // stores back the 16 C values we calculated in the above loop
          _mm_storeu_ps(c+i+(j+1)*m_a4, tempC2);
          _mm_storeu_ps(c+i+(j+2)*m_a4, tempC3);
          _mm_storeu_ps(c+i+(j+3)*m_a4, tempC4);
        }
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
