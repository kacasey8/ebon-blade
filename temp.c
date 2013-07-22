#include <emmintrin.h> /* where intrinsics are defined */

void sgemm( int m_a, int n_a, float *A, float *B, float *C ) {
  __m128 tempA, tempB, tempC;
  __m128 tempB1, tempB2, tempB3, tempB4;
  int m_a4 = m_a/4*4;
  int m_a16 = m_a/16*16;
  if (m_a4 == m_a) {
    for( int k = 0; k < n_a; k++ ) {
      for( int j = 0; j < m_a; j++ ) {
        tempB = _mm_load1_ps(B+j+k*m_a); // load B[j + k*m_a] into all 4
        for( int i = 0; i < m_a16; i += 16 ) {
          tempA = _mm_loadu_ps(A+i+k*m_a); // load A[i + k*m_a], A[i+1+k*m_a], etc.
          tempC = _mm_loadu_ps(C+i+j*m_a); // load C[i+j*m_a)], C[i+1+j*m_a], etc.
          _mm_storeu_ps(C+i+j*m_a, _mm_add_ps(tempC, _mm_mul_ps(tempA, tempB))); //C[i+j*m_a] += A[i+k*m_a] * B[j+k*m_a];
          tempA = _mm_loadu_ps(A+i+4+k*m_a);
          tempC = _mm_loadu_ps(C+i+4+j*m_a);
          _mm_storeu_ps(C+i+4+j*m_a, _mm_add_ps(tempC, _mm_mul_ps(tempA, tempB)));
          tempA = _mm_loadu_ps(A+i+8+k*m_a);
          tempC = _mm_loadu_ps(C+i+8+j*m_a);
          _mm_storeu_ps(C+i+8+j*m_a, _mm_add_ps(tempC, _mm_mul_ps(tempA, tempB)));
          tempA = _mm_loadu_ps(A+i+12+k*m_a);
          tempC = _mm_loadu_ps(C+i+12+j*m_a);
          _mm_storeu_ps(C+i+12+j*m_a, _mm_add_ps(tempC, _mm_mul_ps(tempA, tempB)));
        }
        for (int i = m_a16; i < m_a; i += 4)
        {
          tempA = _mm_loadu_ps(A+i+k*m_a);
          tempC = _mm_loadu_ps(C+i+j*m_a);
          _mm_storeu_ps(C+i+j*m_a, _mm_add_ps(tempC, _mm_mul_ps(tempA, tempB)));
        }
      }
    }
  } else {
    for( int k = 0; k < n_a; k++ ) {
      for( int j = 0; j < m_a; j++ ) {
        tempB = _mm_load1_ps(B+j+k*m_a); // load B[j + k*m_a] into all 4
        for( int i = 0; i < m_a4; i += 4 ) {
          tempA = _mm_loadu_ps(A+i+k*m_a); // load A[i + k*m_a], A[i+1+k*m_a],
          tempC = _mm_loadu_ps(C+i+j*m_a); // load C[i+j*m_a)], C[i+1+j*m_a], etc.
          _mm_storeu_ps(C+i+j*m_a, _mm_add_ps(tempC, _mm_mul_ps(tempA, tempB))); 
          //C[i+j*m_a] += A[i+k*m_a] * B[j+k*m_a];
        }
        for( int i = m_a4; i < m_a; i++) {
          C[i+j*m_a] += A[i+k*m_a] * B[j+k*m_a];
        }
      }
    }
  }
}
