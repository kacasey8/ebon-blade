#include <emmintrin.h> /* where intrinsics are defined */
#include <stdio.h>

void sgemm( int m_a, int n_a, float *A, float *B, float *C ) {
	__m128 tempA, tempB, tempC;
	__m128 tempB1, tempB2, tempB3, tempB4;
  __m128 tempA1, tempA2, tempA3, tempA4;
  __m128 tempC1, tempC2, tempC3, tempC4;
	int m_a4 = m_a/4*4;
	int n_a4 = n_a/4*4;
	if (m_a4 == m_a && n_a4 == n_a) {
		for( int j = 0; j < m_a; j += 4 ) {
	    for( int i = 0; i < m_a4; i += 4 ) {
        tempC1 = _mm_loadu_ps(C+i+j*m_a); // load C[i+j*m_a)], C[i+1+j*m_a], etc.
        tempC2 = _mm_loadu_ps(C+i+(j+1)*m_a);
        tempC3 = _mm_loadu_ps(C+i+(j+2)*m_a);
        tempC4 = _mm_loadu_ps(C+i+(j+3)*m_a);
    	  for( int k = 0; k < n_a4; k += 4 ) {
          tempA1 = _mm_loadu_ps(A+i+k*m_a);
          tempA2 = _mm_loadu_ps(A+i+(k+1)*m_a);
          tempA3 = _mm_loadu_ps(A+i+(k+2)*m_a);
          tempA4 = _mm_loadu_ps(A+i+(k+3)*m_a);

          tempB1 = _mm_load1_ps(B+j+k*m_a);
          tempB2 = _mm_load1_ps(B+j+(k+1)*m_a); 
          tempB3 = _mm_load1_ps(B+j+(k+2)*m_a); 
          tempB4 = _mm_load1_ps(B+j+(k+3)*m_a); 

          tempC1 = _mm_add_ps(tempC1, _mm_mul_ps(tempA1, tempB1));
          tempC1 = _mm_add_ps(tempC1, _mm_mul_ps(tempA2, tempB2));
          tempC1 = _mm_add_ps(tempC1, _mm_mul_ps(tempA3, tempB3));
          tempC1 = _mm_add_ps(tempC1, _mm_mul_ps(tempA4, tempB4));

          tempB1 = _mm_load1_ps(B+j+1+k*m_a);
          tempB2 = _mm_load1_ps(B+j+1+(k+1)*m_a); 
          tempB3 = _mm_load1_ps(B+j+1+(k+2)*m_a); 
          tempB4 = _mm_load1_ps(B+j+1+(k+3)*m_a); 

          tempC2 = _mm_add_ps(tempC2, _mm_mul_ps(tempA1, tempB1));
          tempC2 = _mm_add_ps(tempC2, _mm_mul_ps(tempA2, tempB2));
          tempC2 = _mm_add_ps(tempC2, _mm_mul_ps(tempA3, tempB3));
          tempC2 = _mm_add_ps(tempC2, _mm_mul_ps(tempA4, tempB4));

          tempB1 = _mm_load1_ps(B+j+2+k*m_a);
          tempB2 = _mm_load1_ps(B+j+2+(k+1)*m_a); 
          tempB3 = _mm_load1_ps(B+j+2+(k+2)*m_a); 
          tempB4 = _mm_load1_ps(B+j+2+(k+3)*m_a); 

          tempC3 = _mm_add_ps(tempC3, _mm_mul_ps(tempA1, tempB1));
          tempC3 = _mm_add_ps(tempC3, _mm_mul_ps(tempA2, tempB2));
          tempC3 = _mm_add_ps(tempC3, _mm_mul_ps(tempA3, tempB3));
          tempC3 = _mm_add_ps(tempC3, _mm_mul_ps(tempA4, tempB4));

          tempB1 = _mm_load1_ps(B+j+3+k*m_a);
          tempB2 = _mm_load1_ps(B+j+3+(k+1)*m_a); 
          tempB3 = _mm_load1_ps(B+j+3+(k+2)*m_a); 
          tempB4 = _mm_load1_ps(B+j+3+(k+3)*m_a); 

          tempC4 = _mm_add_ps(tempC4, _mm_mul_ps(tempA1, tempB1));
          tempC4 = _mm_add_ps(tempC4, _mm_mul_ps(tempA2, tempB2));
          tempC4 = _mm_add_ps(tempC4, _mm_mul_ps(tempA3, tempB3));
          tempC4 = _mm_add_ps(tempC4, _mm_mul_ps(tempA4, tempB4));

          /*tempB = _mm_load1_ps(B+j+k*m_a); // load B[j + k*m_a] into all 4
          tempA = _mm_loadu_ps(A+i+k*m_a); // load A[i + k*m_a], A[i+1+k*m_a], etc.
          tempC = _mm_add_ps(tempC, _mm_mul_ps(tempA, tempB));

          tempB = _mm_load1_ps(B+j+(k+1)*m_a); 
          tempA = _mm_loadu_ps(A+i+(k+1)*m_a);
          tempC = _mm_add_ps(tempC, _mm_mul_ps(tempA, tempB));

          tempB = _mm_load1_ps(B+j+(k+2)*m_a); 
          tempA = _mm_loadu_ps(A+i+(k+2)*m_a);
          tempC = _mm_add_ps(tempC, _mm_mul_ps(tempA, tempB));

          tempB = _mm_load1_ps(B+j+(k+3)*m_a); 
          tempA = _mm_loadu_ps(A+i+(k+3)*m_a);
          tempC = _mm_add_ps(tempC, _mm_mul_ps(tempA, tempB));*/
    	  }
        _mm_storeu_ps(C+i+j*m_a, tempC1);
        _mm_storeu_ps(C+i+(j+1)*m_a, tempC2);
        _mm_storeu_ps(C+i+(j+2)*m_a, tempC3);
        _mm_storeu_ps(C+i+(j+3)*m_a, tempC4);
    	}
    }
	} else {
    printf("Else case\n");
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
