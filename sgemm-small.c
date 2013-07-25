#include <nmmintrin.h> /* where intrinsics are defined */
#include <stdio.h>

// Autogenerated, do not edit. All changes will be undone.
void printstuff( int m_a, int n_a, float *A, float *B, float *C ) {
  printf("\nA:\n");
  for( int i = 0; i < m_a * n_a; i++ ){
    printf("%f ", A[i]);
    if(i%m_a == m_a-1) printf("\n");
  }

  printf("\nB:\n");
  for( int i = 0; i < m_a * n_a; i++ ){
    printf("%f ", B[i]);
    if(i%m_a == m_a-1) printf("\n");
  }

  printf("\nC:\n");
  for( int i = 0; i < m_a * m_a; i++ ){
    printf("%f ", C[i]);
    if(i%m_a == m_a-1) printf("\n");
  }
}

void sgemm( int m_a, int n_a, float *A, float *B, float *C ) {
  __m128 tempA1, tempA2, tempA3, tempA4;
  __m128 tempB1, tempB2, tempB3, tempB4;
  __m128 tempC1, tempC2, tempC3, tempC4;

  float *a, *b, *c;

  //__m128 B1, B2, B3, B4;
  int m_a4 = m_a/4*4, m_diff=0;
  int n_a4 = n_a/4*4, n_diff=0;

  //printf("\nfirst");
  //printstuff( m_a, n_a, A, B, C);

  if( m_a4 != m_a ){
    m_a4 += 4;
    m_diff = m_a4 - m_a;
  }
  if( n_a4 != m_a ){
    n_a4 += 4;
    n_diff = n_a4 - n_a;
  }

  if( m_diff || n_diff ){
    a = calloc(m_a4*n_a4, sizeof(float));
    b = calloc(m_a4*n_a4, sizeof(float));
    if( m_diff ){
      c = calloc(m_a4*m_a4, sizeof(float));
    }else{
      c = C;
    }

    for(int i = 0; i < n_a; i++){
      for(int j = 0; j < m_a; j++){
        *(a + i*m_a4 + j) = *(A + i*m_a + j);
        *(b + i*m_a4 + j) = *(B + i*m_a + j);
      }
    }
  }else{
    a = A;
    b = B;
    c = C;
  }

  //printf("\nsecond");
  //printstuff( m_a4, n_a4, a, b, c);

  for( int j = 0; j < m_a4; j += 4 ) {
    for( int i = 0; i < m_a4; i += 4 ) {
      /*
      tempC1 = _mm_loadu_ps(c+i+j*m_a); // load the 16 C values we will be using in the next loop
      tempC2 = _mm_loadu_ps(c+i+(j+1)*m_a);
      tempC3 = _mm_loadu_ps(c+i+(j+2)*m_a);
      tempC4 = _mm_loadu_ps(c+i+(j+3)*m_a);
      */
      tempC1 = _mm_setzero_ps();
      tempC2 = _mm_setzero_ps();
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

        tempB1 = _mm_load1_ps(b+j+1+k*m_a4); // load 4 new B values
        tempB2 = _mm_load1_ps(b+j+1+(k+1)*m_a4);
        tempB3 = _mm_load1_ps(b+j+1+(k+2)*m_a4);
        tempB4 = _mm_load1_ps(b+j+1+(k+3)*m_a4);

        tempC2 += _mm_mul_ps(tempA1, tempB1);
        tempC2 += _mm_mul_ps(tempA2, tempB2);
        tempC2 += _mm_mul_ps(tempA3, tempB3);
        tempC2 += _mm_mul_ps(tempA4, tempB4);

        tempB1 = _mm_load1_ps(b+j+2+k*m_a4);
        tempB2 = _mm_load1_ps(b+j+2+(k+1)*m_a4);
        tempB3 = _mm_load1_ps(b+j+2+(k+2)*m_a4);
        tempB4 = _mm_load1_ps(b+j+2+(k+3)*m_a4);

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

        tempB1 = _mm_load1_ps(b+j+3+k*m_a4);
        tempB2 = _mm_load1_ps(b+j+3+(k+1)*m_a4);
        tempB3 = _mm_load1_ps(b+j+3+(k+2)*m_a4);
        tempB4 = _mm_load1_ps(b+j+3+(k+3)*m_a4);

        tempC4 += _mm_mul_ps(tempA1, tempB1);
        tempC4 += _mm_mul_ps(tempA2, tempB2);
        tempC4 += _mm_mul_ps(tempA3, tempB3);
        tempC4 += _mm_mul_ps(tempA4, tempB4);

        /*tempC4 =         // a replica of the mod made on tempC3
        _mm_add_ps(      // appears to conteract the 
          tempC4,        // benefit
          _mm_add_ps(
            _mm_mul_ps(tempA1, tempB1), 
            _mm_add_ps(
              _mm_mul_ps(tempA2, tempB2),
              _mm_add_ps(
                _mm_mul_ps(tempA3, tempB3), 
                _mm_mul_ps(tempA4, tempB4)))));*/
      }
      _mm_storeu_ps(c+i+j*m_a4, tempC1); // store the 16 C values we calculated in the above loop
      _mm_storeu_ps(c+i+(j+1)*m_a4, tempC2);
      _mm_storeu_ps(c+i+(j+2)*m_a4, tempC3);
      _mm_storeu_ps(c+i+(j+3)*m_a4, tempC4);
    }
  }

  //printf("\nthird");
  //printstuff( m_a4, n_a4, a, b, c);

  if( m_diff || n_diff ){
    for(int i = 0; i < n_a; i++){
      for(int j = 0; j < m_a; j++){
        *(A + i*m_a + j) = *(a + i*m_a4 + j);
        *(B + i*m_a + j) = *(b + i*m_a4 + j);
      }
    }
    free(a);
    free(b);
    if( m_diff ){
      for(int i = 0; i < m_a; i++){
        for(int j = 0; j < m_a; j++){
          *(C + i*m_a + j) = *(c + i*m_a4 + j);
        }
      }
      free(c);
    }
  }

  //printf("\nfourth");
  //printstuff( m_a, n_a, A, B, C);

}


