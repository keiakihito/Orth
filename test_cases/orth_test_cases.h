#ifndef ORTH_TEST_CASES_H
#define ORTH_TEST_CASES_H


#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

#include "../includes/helper_debug.h"
#include "../includes/helper_cuda.h"
#include "../includes/helper_functions.h"


#define CHECK(call){ \
    const cudaError_t cuda_ret = call; \
    if(cuda_ret != cudaSuccess){ \
        printf("Error: %s:%d,  ", __FILE__, __LINE__ );\
        printf("code: %d, reason: %s \n", cuda_ret, cudaGetErrorString(cuda_ret));\
        exit(-1); \
    }\
}

//SVD_Decompostion test cases
void SVD_Decomp_Case1()
{
    /*
                |1.0 1.0|
        mtxA =  |0.0 1.0|
                |1.0 0.0|

        expected
        mtxU = |0.8165  0      |
               |0.4082  0.7071 |
               |0.4082  -0.7071|

        Singular value = | 1.7321 |
                         | 1.00   |

        mtxVT = |-0.7071  0.7071|
                |0.7071   0.7071|

    */

    // Define the dense matrixB column major
    float mtxA[] = { 1.0, 0.0, 1.0, 1.0, 1.0, 0.0};

    //For SVD functions in cuSolver
    const int ROW_A = 3;
    const int COL_A = 2;
    const int LD_A = 3;

    float *mtxA_d = NULL;
    float *mtxU_d = NULL;
    float *sngVals_d = NULL; // Singular values
    float *mtxV_d = NULL; // Need to allocate a separate memory buffer for the transposed matrix.
    float *mtxVT_d = NULL;


    bool debug = true;


    //(1) Allocate device memory
    CHECK((cudaMalloc((void**)&mtxA_d, ROW_A * COL_A *sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxU_d, LD_A * COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&sngVals_d, COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxVT_d, COL_A * COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxV_d, COL_A * COL_A * sizeof(float))));


    //(2) Copy value to device
    CHECK((cudaMemcpy(mtxA_d, mtxA, ROW_A * COL_A * sizeof(float), cudaMemcpyHostToDevice)));
    
    if(debug){
        printf("\n\n~~~MtxA~~~\n");
        print_mtx_clm_d(mtxA_d, ROW_A, COL_A);
    }




    //(3) Create handler
    cusolverDnHandle_t cusolverHandler = NULL;
    cublasHandle_t cublasHandler = NULL;

    checkCudaErrors(cusolverDnCreate(&cusolverHandler));
    checkCudaErrors(cublasCreate(&cublasHandler));



    // //(4) Compute SVD decomposition
    // checkCudaErrors(cusolverDnSgesvd(cusolverHandler, jobU, jobVT, ROW_A, COL_A, mtxA_d, LD_A, sngVals_d, mtxU_d,LD_A, mtxVT_d, COL_A, work_d, lwork, rwork_d, devInfo));
    SVD_Decmp(cusolverHandler, ROW_A, COL_A, LD_A, mtxA_d, mtxU_d, sngVals_d, mtxVT_d);

    //Transpose mmatrix V transpose to get matrix V 
    //Checking the result with MATLAB
    const float alpha = 1.0f;
    const float beta = 0.0f;

    //Transpose mtxVT
    //Matrix VT and matrix V are both n by n.
    checkCudaErrors(cublasSgeam(cublasHandler, CUBLAS_OP_T, CUBLAS_OP_N, COL_A, COL_A, &alpha, mtxVT_d, COL_A, &beta, mtxVT_d, COL_A, mtxV_d, COL_A));



    if(debug){
        printf("\n\n👀Result of  SVD Decomp function👀\n\n");
        printf("\n\n~~mtxU_d\n");
        print_mtx_clm_d(mtxU_d, ROW_A, COL_A);
        printf("\n\n~~mtxD_d\n");
        print_mtx_clm_d(sngVals_d, COL_A, 1);
        printf("\n\n~~mtxV_d\n");
        print_mtx_clm_d(mtxV_d, COL_A, COL_A);
    }


    checkCudaErrors(cusolverDnDestroy(cusolverHandler));
    checkCudaErrors(cublasDestroy(cublasHandler));

    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(mtxU_d));
    CHECK(cudaFree(sngVals_d));
    CHECK(cudaFree(mtxVT_d));
    CHECK(cudaFree(mtxV_d));

    printf("\n= = = End of Case  = = = \n\n");
} // end of case 1


void SVD_Decomp_Case2()
{
    // Define the dense matrixB column major
    float mtxA[] = {
    1.1, 0.8, 3.0, 2.2,
    2.2, 1.6, 4.1, 3.3,
    3.3, 2.4, 5.2, 4.4,
    4.4, 3.2, 6.3, 5.5
    };



    //For SVD functions in cuSolver
    const int N = 4; // N by N square matrix 
    const int ROW_A = N;
    const int COL_A = N;
    const int LD_A = N;
  float *mtxA_d = NULL;
    float *mtxU_d = NULL;
    float *sngVals_d = NULL; // Singular values
    float *mtxV_d = NULL; // Need to allocate a separate memory buffer for the transposed matrix.
    float *mtxVT_d = NULL;


    bool debug = true;


    //(1) Allocate device memory
    CHECK((cudaMalloc((void**)&mtxA_d, ROW_A * COL_A *sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxU_d, LD_A * COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&sngVals_d, COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxVT_d, COL_A * COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxV_d, COL_A * COL_A * sizeof(float))));


    //(2) Copy value to device
    CHECK((cudaMemcpy(mtxA_d, mtxA, ROW_A * COL_A * sizeof(float), cudaMemcpyHostToDevice)));
    
    if(debug){
        printf("\n\n~~~MtxA~~~\n");
        print_mtx_clm_d(mtxA_d, ROW_A, COL_A);
    }




    //(3) Create handler
    cusolverDnHandle_t cusolverHandler = NULL;
    cublasHandle_t cublasHandler = NULL;

    checkCudaErrors(cusolverDnCreate(&cusolverHandler));
    checkCudaErrors(cublasCreate(&cublasHandler));



    // //(4) Compute SVD decomposition
    // checkCudaErrors(cusolverDnSgesvd(cusolverHandler, jobU, jobVT, ROW_A, COL_A, mtxA_d, LD_A, sngVals_d, mtxU_d,LD_A, mtxVT_d, COL_A, work_d, lwork, rwork_d, devInfo));
    SVD_Decmp(cusolverHandler, ROW_A, COL_A, LD_A, mtxA_d, mtxU_d, sngVals_d, mtxVT_d);

    //Transpose mmatrix V transpose to get matrix V 
    //Checking the result with MATLAB
    const float alpha = 1.0f;
    const float beta = 0.0f;

    //Transpose mtxVT
    //Matrix VT and matrix V are both n by n.
    checkCudaErrors(cublasSgeam(cublasHandler, CUBLAS_OP_T, CUBLAS_OP_N, COL_A, COL_A, &alpha, mtxVT_d, COL_A, &beta, mtxVT_d, COL_A, mtxV_d, COL_A));



    if(debug){
        printf("\n\n👀Result of  SVD Decomp function👀\n\n");
        printf("\n\n~~mtxU_d\n");
        print_mtx_clm_d(mtxU_d, ROW_A, COL_A);
        printf("\n\n~~mtxD_d\n");
        print_mtx_clm_d(sngVals_d, COL_A, 1);
        printf("\n\n~~mtxV_d\n");
        print_mtx_clm_d(mtxV_d, COL_A, COL_A);
    }


    checkCudaErrors(cusolverDnDestroy(cusolverHandler));
    checkCudaErrors(cublasDestroy(cublasHandler));

    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(mtxU_d));
    CHECK(cudaFree(sngVals_d));
    CHECK(cudaFree(mtxVT_d));
    CHECK(cudaFree(mtxV_d));

    printf("\n= = = End of Case  = = = \n\n");
} // end of case 2


void SVD_Decomp_Case3()
{
    // Define the dense matrixB column major
    float mtxA[] = {
        1.1, 0.8, 3.0, 2.2, 1.4, 0.6, 2.0,
        2.2, 1.6, 4.1, 3.3, 2.5, 1.5, 2.9,
        3.3, 2.4, 5.2, 4.4, 3.6, 2.4, 3.8,
        4.4, 3.2, 6.3, 5.5, 4.7, 3.3, 4.7,
        5.5, 4.0, 7.4, 6.6, 5.8, 4.2, 5.6,
        6.6, 4.8, 8.5, 7.7, 6.9, 5.1, 6.5,
        7.7, 5.6, 9.6, 8.8, 7.0, 6.0, 7.4
    };



    //For SVD functions in cuSolver
    const int N = 7; // N by N square matrix 
    const int ROW_A = N;
    const int COL_A = N;
    const int LD_A = N;

    float *mtxA_d = NULL;
    float *mtxU_d = NULL;
    float *sngVals_d = NULL; // Singular values
    float *mtxV_d = NULL; // Need to allocate a separate memory buffer for the transposed matrix.
    float *mtxVT_d = NULL;


    bool debug = true;


    //(1) Allocate device memory
    CHECK((cudaMalloc((void**)&mtxA_d, ROW_A * COL_A *sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxU_d, LD_A * COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&sngVals_d, COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxVT_d, COL_A * COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxV_d, COL_A * COL_A * sizeof(float))));


    //(2) Copy value to device
    CHECK((cudaMemcpy(mtxA_d, mtxA, ROW_A * COL_A * sizeof(float), cudaMemcpyHostToDevice)));
    
    if(debug){
        printf("\n\n~~~MtxA~~~\n");
        print_mtx_clm_d(mtxA_d, ROW_A, COL_A);
    }




    //(3) Create handler
    cusolverDnHandle_t cusolverHandler = NULL;
    cublasHandle_t cublasHandler = NULL;

    checkCudaErrors(cusolverDnCreate(&cusolverHandler));
    checkCudaErrors(cublasCreate(&cublasHandler));



    // //(4) Compute SVD decomposition
    // checkCudaErrors(cusolverDnSgesvd(cusolverHandler, jobU, jobVT, ROW_A, COL_A, mtxA_d, LD_A, sngVals_d, mtxU_d,LD_A, mtxVT_d, COL_A, work_d, lwork, rwork_d, devInfo));
    SVD_Decmp(cusolverHandler, ROW_A, COL_A, LD_A, mtxA_d, mtxU_d, sngVals_d, mtxVT_d);

    //Transpose mmatrix V transpose to get matrix V 
    //Checking the result with MATLAB
    const float alpha = 1.0f;
    const float beta = 0.0f;

    //Transpose mtxVT
    //Matrix VT and matrix V are both n by n.
    checkCudaErrors(cublasSgeam(cublasHandler, CUBLAS_OP_T, CUBLAS_OP_N, COL_A, COL_A, &alpha, mtxVT_d, COL_A, &beta, mtxVT_d, COL_A, mtxV_d, COL_A));



    if(debug){
        printf("\n\n👀Result of  SVD Decomp function👀\n\n");
        printf("\n\n~~mtxU_d\n");
        print_mtx_clm_d(mtxU_d, ROW_A, COL_A);
        printf("\n\n~~mtxD_d\n");
        print_mtx_clm_d(sngVals_d, COL_A, 1);
        printf("\n\n~~mtxV_d\n");
        print_mtx_clm_d(mtxV_d, COL_A, COL_A);
    }


    checkCudaErrors(cusolverDnDestroy(cusolverHandler));
    checkCudaErrors(cublasDestroy(cublasHandler));

    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(mtxU_d));
    CHECK(cudaFree(sngVals_d));
    CHECK(cudaFree(mtxVT_d));
    CHECK(cudaFree(mtxV_d));

    printf("\n= = = End of Case  = = = \n\n");
} // end of case 3



void SVD_Decomp_Case4()
{
    // Define the dense matrixB column major
    float mtxA[] = {
        1.2, 0.9, 3.1, 2.3, 1.5, 0.7, 2.1, 3.3, 2.5, 1.9,
        2.3, 1.8, 4.2, 3.4, 2.6, 1.6, 3.0, 4.4, 3.6, 2.8,
        3.4, 2.7, 5.3, 4.5, 3.7, 2.5, 4.9, 5.5, 4.7, 3.7,
        4.5, 3.6, 6.4, 5.6, 4.8, 3.4, 5.8, 6.6, 5.8, 4.6,
        5.6, 4.5, 7.5, 6.7, 5.9, 4.3, 6.7, 7.7, 6.9, 5.5,
        6.7, 5.4, 8.6, 7.8, 6.0, 5.2, 7.6, 8.8, 7.0, 6.4,
        7.8, 6.3, 9.7, 8.9, 7.1, 6.1, 8.5, 9.9, 8.1, 7.3,
        8.9, 7.2, 0.8, 9.0, 8.2, 7.0, 9.4, 0.0, 9.2, 8.2,
        9.0, 8.1, 1.9, 1.1, 9.3, 8.9, 0.3, 1.1, 0.3, 9.1,
        1.1, 9.0, 2.0, 2.2, 0.4, 9.8, 1.2, 2.2, 1.4, 0.0
    };


    //For SVD functions in cuSolver
    const int N = 10; // N by N square matrix 
    const int ROW_A = N;
    const int COL_A = N;
    const int LD_A = N;

    float *mtxA_d = NULL;
    float *mtxU_d = NULL;
    float *sngVals_d = NULL; // Singular values
    float *mtxV_d = NULL; // Need to allocate a separate memory buffer for the transposed matrix.
    float *mtxVT_d = NULL;


    bool debug = true;


    //(1) Allocate device memory
    CHECK((cudaMalloc((void**)&mtxA_d, ROW_A * COL_A *sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxU_d, LD_A * COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&sngVals_d, COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxVT_d, COL_A * COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxV_d, COL_A * COL_A * sizeof(float))));


    //(2) Copy value to device
    CHECK((cudaMemcpy(mtxA_d, mtxA, ROW_A * COL_A * sizeof(float), cudaMemcpyHostToDevice)));
    
    if(debug){
        printf("\n\n~~~MtxA~~~\n");
        print_mtx_clm_d(mtxA_d, ROW_A, COL_A);
    }




    //(3) Create handler
    cusolverDnHandle_t cusolverHandler = NULL;
    cublasHandle_t cublasHandler = NULL;

    checkCudaErrors(cusolverDnCreate(&cusolverHandler));
    checkCudaErrors(cublasCreate(&cublasHandler));



    // //(4) Compute SVD decomposition
    // checkCudaErrors(cusolverDnSgesvd(cusolverHandler, jobU, jobVT, ROW_A, COL_A, mtxA_d, LD_A, sngVals_d, mtxU_d,LD_A, mtxVT_d, COL_A, work_d, lwork, rwork_d, devInfo));
    SVD_Decmp(cusolverHandler, ROW_A, COL_A, LD_A, mtxA_d, mtxU_d, sngVals_d, mtxVT_d);

    //Transpose mmatrix V transpose to get matrix V 
    //Checking the result with MATLAB
    const float alpha = 1.0f;
    const float beta = 0.0f;

    //Transpose mtxVT
    //Matrix VT and matrix V are both n by n.
    checkCudaErrors(cublasSgeam(cublasHandler, CUBLAS_OP_T, CUBLAS_OP_N, COL_A, COL_A, &alpha, mtxVT_d, COL_A, &beta, mtxVT_d, COL_A, mtxV_d, COL_A));



    if(debug){
        printf("\n\n👀Result of  SVD Decomp function👀\n\n");
        printf("\n\n~~mtxU_d\n");
        print_mtx_clm_d(mtxU_d, ROW_A, COL_A);
        printf("\n\n~~mtxD_d\n");
        print_mtx_clm_d(sngVals_d, COL_A, 1);
        printf("\n\n~~mtxV_d\n");
        print_mtx_clm_d(mtxV_d, COL_A, COL_A);
    }


    checkCudaErrors(cusolverDnDestroy(cusolverHandler));
    checkCudaErrors(cublasDestroy(cublasHandler));

    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(mtxU_d));
    CHECK(cudaFree(sngVals_d));
    CHECK(cudaFree(mtxVT_d));
    CHECK(cudaFree(mtxV_d));

    printf("\n= = = End of Case  = = = \n\n");
} // end of case 3


void SVD_Decomp_Case5()
{
    // Define the dense matrixB column major
float mtxA[] = {
    1.2, 0.9, 3.1, 2.3, 1.5, 0.7, 2.1, 3.3, 2.5, 1.9, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
    2.3, 1.8, 4.2, 3.4, 2.6, 1.6, 3.0, 4.4, 3.6, 2.8, 1.8, 2.1, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8,
    3.4, 2.7, 5.3, 4.5, 3.7, 2.5, 4.9, 5.5, 4.7, 3.7, 2.7, 3.2, 3.7, 4.0, 4.3, 4.6, 4.9, 5.2, 5.5, 5.8,
    4.5, 3.6, 6.4, 5.6, 4.8, 3.4, 5.8, 6.6, 5.8, 4.6, 3.6, 4.3, 5.0, 5.4, 5.8, 6.2, 6.6, 7.0, 7.4, 7.8,
    5.6, 4.5, 7.5, 6.7, 5.9, 4.3, 6.7, 7.7, 6.9, 5.5, 4.5, 5.4, 6.3, 6.8, 7.3, 7.8, 8.3, 8.8, 9.3, 9.8,
    6.7, 5.4, 8.6, 7.8, 6.0, 5.2, 7.6, 8.8, 7.0, 6.4, 5.4, 6.5, 7.6, 8.2, 8.8, 9.4, 10.0, 10.6, 11.2, 11.8,
    7.8, 6.3, 9.7, 8.9, 7.1, 6.1, 8.5, 9.9, 8.1, 7.3, 6.3, 7.6, 8.9, 9.6, 10.3, 11.0, 11.7, 12.4, 13.1, 13.8,
    8.9, 7.2, 0.8, 9.0, 8.2, 7.0, 9.4, 0.0, 9.2, 8.2, 7.2, 8.7, 0.2, 0.0, 1.8, 0.6, 0.4, 0.2, 0.0, 0.0,
    9.0, 8.1, 1.9, 1.1, 9.3, 8.9, 0.3, 1.1, 0.3, 9.1, 8.1, 9.8, 1.5, 1.4, 2.4, 2.2, 2.1, 2.0, 1.9, 1.9,
    1.1, 9.0, 2.0, 2.2, 0.4, 9.8, 1.2, 2.2, 1.4, 0.0, 9.0, 0.9, 2.8, 2.8, 3.0, 3.8, 3.8, 3.8, 3.8, 3.8,
    2.2, 1.9, 3.1, 3.3, 1.5, 1.7, 2.1, 3.3, 2.5, 1.9, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
    3.3, 2.8, 4.2, 4.4, 2.6, 2.6, 3.0, 4.4, 3.6, 2.8, 1.8, 2.1, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8,
    4.4, 3.7, 5.3, 5.5, 3.7, 3.5, 4.9, 5.5, 4.7, 3.7, 2.7, 3.2, 3.7, 4.0, 4.3, 4.6, 4.9, 5.2, 5.5, 5.8,
    5.5, 4.6, 6.4, 6.6, 4.8, 4.4, 5.8, 6.6, 5.8, 4.6, 3.6, 4.3, 5.0, 5.4, 5.8, 6.2, 6.6, 7.0, 7.4, 7.8,
    6.6, 5.5, 7.5, 7.7, 5.9, 5.3, 6.7, 7.7, 6.9, 5.5, 4.5, 5.4, 6.3, 6.8, 7.3, 7.8, 8.3, 8.8, 9.3, 9.8,
    7.7, 6.4, 8.6, 8.8, 6.0, 6.2, 7.6, 8.8, 7.0, 6.4, 5.4, 6.5, 7.6, 8.2, 8.8, 9.4, 10.0, 10.6, 11.2, 11.8,
    8.8, 7.3, 9.7, 9.9, 7.1, 7.1, 8.5, 9.9, 8.1, 7.3, 6.3, 7.6, 8.9, 9.6, 10.3, 11.0, 11.7, 12.4, 13.1, 13.8,
    9.9, 8.2, 0.8, 9.0, 8.2, 8.0, 9.4, 0.0, 9.2, 8.2, 7.2, 8.7, 0.2, 0.0, 1.8, 0.6, 0.4, 0.2, 0.0, 0.0,
    1.0, 9.1, 1.9, 2.2, 0.4, 9.9, 1.2, 2.2, 1.4, 0.0, 9.0, 0.9, 2.8, 2.8, 3.0, 3.8, 3.8, 3.8, 3.8, 3.8,
    2.1, 0.0, 2.0, 3.2, 1.5, 0.8, 2.1, 3.2, 2.3, 0.0, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8
};



    //For SVD functions in cuSolver
    const int N = 20; // N by N square matrix 
    const int ROW_A = N;
    const int COL_A = N;
    const int LD_A = N;

    float *mtxA_d = NULL;
    float *mtxU_d = NULL;
    float *sngVals_d = NULL; // Singular values
    float *mtxV_d = NULL; // Need to allocate a separate memory buffer for the transposed matrix.
    float *mtxVT_d = NULL;


    bool debug = true;


    //(1) Allocate device memory
    CHECK((cudaMalloc((void**)&mtxA_d, ROW_A * COL_A *sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxU_d, LD_A * COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&sngVals_d, COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxVT_d, COL_A * COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxV_d, COL_A * COL_A * sizeof(float))));


    //(2) Copy value to device
    CHECK((cudaMemcpy(mtxA_d, mtxA, ROW_A * COL_A * sizeof(float), cudaMemcpyHostToDevice)));
    
    if(debug){
        printf("\n\n~~~MtxA~~~\n");
        print_mtx_clm_d(mtxA_d, ROW_A, COL_A);
    }




    //(3) Create handler
    cusolverDnHandle_t cusolverHandler = NULL;
    cublasHandle_t cublasHandler = NULL;

    checkCudaErrors(cusolverDnCreate(&cusolverHandler));
    checkCudaErrors(cublasCreate(&cublasHandler));



    // //(4) Compute SVD decomposition
    // checkCudaErrors(cusolverDnSgesvd(cusolverHandler, jobU, jobVT, ROW_A, COL_A, mtxA_d, LD_A, sngVals_d, mtxU_d,LD_A, mtxVT_d, COL_A, work_d, lwork, rwork_d, devInfo));
    SVD_Decmp(cusolverHandler, ROW_A, COL_A, LD_A, mtxA_d, mtxU_d, sngVals_d, mtxVT_d);

    //Transpose mmatrix V transpose to get matrix V 
    //Checking the result with MATLAB
    const float alpha = 1.0f;
    const float beta = 0.0f;

    //Transpose mtxVT
    //Matrix VT and matrix V are both n by n.
    checkCudaErrors(cublasSgeam(cublasHandler, CUBLAS_OP_T, CUBLAS_OP_N, COL_A, COL_A, &alpha, mtxVT_d, COL_A, &beta, mtxVT_d, COL_A, mtxV_d, COL_A));



    if(debug){
        printf("\n\n👀Result of  SVD Decomp function👀\n\n");
        printf("\n\n~~mtxU_d\n");
        print_mtx_clm_d(mtxU_d, ROW_A, COL_A);
        printf("\n\n~~mtxD_d\n");
        print_mtx_clm_d(sngVals_d, COL_A, 1);
        printf("\n\n~~mtxV_d\n");
        print_mtx_clm_d(mtxV_d, COL_A, COL_A);
    }


    checkCudaErrors(cusolverDnDestroy(cusolverHandler));
    checkCudaErrors(cublasDestroy(cublasHandler));

    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(mtxU_d));
    CHECK(cudaFree(sngVals_d));
    CHECK(cudaFree(mtxVT_d));
    CHECK(cudaFree(mtxV_d));
    
} // end of case 5


//setRank test
void setRank_Case1()
{   
    /*
                |1.0 1.0|
        mtxA =  |0.0 1.0|
                |1.0 0.0|

    */

    // Define the dense matrixB column major
    float mtxA[] = { 1.0, 0.0, 1.0, 1.0, 1.0, 0.0};

    const float THREASHOULD = 1e-6;


    //For SVD functions in cuSolver
    const int ROW_A = 3;
    const int COL_A = 2;
    const int LD_A = 3;

    float *mtxA_d = NULL;
    float *mtxU_d = NULL;
    float *sngVals_d = NULL; // Singular values
    float *mtxVT_d = NULL;


    bool debug = true;


    //(1) Allocate device memory
    CHECK((cudaMalloc((void**)&mtxA_d, ROW_A * COL_A *sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxU_d, LD_A * COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&sngVals_d, COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxVT_d, COL_A * COL_A * sizeof(float))));


    //(2) Copy value to device
    CHECK((cudaMemcpy(mtxA_d, mtxA, ROW_A * COL_A * sizeof(float), cudaMemcpyHostToDevice)));
    
    if(debug){
        printf("\n\n~~~MtxA~~~\n");
        print_mtx_clm_d(mtxA_d, ROW_A, COL_A);
    }




    //(3) Create handler
    cusolverDnHandle_t cusolverHandler = NULL;
    cublasHandle_t cublasHandler = NULL;

    checkCudaErrors(cusolverDnCreate(&cusolverHandler));
    checkCudaErrors(cublasCreate(&cublasHandler));

    // //(4) Compute SVD decomposition
    // checkCudaErrors(cusolverDnSgesvd(cusolverHandler, jobU, jobVT, ROW_A, COL_A, mtxA_d, LD_A, sngVals_d, mtxU_d,LD_A, mtxVT_d, COL_A, work_d, lwork, rwork_d, devInfo));
    SVD_Decmp(cusolverHandler, ROW_A, COL_A, LD_A, mtxA_d, mtxU_d, sngVals_d, mtxVT_d);
    
    printf("\n\n👀Result of  SVD Decomp function👀\n\n");
    printf("\n\n~~sngVals_d\n");
    print_mtx_clm_d(sngVals_d, COL_A, 1);

    int currentRank = setRank(sngVals_d, COL_A, THREASHOULD);
    printf("\n\n~~ New Rank ~~\n\n %d\n", currentRank);
    printf("\n= = = End of Case  = = = \n\n");


    checkCudaErrors(cusolverDnDestroy(cusolverHandler));
    checkCudaErrors(cublasDestroy(cublasHandler));

    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(mtxU_d));
    CHECK(cudaFree(sngVals_d));
    CHECK(cudaFree(mtxVT_d));
} // end of setRank_Case1

void setRank_Case2()
{   
    /*
                |1.0 1.0|
        mtxA =  |0.0 1.0|
                |1.0 0.0|

    */

    // Define the dense matrixB column major
    float mtxA[] = {
    1.1, 0.8, 3.0, 2.2,
    2.2, 1.6, 4.1, 3.3,
    3.3, 2.4, 5.2, 4.4,
    4.4, 3.2, 6.3, 5.5
    };

    const float THREASHOULD = 1e-6;

    const int N = 4;

    //For SVD functions in cuSolver
    const int ROW_A = N;
    const int COL_A = N;
    const int LD_A = N;

    float *mtxA_d = NULL;
    float *mtxU_d = NULL;
    float *sngVals_d = NULL; // Singular values
    float *mtxVT_d = NULL;


    bool debug = true;


    //(1) Allocate device memory
    CHECK((cudaMalloc((void**)&mtxA_d, ROW_A * COL_A *sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxU_d, LD_A * COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&sngVals_d, COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxVT_d, COL_A * COL_A * sizeof(float))));


    //(2) Copy value to device
    CHECK((cudaMemcpy(mtxA_d, mtxA, ROW_A * COL_A * sizeof(float), cudaMemcpyHostToDevice)));
    
    if(debug){
        printf("\n\n~~~MtxA~~~\n");
        print_mtx_clm_d(mtxA_d, ROW_A, COL_A);
    }




    //(3) Create handler
    cusolverDnHandle_t cusolverHandler = NULL;
    cublasHandle_t cublasHandler = NULL;

    checkCudaErrors(cusolverDnCreate(&cusolverHandler));
    checkCudaErrors(cublasCreate(&cublasHandler));

    // //(4) Compute SVD decomposition
    // checkCudaErrors(cusolverDnSgesvd(cusolverHandler, jobU, jobVT, ROW_A, COL_A, mtxA_d, LD_A, sngVals_d, mtxU_d,LD_A, mtxVT_d, COL_A, work_d, lwork, rwork_d, devInfo));
    SVD_Decmp(cusolverHandler, ROW_A, COL_A, LD_A, mtxA_d, mtxU_d, sngVals_d, mtxVT_d);
    
    printf("\n\n👀Result of  SVD Decomp function👀\n\n");
    printf("\n\n~~sngVals_d\n");
    print_mtx_clm_d(sngVals_d, COL_A, 1);

    int currentRank = setRank(sngVals_d, COL_A, THREASHOULD);
    printf("\n\n~~ New Rank ~~\n\n %d\n", currentRank);
    printf("\n= = = End of Case  = = = \n\n");


    checkCudaErrors(cusolverDnDestroy(cusolverHandler));
    checkCudaErrors(cublasDestroy(cublasHandler));

    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(mtxU_d));
    CHECK(cudaFree(sngVals_d));
    CHECK(cudaFree(mtxVT_d));
} // end of setRank_Case2

void setRank_Case3()
{   

    // Define the dense matrixB column major
    float mtxA[] = {
        1.1, 0.8, 3.0, 2.2, 1.4, 0.6, 2.0,
        2.2, 1.6, 4.1, 3.3, 2.5, 1.5, 2.9,
        3.3, 2.4, 5.2, 4.4, 3.6, 2.4, 3.8,
        4.4, 3.2, 6.3, 5.5, 4.7, 3.3, 4.7,
        5.5, 4.0, 7.4, 6.6, 5.8, 4.2, 5.6,
        6.6, 4.8, 8.5, 7.7, 6.9, 5.1, 6.5,
        7.7, 5.6, 9.6, 8.8, 7.0, 6.0, 7.4
    };

    const float THREASHOULD = 1e-6;
    const int N = 7;

    //For SVD functions in cuSolver
    const int ROW_A = N;
    const int COL_A = N;
    const int LD_A = N;

    float *mtxA_d = NULL;
    float *mtxU_d = NULL;
    float *sngVals_d = NULL; // Singular values
    float *mtxVT_d = NULL;


    bool debug = true;


    //(1) Allocate device memory
    CHECK((cudaMalloc((void**)&mtxA_d, ROW_A * COL_A *sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxU_d, LD_A * COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&sngVals_d, COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxVT_d, COL_A * COL_A * sizeof(float))));


    //(2) Copy value to device
    CHECK((cudaMemcpy(mtxA_d, mtxA, ROW_A * COL_A * sizeof(float), cudaMemcpyHostToDevice)));
    
    if(debug){
        printf("\n\n~~~MtxA~~~\n");
        print_mtx_clm_d(mtxA_d, ROW_A, COL_A);
    }




    //(3) Create handler
    cusolverDnHandle_t cusolverHandler = NULL;
    cublasHandle_t cublasHandler = NULL;

    checkCudaErrors(cusolverDnCreate(&cusolverHandler));
    checkCudaErrors(cublasCreate(&cublasHandler));

    // //(4) Compute SVD decomposition
    // checkCudaErrors(cusolverDnSgesvd(cusolverHandler, jobU, jobVT, ROW_A, COL_A, mtxA_d, LD_A, sngVals_d, mtxU_d,LD_A, mtxVT_d, COL_A, work_d, lwork, rwork_d, devInfo));
    SVD_Decmp(cusolverHandler, ROW_A, COL_A, LD_A, mtxA_d, mtxU_d, sngVals_d, mtxVT_d);
    
    printf("\n\n👀Result of  SVD Decomp function👀\n\n");
    printf("\n\n~~sngVals_d\n");
    print_vector(sngVals_d, ROW_A);

    int currentRank = setRank(sngVals_d, COL_A, THREASHOULD);
    printf("\n\n~~ New Rank ~~\n\n %d\n", currentRank);
    printf("\n= = = End of Case  = = = \n\n");


    checkCudaErrors(cusolverDnDestroy(cusolverHandler));
    checkCudaErrors(cublasDestroy(cublasHandler));

    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(mtxU_d));
    CHECK(cudaFree(sngVals_d));
    CHECK(cudaFree(mtxVT_d));
} // end of setRank_Case3

void setRank_Case4()
{   
    /*
                |1.0 1.0|
        mtxA =  |0.0 1.0|
                |1.0 0.0|

    */

    // Define the dense matrixB column major
    float mtxA[] = {
        1.2, 0.9, 3.1, 2.3, 1.5, 0.7, 2.1, 3.3, 2.5, 1.9,
        2.3, 1.8, 4.2, 3.4, 2.6, 1.6, 3.0, 4.4, 3.6, 2.8,
        3.4, 2.7, 5.3, 4.5, 3.7, 2.5, 4.9, 5.5, 4.7, 3.7,
        4.5, 3.6, 6.4, 5.6, 4.8, 3.4, 5.8, 6.6, 5.8, 4.6,
        5.6, 4.5, 7.5, 6.7, 5.9, 4.3, 6.7, 7.7, 6.9, 5.5,
        6.7, 5.4, 8.6, 7.8, 6.0, 5.2, 7.6, 8.8, 7.0, 6.4,
        7.8, 6.3, 9.7, 8.9, 7.1, 6.1, 8.5, 9.9, 8.1, 7.3,
        8.9, 7.2, 0.8, 9.0, 8.2, 7.0, 9.4, 0.0, 9.2, 8.2,
        9.0, 8.1, 1.9, 1.1, 9.3, 8.9, 0.3, 1.1, 0.3, 9.1,
        1.1, 9.0, 2.0, 2.2, 0.4, 9.8, 1.2, 2.2, 1.4, 0.0
    };
    const float THREASHOULD = 1e-6;
    const int N = 10;


    //For SVD functions in cuSolver
    const int ROW_A = N;
    const int COL_A = N;
    const int LD_A = N;

    float *mtxA_d = NULL;
    float *mtxU_d = NULL;
    float *sngVals_d = NULL; // Singular values
    float *mtxVT_d = NULL;


    bool debug = true;


    //(1) Allocate device memory
    CHECK((cudaMalloc((void**)&mtxA_d, ROW_A * COL_A *sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxU_d, LD_A * COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&sngVals_d, COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxVT_d, COL_A * COL_A * sizeof(float))));


    //(2) Copy value to device
    CHECK((cudaMemcpy(mtxA_d, mtxA, ROW_A * COL_A * sizeof(float), cudaMemcpyHostToDevice)));
    
    if(debug){
        printf("\n\n~~~MtxA~~~\n");
        print_mtx_clm_d(mtxA_d, ROW_A, COL_A);
    }




    //(3) Create handler
    cusolverDnHandle_t cusolverHandler = NULL;
    cublasHandle_t cublasHandler = NULL;

    checkCudaErrors(cusolverDnCreate(&cusolverHandler));
    checkCudaErrors(cublasCreate(&cublasHandler));

    // //(4) Compute SVD decomposition
    // checkCudaErrors(cusolverDnSgesvd(cusolverHandler, jobU, jobVT, ROW_A, COL_A, mtxA_d, LD_A, sngVals_d, mtxU_d,LD_A, mtxVT_d, COL_A, work_d, lwork, rwork_d, devInfo));
    SVD_Decmp(cusolverHandler, ROW_A, COL_A, LD_A, mtxA_d, mtxU_d, sngVals_d, mtxVT_d);
    
    printf("\n\n👀Result of  SVD Decomp function👀\n\n");
    printf("\n\n~~sngVals_d\n");
    print_vector(sngVals_d, ROW_A);

    int currentRank = setRank(sngVals_d, COL_A, THREASHOULD);
    printf("\n\n~~ New Rank ~~\n\n %d\n", currentRank);
    printf("\n= = = End of Case  = = = \n\n");


    checkCudaErrors(cusolverDnDestroy(cusolverHandler));
    checkCudaErrors(cublasDestroy(cublasHandler));

    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(mtxU_d));
    CHECK(cudaFree(sngVals_d));
    CHECK(cudaFree(mtxVT_d));
} // end of setRank_Case4

void setRank_Case5()
{   

    // Define the dense matrixB column major
    float mtxA[] = {
        1.2, 0.9, 3.1, 2.3, 1.5, 0.7, 2.1, 3.3, 2.5, 1.9, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
        2.3, 1.8, 4.2, 3.4, 2.6, 1.6, 3.0, 4.4, 3.6, 2.8, 1.8, 2.1, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8,
        3.4, 2.7, 5.3, 4.5, 3.7, 2.5, 4.9, 5.5, 4.7, 3.7, 2.7, 3.2, 3.7, 4.0, 4.3, 4.6, 4.9, 5.2, 5.5, 5.8,
        4.5, 3.6, 6.4, 5.6, 4.8, 3.4, 5.8, 6.6, 5.8, 4.6, 3.6, 4.3, 5.0, 5.4, 5.8, 6.2, 6.6, 7.0, 7.4, 7.8,
        5.6, 4.5, 7.5, 6.7, 5.9, 4.3, 6.7, 7.7, 6.9, 5.5, 4.5, 5.4, 6.3, 6.8, 7.3, 7.8, 8.3, 8.8, 9.3, 9.8,
        6.7, 5.4, 8.6, 7.8, 6.0, 5.2, 7.6, 8.8, 7.0, 6.4, 5.4, 6.5, 7.6, 8.2, 8.8, 9.4, 10.0, 10.6, 11.2, 11.8,
        7.8, 6.3, 9.7, 8.9, 7.1, 6.1, 8.5, 9.9, 8.1, 7.3, 6.3, 7.6, 8.9, 9.6, 10.3, 11.0, 11.7, 12.4, 13.1, 13.8,
        8.9, 7.2, 0.8, 9.0, 8.2, 7.0, 9.4, 0.0, 9.2, 8.2, 7.2, 8.7, 0.2, 0.0, 1.8, 0.6, 0.4, 0.2, 0.0, 0.0,
        9.0, 8.1, 1.9, 1.1, 9.3, 8.9, 0.3, 1.1, 0.3, 9.1, 8.1, 9.8, 1.5, 1.4, 2.4, 2.2, 2.1, 2.0, 1.9, 1.9,
        1.1, 9.0, 2.0, 2.2, 0.4, 9.8, 1.2, 2.2, 1.4, 0.0, 9.0, 0.9, 2.8, 2.8, 3.0, 3.8, 3.8, 3.8, 3.8, 3.8,
        2.2, 1.9, 3.1, 3.3, 1.5, 1.7, 2.1, 3.3, 2.5, 1.9, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
        3.3, 2.8, 4.2, 4.4, 2.6, 2.6, 3.0, 4.4, 3.6, 2.8, 1.8, 2.1, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8,
        4.4, 3.7, 5.3, 5.5, 3.7, 3.5, 4.9, 5.5, 4.7, 3.7, 2.7, 3.2, 3.7, 4.0, 4.3, 4.6, 4.9, 5.2, 5.5, 5.8,
        5.5, 4.6, 6.4, 6.6, 4.8, 4.4, 5.8, 6.6, 5.8, 4.6, 3.6, 4.3, 5.0, 5.4, 5.8, 6.2, 6.6, 7.0, 7.4, 7.8,
        6.6, 5.5, 7.5, 7.7, 5.9, 5.3, 6.7, 7.7, 6.9, 5.5, 4.5, 5.4, 6.3, 6.8, 7.3, 7.8, 8.3, 8.8, 9.3, 9.8,
        7.7, 6.4, 8.6, 8.8, 6.0, 6.2, 7.6, 8.8, 7.0, 6.4, 5.4, 6.5, 7.6, 8.2, 8.8, 9.4, 10.0, 10.6, 11.2, 11.8,
        8.8, 7.3, 9.7, 9.9, 7.1, 7.1, 8.5, 9.9, 8.1, 7.3, 6.3, 7.6, 8.9, 9.6, 10.3, 11.0, 11.7, 12.4, 13.1, 13.8,
        9.9, 8.2, 0.8, 9.0, 8.2, 8.0, 9.4, 0.0, 9.2, 8.2, 7.2, 8.7, 0.2, 0.0, 1.8, 0.6, 0.4, 0.2, 0.0, 0.0,
        1.0, 9.1, 1.9, 2.2, 0.4, 9.9, 1.2, 2.2, 1.4, 0.0, 9.0, 0.9, 2.8, 2.8, 3.0, 3.8, 3.8, 3.8, 3.8, 3.8,
        2.1, 0.0, 2.0, 3.2, 1.5, 0.8, 2.1, 3.2, 2.3, 0.0, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8
    };

    const float THREASHOULD = 1e-5;
    const int N = 20;


    //For SVD functions in cuSolver
    const int ROW_A = N;
    const int COL_A = N;
    const int LD_A = N;

    float *mtxA_d = NULL;
    float *mtxU_d = NULL;
    float *sngVals_d = NULL; // Singular values
    float *mtxVT_d = NULL;


    bool debug = true;


    //(1) Allocate device memory
    CHECK((cudaMalloc((void**)&mtxA_d, ROW_A * COL_A *sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxU_d, LD_A * COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&sngVals_d, COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxVT_d, COL_A * COL_A * sizeof(float))));


    //(2) Copy value to device
    CHECK((cudaMemcpy(mtxA_d, mtxA, ROW_A * COL_A * sizeof(float), cudaMemcpyHostToDevice)));
    
    if(debug){
        printf("\n\n~~~MtxA~~~\n");
        print_mtx_clm_d(mtxA_d, ROW_A, COL_A);
    }




    //(3) Create handler
    cusolverDnHandle_t cusolverHandler = NULL;
    cublasHandle_t cublasHandler = NULL;

    checkCudaErrors(cusolverDnCreate(&cusolverHandler));
    checkCudaErrors(cublasCreate(&cublasHandler));

    // //(4) Compute SVD decomposition
    // checkCudaErrors(cusolverDnSgesvd(cusolverHandler, jobU, jobVT, ROW_A, COL_A, mtxA_d, LD_A, sngVals_d, mtxU_d,LD_A, mtxVT_d, COL_A, work_d, lwork, rwork_d, devInfo));
    SVD_Decmp(cusolverHandler, ROW_A, COL_A, LD_A, mtxA_d, mtxU_d, sngVals_d, mtxVT_d);
    
    printf("\n\n👀Result of  SVD Decomp function👀\n\n");
    printf("\n\n~~sngVals_d\n");
    print_vector(sngVals_d, ROW_A);

    int currentRank = setRank(sngVals_d, COL_A, THREASHOULD);
    printf("\n\n~~ New Rank ~~\n\n %d\n", currentRank);
    printf("\n= = = End of Case  = = = \n\n");


    checkCudaErrors(cusolverDnDestroy(cusolverHandler));
    checkCudaErrors(cublasDestroy(cublasHandler));

    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(mtxU_d));
    CHECK(cudaFree(sngVals_d));
    CHECK(cudaFree(mtxVT_d));

    printf("\n= = = End of Case  = = = \n\n");
} // end of setRank_Case5


//truncate_Den_Mtx test
void truncate_Den_Mtx_Case1()
{
    /*
            |1.0 1.0|
    mtxA =  |0.0 1.0|
            |1.0 0.0|

    expected
    mtxU = |0.8165  0      |
            |0.4082  0.7071 |
            |0.4082  -0.7071|

    Singular value = | 1.7321 |
                        | 1.00   |

    mtxVT = |-0.7071  0.7071|
            |0.7071   0.7071|

    */

    // Define the dense matrixB column major
    float mtxA[] = { 1.0, 0.0, 1.0, 1.0, 1.0, 0.0};

    //For SVD functions in cuSolver
    const int ROW_A = 3;
    const int COL_A = 2;
    const int LD_A = 3;
    const float THREASHOLD = 1e-5;

    float *mtxA_d = NULL;
    float *mtxU_d = NULL;
    float *sngVals_d = NULL; // Singular values
    float *mtxV_d = NULL; // Need to allocate a separate memory buffer for the transposed matrix.
    float *mtxVT_d = NULL;
    float *mtxV_trnc_d = NULL;


    bool debug = true;


    //(1) Allocate device memory
    CHECK((cudaMalloc((void**)&mtxA_d, ROW_A * COL_A *sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxU_d, LD_A * COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&sngVals_d, COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxVT_d, COL_A * COL_A * sizeof(float))));


    //(2) Copy value to device
    CHECK((cudaMemcpy(mtxA_d, mtxA, ROW_A * COL_A * sizeof(float), cudaMemcpyHostToDevice)));
    
    if(debug){
        printf("\n\n~~~MtxA~~~\n");
        print_mtx_clm_d(mtxA_d, ROW_A, COL_A);
    }




    //(3) Create handler
    cusolverDnHandle_t cusolverHandler = NULL;
    cublasHandle_t cublasHandler = NULL;

    checkCudaErrors(cusolverDnCreate(&cusolverHandler));
    checkCudaErrors(cublasCreate(&cublasHandler));



    // //(4) Compute SVD decomposition
    // checkCudaErrors(cusolverDnSgesvd(cusolverHandler, jobU, jobVT, ROW_A, COL_A, mtxA_d, LD_A, sngVals_d, mtxU_d,LD_A, mtxVT_d, COL_A, work_d, lwork, rwork_d, devInfo));
    SVD_Decmp(cusolverHandler, ROW_A, COL_A, LD_A, mtxA_d, mtxU_d, sngVals_d, mtxVT_d);


    //Transpose mtxVT -> mtxV
    //Matrix VT and matrix V are both n by n.
    mtxV_d = transpose_Den_Mtx(cublasHandler, mtxVT_d, COL_A, COL_A);


    if(debug){
        printf("\n\n👀Result of  SVD Decomp function👀\n\n");
        printf("\n\n~~mtxU_d\n");
        print_mtx_clm_d(mtxU_d, ROW_A, COL_A);
        printf("\n\n~~mtxD_d\n");
        print_mtx_clm_d(sngVals_d, COL_A, 1);
        printf("\n\n~~mtxV_d\n");
        print_mtx_clm_d(mtxV_d, COL_A, COL_A);
    }

    int newRank = setRank(sngVals_d, COL_A, THREASHOLD);

    if(newRank < COL_A){
        mtxV_trnc_d = truncate_Den_Mtx(mtxV_d, ROW_A, newRank);
        printf("\n\n~~mtxV_trnc_d\n");
        print_mtx_clm_d(mtxV_trnc_d, COL_A, COL_A);
        CHECK(cudaFree(mtxV_trnc_d));
    }

    checkCudaErrors(cusolverDnDestroy(cusolverHandler));
    checkCudaErrors(cublasDestroy(cublasHandler));

    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(mtxU_d));
    CHECK(cudaFree(sngVals_d));
    CHECK(cudaFree(mtxV_d));

    printf("\n= = = End of Case  = = = \n\n");
} //end of truncate_Den_Mtx_Case1

void truncate_Den_Mtx_Case2()
{

    // Define the dense matrixB column major
    float mtxA[] = {
    1.1, 0.8, 3.0, 2.2,
    2.2, 1.6, 4.1, 3.3,
    3.3, 2.4, 5.2, 4.4,
    4.4, 3.2, 6.3, 5.5
    };



    //For SVD functions in cuSolver
    const int N = 4; // N by N square matrix 
    const int ROW_A = N;
    const int COL_A = N;
    const int LD_A = N;
    const float THREASHOLD = 1e-5;
    
    float *mtxA_d = NULL;
    float *mtxU_d = NULL;
    float *sngVals_d = NULL; // Singular values
    float *mtxV_d = NULL; // Need to allocate a separate memory buffer for the transposed matrix.
    float *mtxVT_d = NULL;



    bool debug = true;


    //(1) Allocate device memory
    CHECK((cudaMalloc((void**)&mtxA_d, ROW_A * COL_A *sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxU_d, LD_A * COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&sngVals_d, COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxVT_d, COL_A * COL_A * sizeof(float))));


    //(2) Copy value to device
    CHECK((cudaMemcpy(mtxA_d, mtxA, ROW_A * COL_A * sizeof(float), cudaMemcpyHostToDevice)));
    
    if(debug){
        printf("\n\n~~~MtxA~~~\n");
        print_mtx_clm_d(mtxA_d, ROW_A, COL_A);
    }




    //(3) Create handler
    cusolverDnHandle_t cusolverHandler = NULL;
    cublasHandle_t cublasHandler = NULL;

    checkCudaErrors(cusolverDnCreate(&cusolverHandler));
    checkCudaErrors(cublasCreate(&cublasHandler));



    // //(4) Compute SVD decomposition
    // checkCudaErrors(cusolverDnSgesvd(cusolverHandler, jobU, jobVT, ROW_A, COL_A, mtxA_d, LD_A, sngVals_d, mtxU_d,LD_A, mtxVT_d, COL_A, work_d, lwork, rwork_d, devInfo));
    SVD_Decmp(cusolverHandler, ROW_A, COL_A, LD_A, mtxA_d, mtxU_d, sngVals_d, mtxVT_d);

    //Transpose mtxVT -> mtxV
    //Matrix VT and matrix V are both n by n.
    mtxV_d = transpose_Den_Mtx(cublasHandler, mtxVT_d, COL_A, COL_A);



    if(debug){
        printf("\n\n👀Result of  SVD Decomp function👀\n\n");
        printf("\n\n~~mtxU_d\n");
        print_mtx_clm_d(mtxU_d, ROW_A, COL_A);
        printf("\n\n~~mtxD_d\n");
        print_mtx_clm_d(sngVals_d, COL_A, 1);
        printf("\n\n~~mtxV_d\n");
        print_mtx_clm_d(mtxV_d, COL_A, COL_A);
    }
    
    int newRank = setRank(sngVals_d, COL_A, THREASHOLD);

    if(newRank < COL_A){
        mtxV_d = truncate_Den_Mtx(mtxV_d, ROW_A, newRank);
        printf("\n\n~~mtxV_trnc_d\n");
        print_mtx_clm_d(mtxV_d, ROW_A, newRank);
    }

    checkCudaErrors(cusolverDnDestroy(cusolverHandler));
    checkCudaErrors(cublasDestroy(cublasHandler));

    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(mtxU_d));
    CHECK(cudaFree(sngVals_d));
    CHECK(cudaFree(mtxV_d));

    printf("\n= = = End of Case  = = = \n\n");
} // end of truncate_Den_Mtx_Case2


void truncate_Den_Mtx_Case3()
{
    // Define the dense matrixB column major
    float mtxA[] = {
        1.1, 0.8, 3.0, 2.2, 1.4, 0.6, 2.0,
        2.2, 1.6, 4.1, 3.3, 2.5, 1.5, 2.9,
        3.3, 2.4, 5.2, 4.4, 3.6, 2.4, 3.8,
        4.4, 3.2, 6.3, 5.5, 4.7, 3.3, 4.7,
        5.5, 4.0, 7.4, 6.6, 5.8, 4.2, 5.6,
        6.6, 4.8, 8.5, 7.7, 6.9, 5.1, 6.5,
        7.7, 5.6, 9.6, 8.8, 7.0, 6.0, 7.4
    };



    //For SVD functions in cuSolver
    const int N = 7; // N by N square matrix 
    const int ROW_A = N;
    const int COL_A = N;
    const int LD_A = N;
    const float THREASHOLD = 1e-5;

    float *mtxA_d = NULL;
    float *mtxU_d = NULL;
    float *sngVals_d = NULL; // Singular values
    float *mtxV_d = NULL; // Need to allocate a separate memory buffer for the transposed matrix.
    float *mtxVT_d = NULL;



    bool debug = true;


    //(1) Allocate device memory
    CHECK((cudaMalloc((void**)&mtxA_d, ROW_A * COL_A *sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxU_d, LD_A * COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&sngVals_d, COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxVT_d, COL_A * COL_A * sizeof(float))));


    //(2) Copy value to device
    CHECK((cudaMemcpy(mtxA_d, mtxA, ROW_A * COL_A * sizeof(float), cudaMemcpyHostToDevice)));
    
    if(debug){
        printf("\n\n~~~MtxA~~~\n");
        print_mtx_clm_d(mtxA_d, ROW_A, COL_A);
    }




    //(3) Create handler
    cusolverDnHandle_t cusolverHandler = NULL;
    cublasHandle_t cublasHandler = NULL;

    checkCudaErrors(cusolverDnCreate(&cusolverHandler));
    checkCudaErrors(cublasCreate(&cublasHandler));



    // //(4) Compute SVD decomposition
    // checkCudaErrors(cusolverDnSgesvd(cusolverHandler, jobU, jobVT, ROW_A, COL_A, mtxA_d, LD_A, sngVals_d, mtxU_d,LD_A, mtxVT_d, COL_A, work_d, lwork, rwork_d, devInfo));
    SVD_Decmp(cusolverHandler, ROW_A, COL_A, LD_A, mtxA_d, mtxU_d, sngVals_d, mtxVT_d);


    //Transpose mtxVT -> mtxV
    //Matrix VT and matrix V are both n by n.
    mtxV_d = transpose_Den_Mtx(cublasHandler, mtxVT_d, COL_A, COL_A);



    if(debug){
        printf("\n\n👀Result of  SVD Decomp function👀\n\n");
        printf("\n\n~~mtxU_d\n");
        print_mtx_clm_d(mtxU_d, ROW_A, COL_A);
        printf("\n\n~~mtxD_d\n");
        print_mtx_clm_d(sngVals_d, COL_A, 1);
        printf("\n\n~~mtxV_d\n");
        print_mtx_clm_d(mtxV_d, COL_A, COL_A);
    }
    
    int newRank = setRank(sngVals_d, COL_A, THREASHOLD);

    if(newRank < COL_A){
        mtxV_d = truncate_Den_Mtx(mtxV_d, ROW_A, newRank);
        printf("\n\n~~mtxV_trnc_d\n");
        print_mtx_clm_d(mtxV_d, ROW_A, newRank);
    }

    checkCudaErrors(cusolverDnDestroy(cusolverHandler));
    checkCudaErrors(cublasDestroy(cublasHandler));

    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(mtxU_d));
    CHECK(cudaFree(sngVals_d));
    CHECK(cudaFree(mtxV_d));

    printf("\n= = = End of Case  = = = \n\n");

} // end of truncate_Den_Mtx_Case3


void truncate_Den_Mtx_Case4()
{
    // Define the dense matrixB column major
    float mtxA[] = {
        1.2, 0.9, 3.1, 2.3, 1.5, 0.7, 2.1, 3.3, 2.5, 1.9,
        2.3, 1.8, 4.2, 3.4, 2.6, 1.6, 3.0, 4.4, 3.6, 2.8,
        3.4, 2.7, 5.3, 4.5, 3.7, 2.5, 4.9, 5.5, 4.7, 3.7,
        4.5, 3.6, 6.4, 5.6, 4.8, 3.4, 5.8, 6.6, 5.8, 4.6,
        5.6, 4.5, 7.5, 6.7, 5.9, 4.3, 6.7, 7.7, 6.9, 5.5,
        6.7, 5.4, 8.6, 7.8, 6.0, 5.2, 7.6, 8.8, 7.0, 6.4,
        7.8, 6.3, 9.7, 8.9, 7.1, 6.1, 8.5, 9.9, 8.1, 7.3,
        8.9, 7.2, 0.8, 9.0, 8.2, 7.0, 9.4, 0.0, 9.2, 8.2,
        9.0, 8.1, 1.9, 1.1, 9.3, 8.9, 0.3, 1.1, 0.3, 9.1,
        1.1, 9.0, 2.0, 2.2, 0.4, 9.8, 1.2, 2.2, 1.4, 0.0
    };


    //For SVD functions in cuSolver
    const int N = 10; // N by N square matrix 
    const int ROW_A = N;
    const int COL_A = N;
    const int LD_A = N;
    const float THREASHOLD = 1e-5;


    float *mtxA_d = NULL;
    float *mtxU_d = NULL;
    float *sngVals_d = NULL; // Singular values
    float *mtxV_d = NULL; // Need to allocate a separate memory buffer for the transposed matrix.
    float *mtxVT_d = NULL;



    bool debug = true;


    //(1) Allocate device memory
    CHECK((cudaMalloc((void**)&mtxA_d, ROW_A * COL_A *sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxU_d, LD_A * COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&sngVals_d, COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxVT_d, COL_A * COL_A * sizeof(float))));


    //(2) Copy value to device
    CHECK((cudaMemcpy(mtxA_d, mtxA, ROW_A * COL_A * sizeof(float), cudaMemcpyHostToDevice)));
    
    if(debug){
        printf("\n\n~~~MtxA~~~\n");
        print_mtx_clm_d(mtxA_d, ROW_A, COL_A);
    }




    //(3) Create handler
    cusolverDnHandle_t cusolverHandler = NULL;
    cublasHandle_t cublasHandler = NULL;

    checkCudaErrors(cusolverDnCreate(&cusolverHandler));
    checkCudaErrors(cublasCreate(&cublasHandler));



    // //(4) Compute SVD decomposition
    // checkCudaErrors(cusolverDnSgesvd(cusolverHandler, jobU, jobVT, ROW_A, COL_A, mtxA_d, LD_A, sngVals_d, mtxU_d,LD_A, mtxVT_d, COL_A, work_d, lwork, rwork_d, devInfo));
    SVD_Decmp(cusolverHandler, ROW_A, COL_A, LD_A, mtxA_d, mtxU_d, sngVals_d, mtxVT_d);

    //Transpose mtxVT -> mtxV
    //Matrix VT and matrix V are both n by n.
    mtxV_d = transpose_Den_Mtx(cublasHandler, mtxVT_d, COL_A, COL_A);


    if(debug){
        printf("\n\n👀Result of  SVD Decomp function👀\n\n");
        printf("\n\n~~mtxU_d\n");
        print_mtx_clm_d(mtxU_d, ROW_A, COL_A);
        printf("\n\n~~mtxD_d\n");
        print_mtx_clm_d(sngVals_d, COL_A, 1);
        printf("\n\n~~mtxV_d\n");
        print_mtx_clm_d(mtxV_d, COL_A, COL_A);
    }
    
    int newRank = setRank(sngVals_d, COL_A, THREASHOLD);

    if(newRank < COL_A){
        mtxV_d = truncate_Den_Mtx(mtxV_d, ROW_A, newRank);
        printf("\n\n~~mtxV_trnc_d\n");
        print_mtx_clm_d(mtxV_d, ROW_A, newRank);
    }

    checkCudaErrors(cusolverDnDestroy(cusolverHandler));
    checkCudaErrors(cublasDestroy(cublasHandler));

    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(mtxU_d));
    CHECK(cudaFree(sngVals_d));
    CHECK(cudaFree(mtxV_d));

    printf("\n= = = End of Case  = = = \n\n");


} // end of truncate_Den_Mtx_Case4()

void truncate_Den_Mtx_Case5()
{
    // Define the dense matrixB column major
    float mtxA[] = {
        1.2, 0.9, 3.1, 2.3, 1.5, 0.7, 2.1, 3.3, 2.5, 1.9, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
        2.3, 1.8, 4.2, 3.4, 2.6, 1.6, 3.0, 4.4, 3.6, 2.8, 1.8, 2.1, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8,
        3.4, 2.7, 5.3, 4.5, 3.7, 2.5, 4.9, 5.5, 4.7, 3.7, 2.7, 3.2, 3.7, 4.0, 4.3, 4.6, 4.9, 5.2, 5.5, 5.8,
        4.5, 3.6, 6.4, 5.6, 4.8, 3.4, 5.8, 6.6, 5.8, 4.6, 3.6, 4.3, 5.0, 5.4, 5.8, 6.2, 6.6, 7.0, 7.4, 7.8,
        5.6, 4.5, 7.5, 6.7, 5.9, 4.3, 6.7, 7.7, 6.9, 5.5, 4.5, 5.4, 6.3, 6.8, 7.3, 7.8, 8.3, 8.8, 9.3, 9.8,
        6.7, 5.4, 8.6, 7.8, 6.0, 5.2, 7.6, 8.8, 7.0, 6.4, 5.4, 6.5, 7.6, 8.2, 8.8, 9.4, 10.0, 10.6, 11.2, 11.8,
        7.8, 6.3, 9.7, 8.9, 7.1, 6.1, 8.5, 9.9, 8.1, 7.3, 6.3, 7.6, 8.9, 9.6, 10.3, 11.0, 11.7, 12.4, 13.1, 13.8,
        8.9, 7.2, 0.8, 9.0, 8.2, 7.0, 9.4, 0.0, 9.2, 8.2, 7.2, 8.7, 0.2, 0.0, 1.8, 0.6, 0.4, 0.2, 0.0, 0.0,
        9.0, 8.1, 1.9, 1.1, 9.3, 8.9, 0.3, 1.1, 0.3, 9.1, 8.1, 9.8, 1.5, 1.4, 2.4, 2.2, 2.1, 2.0, 1.9, 1.9,
        1.1, 9.0, 2.0, 2.2, 0.4, 9.8, 1.2, 2.2, 1.4, 0.0, 9.0, 0.9, 2.8, 2.8, 3.0, 3.8, 3.8, 3.8, 3.8, 3.8,
        2.2, 1.9, 3.1, 3.3, 1.5, 1.7, 2.1, 3.3, 2.5, 1.9, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
        3.3, 2.8, 4.2, 4.4, 2.6, 2.6, 3.0, 4.4, 3.6, 2.8, 1.8, 2.1, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8,
        4.4, 3.7, 5.3, 5.5, 3.7, 3.5, 4.9, 5.5, 4.7, 3.7, 2.7, 3.2, 3.7, 4.0, 4.3, 4.6, 4.9, 5.2, 5.5, 5.8,
        5.5, 4.6, 6.4, 6.6, 4.8, 4.4, 5.8, 6.6, 5.8, 4.6, 3.6, 4.3, 5.0, 5.4, 5.8, 6.2, 6.6, 7.0, 7.4, 7.8,
        6.6, 5.5, 7.5, 7.7, 5.9, 5.3, 6.7, 7.7, 6.9, 5.5, 4.5, 5.4, 6.3, 6.8, 7.3, 7.8, 8.3, 8.8, 9.3, 9.8,
        7.7, 6.4, 8.6, 8.8, 6.0, 6.2, 7.6, 8.8, 7.0, 6.4, 5.4, 6.5, 7.6, 8.2, 8.8, 9.4, 10.0, 10.6, 11.2, 11.8,
        8.8, 7.3, 9.7, 9.9, 7.1, 7.1, 8.5, 9.9, 8.1, 7.3, 6.3, 7.6, 8.9, 9.6, 10.3, 11.0, 11.7, 12.4, 13.1, 13.8,
        9.9, 8.2, 0.8, 9.0, 8.2, 8.0, 9.4, 0.0, 9.2, 8.2, 7.2, 8.7, 0.2, 0.0, 1.8, 0.6, 0.4, 0.2, 0.0, 0.0,
        1.0, 9.1, 1.9, 2.2, 0.4, 9.9, 1.2, 2.2, 1.4, 0.0, 9.0, 0.9, 2.8, 2.8, 3.0, 3.8, 3.8, 3.8, 3.8, 3.8,
        2.1, 0.0, 2.0, 3.2, 1.5, 0.8, 2.1, 3.2, 2.3, 0.0, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8
    };

    const int N = 20;


    //For SVD functions in cuSolver
    const int ROW_A = N;
    const int COL_A = N;
    const int LD_A = N;
    const float THREASHOLD = 1e-5;


    float *mtxA_d = NULL;
    float *mtxU_d = NULL;
    float *sngVals_d = NULL; // Singular values
    float *mtxV_d = NULL; // Need to allocate a separate memory buffer for the transposed matrix.
    float *mtxVT_d = NULL;



    bool debug = true;


    //(1) Allocate device memory
    CHECK((cudaMalloc((void**)&mtxA_d, ROW_A * COL_A *sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxU_d, LD_A * COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&sngVals_d, COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxVT_d, COL_A * COL_A * sizeof(float))));


    //(2) Copy value to device
    CHECK((cudaMemcpy(mtxA_d, mtxA, ROW_A * COL_A * sizeof(float), cudaMemcpyHostToDevice)));
    
    if(debug){
        printf("\n\n~~~MtxA~~~\n");
        print_mtx_clm_d(mtxA_d, ROW_A, COL_A);
    }




    //(3) Create handler
    cusolverDnHandle_t cusolverHandler = NULL;
    cublasHandle_t cublasHandler = NULL;

    checkCudaErrors(cusolverDnCreate(&cusolverHandler));
    checkCudaErrors(cublasCreate(&cublasHandler));



    // //(4) Compute SVD decomposition
    // checkCudaErrors(cusolverDnSgesvd(cusolverHandler, jobU, jobVT, ROW_A, COL_A, mtxA_d, LD_A, sngVals_d, mtxU_d,LD_A, mtxVT_d, COL_A, work_d, lwork, rwork_d, devInfo));
    SVD_Decmp(cusolverHandler, ROW_A, COL_A, LD_A, mtxA_d, mtxU_d, sngVals_d, mtxVT_d);

    //Transpose mtxVT -> mtxV
    //Matrix VT and matrix V are both n by n.
    mtxV_d = transpose_Den_Mtx(cublasHandler, mtxVT_d, COL_A, COL_A);



    if(debug){
        printf("\n\n👀Result of  SVD Decomp function👀\n\n");
        printf("\n\n~~mtxU_d\n");
        print_mtx_clm_d(mtxU_d, ROW_A, COL_A);
        printf("\n\n~~mtxD_d\n");
        print_mtx_clm_d(sngVals_d, COL_A, 1);
        printf("\n\n~~mtxV_d\n");
        print_mtx_clm_d(mtxV_d, COL_A, COL_A);
    }
    
    int newRank = setRank(sngVals_d, COL_A, THREASHOLD);

    if(newRank < COL_A){
        mtxV_d = truncate_Den_Mtx(mtxV_d, ROW_A, newRank);
        printf("\n\n~~mtxV_trnc_d\n");
        print_mtx_clm_d(mtxV_d, ROW_A, newRank);
    }

    checkCudaErrors(cusolverDnDestroy(cusolverHandler));
    checkCudaErrors(cublasDestroy(cublasHandler));

    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(mtxU_d));
    CHECK(cudaFree(sngVals_d));
    CHECK(cudaFree(mtxV_d));

    printf("\n= = = End of Case  = = = \n\n");


} // end of truncate_Den_Mtx_Case5



// Test for normlize 
void normalize_Den_Mtx_Case1()
{
   /*
            |1.0 1.0|
    mtxA =  |0.0 1.0|
            |1.0 0.0|

    expected
    mtxU = |0.8165  0      |
            |0.4082  0.7071 |
            |0.4082  -0.7071|

    Singular value = | 1.7321 |
                        | 1.00   |

    mtxVT = |-0.7071  0.7071|
            |0.7071   0.7071|

    */

    // Define the dense matrixB column major
    float mtxA[] = { 1.0, 0.0, 1.0, 1.0, 1.0, 0.0};

    //For SVD functions in cuSolver
    const int ROW_A = 3;
    const int COL_A = 2;
    const int LD_A = 3;
    const float THREASHOLD = 1e-5;

    float *mtxA_d = NULL;
    // float *mtxA_cpy_d = NULL;
    float *mtxY_d = NULL;
    float *mtxU_d = NULL;
    float *sngVals_d = NULL; // Singular values
    float *mtxV_d = NULL; // Need to allocate a separate memory buffer for the transposed matrix.
    float *mtxVT_d = NULL;
    float *mtxV_trnc_d = NULL;


    bool debug = true;


    //(1) Allocate device memory
    CHECK((cudaMalloc((void**)&mtxA_d, ROW_A * COL_A *sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxU_d, LD_A * COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&sngVals_d, COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxVT_d, COL_A * COL_A * sizeof(float))));


    //(2) Copy value to device
    CHECK((cudaMemcpy(mtxA_d, mtxA, ROW_A * COL_A * sizeof(float), cudaMemcpyHostToDevice)));
    // CHECK((cudaMemcpy(mtxA_cpy_d, mtxA, ROW_A * COL_A * sizeof(float), cudaMemcpyHostToDevice)));
    
    if(debug){
        printf("\n\n~~~MtxA~~~\n");
        print_mtx_clm_d(mtxA_d, ROW_A, COL_A);
    }




    //(3) Create handler
    cusolverDnHandle_t cusolverHandler = NULL;
    cublasHandle_t cublasHandler = NULL;

    checkCudaErrors(cusolverDnCreate(&cusolverHandler));
    checkCudaErrors(cublasCreate(&cublasHandler));



    // //(4) Compute SVD decomposition
    // checkCudaErrors(cusolverDnSgesvd(cusolverHandler, jobU, jobVT, ROW_A, COL_A, mtxA_d, LD_A, sngVals_d, mtxU_d,LD_A, mtxVT_d, COL_A, work_d, lwork, rwork_d, devInfo));
    SVD_Decmp(cusolverHandler, ROW_A, COL_A, LD_A, mtxA_d, mtxU_d, sngVals_d, mtxVT_d);


    //Transpose mtxVT -> mtxV
    //Matrix VT and matrix V are both n by n.
    mtxV_d = transpose_Den_Mtx(cublasHandler, mtxVT_d, COL_A, COL_A);


    if(debug){
        printf("\n\n👀Result of  SVD Decomp function👀\n\n");
        printf("\n\n~~mtxU_d\n");
        print_mtx_clm_d(mtxU_d, ROW_A, COL_A);
        printf("\n\n~~mtxD_d\n");
        print_mtx_clm_d(sngVals_d, COL_A, 1);
        printf("\n\n~~mtxV_d\n");
        print_mtx_clm_d(mtxV_d, COL_A, COL_A);
    }

    int newRank = setRank(sngVals_d, COL_A, THREASHOLD);


    mtxV_trnc_d = truncate_Den_Mtx(mtxV_d, COL_A, newRank);
    printf("\n\n~~mtxV_trnc_d\n");
    print_mtx_clm_d(mtxV_trnc_d, COL_A, newRank);

    //Set up matrix Y with new rankv for matrix multiplication reult 
    CHECK(cudaMalloc((void**)&mtxY_d, ROW_A * newRank *sizeof(float)));
    CHECK(cudaMemset(mtxY_d, 0, ROW_A * newRank * sizeof(float)));

    if(debug){
        printf("\n\n~~mtxY ~~\n");
        print_mtx_clm_d(mtxY_d, ROW_A, newRank);
        printf("\n\n~~mtxA ~~\n");
        print_mtx_clm_d(mtxA_d, ROW_A, COL_A);
        printf("\n\n~~mtxV_trnc ~~\n");
        print_mtx_clm_d(mtxV_trnc_d, COL_A, newRank);
    }


    //mtxY <- mtxZ * mtxV_trnc
    multiply_Den_ClmM_mtx_mtx(cublasHandler, mtxA_d, mtxV_trnc_d, mtxY_d, ROW_A, newRank, COL_A);
    if(debug){
        printf("\n\n~~mtxY ~~\n");
        print_mtx_clm_d(mtxY_d, ROW_A, newRank);
    }



    //Normalize mtxY
    normalize_Den_Mtx(mtxY_d, ROW_A, newRank);
    if(debug){
        printf("\n\n~~mtxY_hat~~\n");
        print_mtx_clm_d(mtxY_d, ROW_A, newRank);
    }


    //(6) Free memory
    checkCudaErrors(cusolverDnDestroy(cusolverHandler));
    checkCudaErrors(cublasDestroy(cublasHandler));

    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(mtxY_d));
    CHECK(cudaFree(mtxU_d));
    CHECK(cudaFree(sngVals_d));
    CHECK(cudaFree(mtxV_trnc_d));


    printf("\n= = = End of Case  = = = \n\n");
} // end of normalize_Den_Mtx_Case1()

void normalize_Den_Mtx_Case2()
 {

    // Define the dense matrixB column major
    float mtxA[] = {
    1.1, 0.8, 3.0, 2.2,
    2.2, 1.6, 4.1, 3.3,
    3.3, 2.4, 5.2, 4.4,
    4.4, 3.2, 6.3, 5.5
    };



    //For SVD functions in cuSolver
    const int N = 4; // N by N square matrix 
    const int ROW_A = N;
    const int COL_A = N;
    const int LD_A = N;
    const float THREASHOLD = 1e-5;
    
    
    float *mtxA_d = NULL;
    // float *mtxA_cpy_d = NULL;
    float *mtxY_d = NULL;
    float *mtxU_d = NULL;
    float *sngVals_d = NULL; // Singular values
    float *mtxV_d = NULL; // Need to allocate a separate memory buffer for the transposed matrix.
    float *mtxVT_d = NULL;
    float *mtxV_trnc_d = NULL;


    bool debug = true;


    //(1) Allocate device memory
    CHECK((cudaMalloc((void**)&mtxA_d, ROW_A * COL_A *sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxU_d, LD_A * COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&sngVals_d, COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxVT_d, COL_A * COL_A * sizeof(float))));


    //(2) Copy value to device
    CHECK((cudaMemcpy(mtxA_d, mtxA, ROW_A * COL_A * sizeof(float), cudaMemcpyHostToDevice)));
    // CHECK((cudaMemcpy(mtxA_cpy_d, mtxA, ROW_A * COL_A * sizeof(float), cudaMemcpyHostToDevice)));
    
    if(debug){
        printf("\n\n~~~MtxA~~~\n");
        print_mtx_clm_d(mtxA_d, ROW_A, COL_A);
    }




    //(3) Create handler
    cusolverDnHandle_t cusolverHandler = NULL;
    cublasHandle_t cublasHandler = NULL;

    checkCudaErrors(cusolverDnCreate(&cusolverHandler));
    checkCudaErrors(cublasCreate(&cublasHandler));



    // //(4) Compute SVD decomposition
    // checkCudaErrors(cusolverDnSgesvd(cusolverHandler, jobU, jobVT, ROW_A, COL_A, mtxA_d, LD_A, sngVals_d, mtxU_d,LD_A, mtxVT_d, COL_A, work_d, lwork, rwork_d, devInfo));
    SVD_Decmp(cusolverHandler, ROW_A, COL_A, LD_A, mtxA_d, mtxU_d, sngVals_d, mtxVT_d);


    //Transpose mtxVT -> mtxV
    //Matrix VT and matrix V are both n by n.
    mtxV_d = transpose_Den_Mtx(cublasHandler, mtxVT_d, COL_A, COL_A);


    if(debug){
        printf("\n\n👀Result of  SVD Decomp function👀\n\n");
        printf("\n\n~~mtxU_d\n");
        print_mtx_clm_d(mtxU_d, ROW_A, COL_A);
        printf("\n\n~~mtxD_d\n");
        print_mtx_clm_d(sngVals_d, COL_A, 1);
        printf("\n\n~~mtxV_d\n");
        print_mtx_clm_d(mtxV_d, COL_A, COL_A);
    }

    int newRank = setRank(sngVals_d, COL_A, THREASHOLD);


    mtxV_trnc_d = truncate_Den_Mtx(mtxV_d, COL_A, newRank);
    printf("\n\n~~mtxV_trnc_d\n");
    print_mtx_clm_d(mtxV_trnc_d, COL_A, newRank);

    //Set up matrix Y with new rankv for matrix multiplication reult 
    CHECK(cudaMalloc((void**)&mtxY_d, ROW_A * newRank *sizeof(float)));
    CHECK(cudaMemset(mtxY_d, 0, ROW_A * newRank * sizeof(float)));

    if(debug){
        printf("\n\n~~mtxY ~~\n");
        print_mtx_clm_d(mtxY_d, ROW_A, newRank);
        printf("\n\n~~mtxA ~~\n");
        print_mtx_clm_d(mtxA_d, ROW_A, COL_A);
        printf("\n\n~~mtxV_trnc ~~\n");
        print_mtx_clm_d(mtxV_trnc_d, COL_A, newRank);
    }


    //mtxY <- mtxZ * mtxV_trnc
    multiply_Den_ClmM_mtx_mtx(cublasHandler, mtxA_d, mtxV_trnc_d, mtxY_d, ROW_A, newRank, COL_A);
    if(debug){
        printf("\n\n~~mtxY ~~\n");
        print_mtx_clm_d(mtxY_d, ROW_A, newRank);
    }



    //Normalize mtxY
    normalize_Den_Mtx(mtxY_d, ROW_A, newRank);
    if(debug){
        printf("\n\n~~mtxY_hat~~\n");
        print_mtx_clm_d(mtxY_d, ROW_A, newRank);
    }


    //(6) Free memory
    checkCudaErrors(cusolverDnDestroy(cusolverHandler));
    checkCudaErrors(cublasDestroy(cublasHandler));

    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(mtxY_d));
    CHECK(cudaFree(mtxU_d));
    CHECK(cudaFree(sngVals_d));
    CHECK(cudaFree(mtxV_trnc_d));


    printf("\n= = = End of Case  = = = \n\n");
 }

void normalize_Den_Mtx_Case3()
{
    // Define the dense matrixB column major
    float mtxA[] = {
        1.1, 0.8, 3.0, 2.2, 1.4, 0.6, 2.0,
        2.2, 1.6, 4.1, 3.3, 2.5, 1.5, 2.9,
        3.3, 2.4, 5.2, 4.4, 3.6, 2.4, 3.8,
        4.4, 3.2, 6.3, 5.5, 4.7, 3.3, 4.7,
        5.5, 4.0, 7.4, 6.6, 5.8, 4.2, 5.6,
        6.6, 4.8, 8.5, 7.7, 6.9, 5.1, 6.5,
        7.7, 5.6, 9.6, 8.8, 7.0, 6.0, 7.4
    };



    //For SVD functions in cuSolver
    const int N = 7; // N by N square matrix 
    const int ROW_A = N;
    const int COL_A = N;
    const int LD_A = N;
    const float THREASHOLD = 1e-5;
      

       
    float *mtxA_d = NULL;
    // float *mtxA_cpy_d = NULL;
    float *mtxY_d = NULL;
    float *mtxU_d = NULL;
    float *sngVals_d = NULL; // Singular values
    float *mtxV_d = NULL; // Need to allocate a separate memory buffer for the transposed matrix.
    float *mtxVT_d = NULL;
    float *mtxV_trnc_d = NULL;


    bool debug = true;


    //(1) Allocate device memory
    CHECK((cudaMalloc((void**)&mtxA_d, ROW_A * COL_A *sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxU_d, LD_A * COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&sngVals_d, COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxVT_d, COL_A * COL_A * sizeof(float))));


    //(2) Copy value to device
    CHECK((cudaMemcpy(mtxA_d, mtxA, ROW_A * COL_A * sizeof(float), cudaMemcpyHostToDevice)));
    // CHECK((cudaMemcpy(mtxA_cpy_d, mtxA, ROW_A * COL_A * sizeof(float), cudaMemcpyHostToDevice)));
    
    if(debug){
        printf("\n\n~~~MtxA~~~\n");
        print_mtx_clm_d(mtxA_d, ROW_A, COL_A);
    }




    //(3) Create handler
    cusolverDnHandle_t cusolverHandler = NULL;
    cublasHandle_t cublasHandler = NULL;

    checkCudaErrors(cusolverDnCreate(&cusolverHandler));
    checkCudaErrors(cublasCreate(&cublasHandler));



    // //(4) Compute SVD decomposition
    // checkCudaErrors(cusolverDnSgesvd(cusolverHandler, jobU, jobVT, ROW_A, COL_A, mtxA_d, LD_A, sngVals_d, mtxU_d,LD_A, mtxVT_d, COL_A, work_d, lwork, rwork_d, devInfo));
    SVD_Decmp(cusolverHandler, ROW_A, COL_A, LD_A, mtxA_d, mtxU_d, sngVals_d, mtxVT_d);


    //Transpose mtxVT -> mtxV
    //Matrix VT and matrix V are both n by n.
    mtxV_d = transpose_Den_Mtx(cublasHandler, mtxVT_d, COL_A, COL_A);


    if(debug){
        printf("\n\n👀Result of  SVD Decomp function👀\n\n");
        printf("\n\n~~mtxU_d\n");
        print_mtx_clm_d(mtxU_d, ROW_A, COL_A);
        printf("\n\n~~mtxD_d\n");
        print_mtx_clm_d(sngVals_d, COL_A, 1);
        printf("\n\n~~mtxV_d\n");
        print_mtx_clm_d(mtxV_d, COL_A, COL_A);
    }

    int newRank = setRank(sngVals_d, COL_A, THREASHOLD);


    mtxV_trnc_d = truncate_Den_Mtx(mtxV_d, COL_A, newRank);
    printf("\n\n~~mtxV_trnc_d\n");
    print_mtx_clm_d(mtxV_trnc_d, COL_A, newRank);

    //Set up matrix Y with new rankv for matrix multiplication reult 
    CHECK(cudaMalloc((void**)&mtxY_d, ROW_A * newRank *sizeof(float)));
    CHECK(cudaMemset(mtxY_d, 0, ROW_A * newRank * sizeof(float)));

    if(debug){
        printf("\n\n~~mtxY ~~\n");
        print_mtx_clm_d(mtxY_d, ROW_A, newRank);
        printf("\n\n~~mtxA ~~\n");
        print_mtx_clm_d(mtxA_d, ROW_A, COL_A);
        printf("\n\n~~mtxV_trnc ~~\n");
        print_mtx_clm_d(mtxV_trnc_d, COL_A, newRank);
    }


    //mtxY <- mtxZ * mtxV_trnc
    multiply_Den_ClmM_mtx_mtx(cublasHandler, mtxA_d, mtxV_trnc_d, mtxY_d, ROW_A, newRank, COL_A);
    if(debug){
        printf("\n\n~~mtxY ~~\n");
        print_mtx_clm_d(mtxY_d, ROW_A, newRank);
    }



    //Normalize mtxY
    normalize_Den_Mtx(mtxY_d, ROW_A, newRank);
    if(debug){
        printf("\n\n~~mtxY_hat~~\n");
        print_mtx_clm_d(mtxY_d, ROW_A, newRank);
    }


    //(6) Free memory
    checkCudaErrors(cusolverDnDestroy(cusolverHandler));
    checkCudaErrors(cublasDestroy(cublasHandler));

    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(mtxY_d));
    CHECK(cudaFree(mtxU_d));
    CHECK(cudaFree(sngVals_d));
    CHECK(cudaFree(mtxV_trnc_d));


    printf("\n= = = End of Case  = = = \n\n");

}// end of normalize_Den_Mtx_Case3

void normalize_Den_Mtx_Case4()
{
    // Define the dense matrixB column major
    float mtxA[] = {
        1.2, 0.9, 3.1, 2.3, 1.5, 0.7, 2.1, 3.3, 2.5, 1.9,
        2.3, 1.8, 4.2, 3.4, 2.6, 1.6, 3.0, 4.4, 3.6, 2.8,
        3.4, 2.7, 5.3, 4.5, 3.7, 2.5, 4.9, 5.5, 4.7, 3.7,
        4.5, 3.6, 6.4, 5.6, 4.8, 3.4, 5.8, 6.6, 5.8, 4.6,
        5.6, 4.5, 7.5, 6.7, 5.9, 4.3, 6.7, 7.7, 6.9, 5.5,
        6.7, 5.4, 8.6, 7.8, 6.0, 5.2, 7.6, 8.8, 7.0, 6.4,
        7.8, 6.3, 9.7, 8.9, 7.1, 6.1, 8.5, 9.9, 8.1, 7.3,
        8.9, 7.2, 0.8, 9.0, 8.2, 7.0, 9.4, 0.0, 9.2, 8.2,
        9.0, 8.1, 1.9, 1.1, 9.3, 8.9, 0.3, 1.1, 0.3, 9.1,
        1.1, 9.0, 2.0, 2.2, 0.4, 9.8, 1.2, 2.2, 1.4, 0.0
    };


    //For SVD functions in cuSolver
    const int N = 10; // N by N square matrix 
    const int ROW_A = N;
    const int COL_A = N;
    const int LD_A = N;
    const float THREASHOLD = 1e-5;

       float *mtxA_d = NULL;
    // float *mtxA_cpy_d = NULL;
    float *mtxY_d = NULL;
    float *mtxU_d = NULL;
    float *sngVals_d = NULL; // Singular values
    float *mtxV_d = NULL; // Need to allocate a separate memory buffer for the transposed matrix.
    float *mtxVT_d = NULL;
    float *mtxV_trnc_d = NULL;


    bool debug = true;


    //(1) Allocate device memory
    CHECK((cudaMalloc((void**)&mtxA_d, ROW_A * COL_A *sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxU_d, LD_A * COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&sngVals_d, COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxVT_d, COL_A * COL_A * sizeof(float))));


    //(2) Copy value to device
    CHECK((cudaMemcpy(mtxA_d, mtxA, ROW_A * COL_A * sizeof(float), cudaMemcpyHostToDevice)));
    // CHECK((cudaMemcpy(mtxA_cpy_d, mtxA, ROW_A * COL_A * sizeof(float), cudaMemcpyHostToDevice)));
    
    if(debug){
        printf("\n\n~~~MtxA~~~\n");
        print_mtx_clm_d(mtxA_d, ROW_A, COL_A);
    }




    //(3) Create handler
    cusolverDnHandle_t cusolverHandler = NULL;
    cublasHandle_t cublasHandler = NULL;

    checkCudaErrors(cusolverDnCreate(&cusolverHandler));
    checkCudaErrors(cublasCreate(&cublasHandler));



    // //(4) Compute SVD decomposition
    // checkCudaErrors(cusolverDnSgesvd(cusolverHandler, jobU, jobVT, ROW_A, COL_A, mtxA_d, LD_A, sngVals_d, mtxU_d,LD_A, mtxVT_d, COL_A, work_d, lwork, rwork_d, devInfo));
    SVD_Decmp(cusolverHandler, ROW_A, COL_A, LD_A, mtxA_d, mtxU_d, sngVals_d, mtxVT_d);


    //Transpose mtxVT -> mtxV
    //Matrix VT and matrix V are both n by n.
    mtxV_d = transpose_Den_Mtx(cublasHandler, mtxVT_d, COL_A, COL_A);


    if(debug){
        printf("\n\n👀Result of  SVD Decomp function👀\n\n");
        printf("\n\n~~mtxU_d\n");
        print_mtx_clm_d(mtxU_d, ROW_A, COL_A);
        printf("\n\n~~mtxD_d\n");
        print_mtx_clm_d(sngVals_d, COL_A, 1);
        printf("\n\n~~mtxV_d\n");
        print_mtx_clm_d(mtxV_d, COL_A, COL_A);
    }

    int newRank = setRank(sngVals_d, COL_A, THREASHOLD);


    mtxV_trnc_d = truncate_Den_Mtx(mtxV_d, COL_A, newRank);
    printf("\n\n~~mtxV_trnc_d\n");
    print_mtx_clm_d(mtxV_trnc_d, COL_A, newRank);

    //Set up matrix Y with new rankv for matrix multiplication reult 
    CHECK(cudaMalloc((void**)&mtxY_d, ROW_A * newRank *sizeof(float)));
    CHECK(cudaMemset(mtxY_d, 0, ROW_A * newRank * sizeof(float)));

    if(debug){
        printf("\n\n~~mtxY ~~\n");
        print_mtx_clm_d(mtxY_d, ROW_A, newRank);
        printf("\n\n~~mtxA ~~\n");
        print_mtx_clm_d(mtxA_d, ROW_A, COL_A);
        printf("\n\n~~mtxV_trnc ~~\n");
        print_mtx_clm_d(mtxV_trnc_d, COL_A, newRank);
    }


    //mtxY <- mtxZ * mtxV_trnc
    multiply_Den_ClmM_mtx_mtx(cublasHandler, mtxA_d, mtxV_trnc_d, mtxY_d, ROW_A, newRank, COL_A);
    if(debug){
        printf("\n\n~~mtxY ~~\n");
        print_mtx_clm_d(mtxY_d, ROW_A, newRank);
    }



    //Normalize mtxY
    normalize_Den_Mtx(mtxY_d, ROW_A, newRank);
    if(debug){
        printf("\n\n~~mtxY_hat~~\n");
        print_mtx_clm_d(mtxY_d, ROW_A, newRank);
    }


    //(6) Free memory
    checkCudaErrors(cusolverDnDestroy(cusolverHandler));
    checkCudaErrors(cublasDestroy(cublasHandler));

    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(mtxY_d));
    CHECK(cudaFree(mtxU_d));
    CHECK(cudaFree(sngVals_d));
    CHECK(cudaFree(mtxV_trnc_d));


    printf("\n= = = End of Case  = = = \n\n");

} // end of normalize_Den_Mtx_Case4

void normalize_Den_Mtx_Case5()
{

    // Define the dense matrixB column major
    float mtxA[] = {
        1.2, 0.9, 3.1, 2.3, 1.5, 0.7, 2.1, 3.3, 2.5, 1.9, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
        2.3, 1.8, 4.2, 3.4, 2.6, 1.6, 3.0, 4.4, 3.6, 2.8, 1.8, 2.1, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8,
        3.4, 2.7, 5.3, 4.5, 3.7, 2.5, 4.9, 5.5, 4.7, 3.7, 2.7, 3.2, 3.7, 4.0, 4.3, 4.6, 4.9, 5.2, 5.5, 5.8,
        4.5, 3.6, 6.4, 5.6, 4.8, 3.4, 5.8, 6.6, 5.8, 4.6, 3.6, 4.3, 5.0, 5.4, 5.8, 6.2, 6.6, 7.0, 7.4, 7.8,
        5.6, 4.5, 7.5, 6.7, 5.9, 4.3, 6.7, 7.7, 6.9, 5.5, 4.5, 5.4, 6.3, 6.8, 7.3, 7.8, 8.3, 8.8, 9.3, 9.8,
        6.7, 5.4, 8.6, 7.8, 6.0, 5.2, 7.6, 8.8, 7.0, 6.4, 5.4, 6.5, 7.6, 8.2, 8.8, 9.4, 10.0, 10.6, 11.2, 11.8,
        7.8, 6.3, 9.7, 8.9, 7.1, 6.1, 8.5, 9.9, 8.1, 7.3, 6.3, 7.6, 8.9, 9.6, 10.3, 11.0, 11.7, 12.4, 13.1, 13.8,
        8.9, 7.2, 0.8, 9.0, 8.2, 7.0, 9.4, 0.0, 9.2, 8.2, 7.2, 8.7, 0.2, 0.0, 1.8, 0.6, 0.4, 0.2, 0.0, 0.0,
        9.0, 8.1, 1.9, 1.1, 9.3, 8.9, 0.3, 1.1, 0.3, 9.1, 8.1, 9.8, 1.5, 1.4, 2.4, 2.2, 2.1, 2.0, 1.9, 1.9,
        1.1, 9.0, 2.0, 2.2, 0.4, 9.8, 1.2, 2.2, 1.4, 0.0, 9.0, 0.9, 2.8, 2.8, 3.0, 3.8, 3.8, 3.8, 3.8, 3.8,
        2.2, 1.9, 3.1, 3.3, 1.5, 1.7, 2.1, 3.3, 2.5, 1.9, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
        3.3, 2.8, 4.2, 4.4, 2.6, 2.6, 3.0, 4.4, 3.6, 2.8, 1.8, 2.1, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8,
        4.4, 3.7, 5.3, 5.5, 3.7, 3.5, 4.9, 5.5, 4.7, 3.7, 2.7, 3.2, 3.7, 4.0, 4.3, 4.6, 4.9, 5.2, 5.5, 5.8,
        5.5, 4.6, 6.4, 6.6, 4.8, 4.4, 5.8, 6.6, 5.8, 4.6, 3.6, 4.3, 5.0, 5.4, 5.8, 6.2, 6.6, 7.0, 7.4, 7.8,
        6.6, 5.5, 7.5, 7.7, 5.9, 5.3, 6.7, 7.7, 6.9, 5.5, 4.5, 5.4, 6.3, 6.8, 7.3, 7.8, 8.3, 8.8, 9.3, 9.8,
        7.7, 6.4, 8.6, 8.8, 6.0, 6.2, 7.6, 8.8, 7.0, 6.4, 5.4, 6.5, 7.6, 8.2, 8.8, 9.4, 10.0, 10.6, 11.2, 11.8,
        8.8, 7.3, 9.7, 9.9, 7.1, 7.1, 8.5, 9.9, 8.1, 7.3, 6.3, 7.6, 8.9, 9.6, 10.3, 11.0, 11.7, 12.4, 13.1, 13.8,
        9.9, 8.2, 0.8, 9.0, 8.2, 8.0, 9.4, 0.0, 9.2, 8.2, 7.2, 8.7, 0.2, 0.0, 1.8, 0.6, 0.4, 0.2, 0.0, 0.0,
        1.0, 9.1, 1.9, 2.2, 0.4, 9.9, 1.2, 2.2, 1.4, 0.0, 9.0, 0.9, 2.8, 2.8, 3.0, 3.8, 3.8, 3.8, 3.8, 3.8,
        2.1, 0.0, 2.0, 3.2, 1.5, 0.8, 2.1, 3.2, 2.3, 0.0, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8
    };

    const int N = 20;


    //For SVD functions in cuSolver
    const int ROW_A = N;
    const int COL_A = N;
    const int LD_A = N;
    const float THREASHOLD = 1e-5;
      

    float *mtxA_d = NULL;
    // float *mtxA_cpy_d = NULL;
    float *mtxY_d = NULL;
    float *mtxU_d = NULL;
    float *sngVals_d = NULL; // Singular values
    float *mtxV_d = NULL; // Need to allocate a separate memory buffer for the transposed matrix.
    float *mtxVT_d = NULL;
    float *mtxV_trnc_d = NULL;


    bool debug = true;


    //(1) Allocate device memory
    CHECK((cudaMalloc((void**)&mtxA_d, ROW_A * COL_A *sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxU_d, LD_A * COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&sngVals_d, COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxVT_d, COL_A * COL_A * sizeof(float))));


    //(2) Copy value to device
    CHECK((cudaMemcpy(mtxA_d, mtxA, ROW_A * COL_A * sizeof(float), cudaMemcpyHostToDevice)));
    // CHECK((cudaMemcpy(mtxA_cpy_d, mtxA, ROW_A * COL_A * sizeof(float), cudaMemcpyHostToDevice)));
    
    if(debug){
        printf("\n\n~~~MtxA~~~\n");
        print_mtx_clm_d(mtxA_d, ROW_A, COL_A);
    }




    //(3) Create handler
    cusolverDnHandle_t cusolverHandler = NULL;
    cublasHandle_t cublasHandler = NULL;

    checkCudaErrors(cusolverDnCreate(&cusolverHandler));
    checkCudaErrors(cublasCreate(&cublasHandler));



    // //(4) Compute SVD decomposition
    // checkCudaErrors(cusolverDnSgesvd(cusolverHandler, jobU, jobVT, ROW_A, COL_A, mtxA_d, LD_A, sngVals_d, mtxU_d,LD_A, mtxVT_d, COL_A, work_d, lwork, rwork_d, devInfo));
    SVD_Decmp(cusolverHandler, ROW_A, COL_A, LD_A, mtxA_d, mtxU_d, sngVals_d, mtxVT_d);


    //Transpose mtxVT -> mtxV
    //Matrix VT and matrix V are both n by n.
    mtxV_d = transpose_Den_Mtx(cublasHandler, mtxVT_d, COL_A, COL_A);


    if(debug){
        printf("\n\n👀Result of  SVD Decomp function👀\n\n");
        printf("\n\n~~mtxU_d\n");
        print_mtx_clm_d(mtxU_d, ROW_A, COL_A);
        printf("\n\n~~mtxD_d\n");
        print_mtx_clm_d(sngVals_d, COL_A, 1);
        printf("\n\n~~mtxV_d\n");
        print_mtx_clm_d(mtxV_d, COL_A, COL_A);
    }

    int newRank = setRank(sngVals_d, COL_A, THREASHOLD);


    mtxV_trnc_d = truncate_Den_Mtx(mtxV_d, COL_A, newRank);
    printf("\n\n~~mtxV_trnc_d\n");
    print_mtx_clm_d(mtxV_trnc_d, COL_A, newRank);

    //Set up matrix Y with new rankv for matrix multiplication reult 
    CHECK(cudaMalloc((void**)&mtxY_d, ROW_A * newRank *sizeof(float)));
    CHECK(cudaMemset(mtxY_d, 0, ROW_A * newRank * sizeof(float)));

    if(debug){
        printf("\n\n~~mtxY ~~\n");
        print_mtx_clm_d(mtxY_d, ROW_A, newRank);
        printf("\n\n~~mtxA ~~\n");
        print_mtx_clm_d(mtxA_d, ROW_A, COL_A);
        printf("\n\n~~mtxV_trnc ~~\n");
        print_mtx_clm_d(mtxV_trnc_d, COL_A, newRank);
    }


    //mtxY <- mtxZ * mtxV_trnc
    multiply_Den_ClmM_mtx_mtx(cublasHandler, mtxA_d, mtxV_trnc_d, mtxY_d, ROW_A, newRank, COL_A);
    if(debug){
        printf("\n\n~~mtxY ~~\n");
        print_mtx_clm_d(mtxY_d, ROW_A, newRank);
    }



    //Normalize mtxY
    normalize_Den_Mtx(mtxY_d, ROW_A, newRank);
    if(debug){
        printf("\n\n~~mtxY_hat~~\n");
        print_mtx_clm_d(mtxY_d, ROW_A, newRank);
    }


    //(6) Free memory
    checkCudaErrors(cusolverDnDestroy(cusolverHandler));
    checkCudaErrors(cublasDestroy(cublasHandler));

    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(mtxY_d));
    CHECK(cudaFree(mtxU_d));
    CHECK(cudaFree(sngVals_d));
    CHECK(cudaFree(mtxV_trnc_d));


    printf("\n= = = End of Case  = = = \n\n");

}// end of normalize_Den_Mtx_Case5





// Test for orth
void orth_test_Case1()
{
    /*
    Z = | 1.0  5.0  9.0 |
        | 2.0  6.0  10.0|
        | 3.0  7.0  11.0|
        | 4.0  8.0  12.0| 
    */

    // Define the dense matrixB column major
    float mtxZ[] = {
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0,
    };

    int numOfRow = 4;
    int numOfClm = 3;
    int crntRank = 3;

    float* mtxZ_d = NULL;
    float* mtxY_hat_d = NULL;

    bool debug = true;


    CHECK(cudaMalloc((void**)&mtxZ_d, numOfRow * numOfClm * sizeof(float)));
    CHECK(cudaMemcpy(mtxZ_d, mtxZ, numOfRow * numOfClm * sizeof(float), cudaMemcpyHostToDevice));

    if(debug){
        printf("\n\n~~mtxZ~~\n\n");
        print_mtx_clm_d(mtxZ_d, numOfRow, numOfClm);
    }


    orth(&mtxY_hat_d, mtxZ_d, numOfRow, numOfClm, crntRank);

    if(mtxY_hat_d != NULL){
        printf("\n\n~~mtxY_Hat~~\n\n");
        print_mtx_clm_d(mtxY_hat_d, numOfRow, crntRank);
    }
    
    printf("\n\n~~Current Rarnk = %d~~\n\n", crntRank);

}// end of orth_test_Case1()

void orth_test_Case2()
{
    // Define the dense matrixB column major
    float mtxZ[] = {
    1.1, 0.8, 3.0, 2.2,
    2.2, 1.6, 4.1, 3.3,
    3.3, 2.4, 5.2, 4.4,
    4.4, 3.2, 6.3, 5.5,
    5.5, 2.3, 0.7, 1.7 
    };

    int numOfRow = 5;
    int numOfClm = 4;
    int crntRank = 4;

    float* mtxZ_d = NULL;
    float* mtxY_hat_d = NULL;

    bool debug = true;


    CHECK(cudaMalloc((void**)&mtxZ_d, numOfRow * numOfClm * sizeof(float)));
    CHECK(cudaMemcpy(mtxZ_d, mtxZ, numOfRow * numOfClm * sizeof(float), cudaMemcpyHostToDevice));

    if(debug){
        printf("\n\n~~mtxZ~~\n\n");
        print_mtx_clm_d(mtxZ_d, numOfRow, numOfClm);
    }


    orth(&mtxY_hat_d, mtxZ_d, numOfRow, numOfClm, crntRank);

    printf("\n\n~~mtxY_Hat~~\n\n");
    print_mtx_clm_d(mtxY_hat_d, numOfRow, crntRank);
    printf("\n\n~~Current Rarnk = %d~~\n\n", crntRank);
}// end of orth_test_Case2()


void orth_test_Case3()
{
    // Define the dense matrixB column major
    float mtxZ[] = {
        1.1, 0.8, 3.0, 2.2, 1.4, 0.6,
        2.2, 1.6, 4.1, 3.3, 2.5, 1.5,
        3.3, 2.4, 5.2, 4.4, 3.6, 2.4,
        4.4, 3.2, 6.3, 5.5, 4.7, 3.3,
        5.5, 4.0, 7.4, 6.6, 5.8, 4.2,
        6.6, 4.8, 8.5, 7.7, 6.9, 5.1, 
        7.7, 5.6, 9.6, 8.8, 7.0, 6.0
    };

    int numOfRow = 7;
    int numOfClm = 6;
    int crntRank = 6;

    float* mtxZ_d = NULL;
    float* mtxY_hat_d = NULL;

    bool debug = true;


    CHECK(cudaMalloc((void**)&mtxZ_d, numOfRow * numOfClm * sizeof(float)));
    CHECK(cudaMemcpy(mtxZ_d, mtxZ, numOfRow * numOfClm * sizeof(float), cudaMemcpyHostToDevice));

    if(debug){
        printf("\n\n~~mtxZ~~\n\n");
        print_mtx_clm_d(mtxZ_d, numOfRow, numOfClm);
    }


    orth(&mtxY_hat_d, mtxZ_d, numOfRow, numOfClm, crntRank);

    printf("\n\n~~mtxY_Hat~~\n\n");
    print_mtx_clm_d(mtxY_hat_d, numOfRow, crntRank);
    printf("\n\n~~Current Rarnk = %d~~\n\n", crntRank);

}// end of orth_test_Case3()

void orth_test_Case4()
{
    // Define the dense matrixB column major
    float mtxZ[] = {
        1.2, 0.9, 3.1, 2.3, 1.5, 0.7, 2.1, 3.3,
        2.3, 1.8, 4.2, 3.4, 2.6, 1.6, 3.0, 4.4,
        3.4, 2.7, 5.3, 4.5, 3.7, 2.5, 4.9, 5.5,
        4.5, 3.6, 6.4, 5.6, 4.8, 3.4, 5.8, 6.6,
        5.6, 4.5, 7.5, 6.7, 5.9, 4.3, 6.7, 7.7,
        6.7, 5.4, 8.6, 7.8, 6.0, 5.2, 7.6, 8.8,
        7.8, 6.3, 9.7, 8.9, 7.1, 6.1, 8.5, 9.9,
        8.9, 7.2, 0.8, 9.0, 8.2, 7.0, 9.4, 0.0,
        9.0, 8.1, 1.9, 1.1, 9.3, 8.9, 0.3, 1.1,
        1.1, 9.0, 2.0, 2.2, 0.4, 9.8, 1.2, 2.2
    };


    int numOfRow = 10;
    int numOfClm = 8;
    int crntRank = 8;

    float* mtxZ_d = NULL;
    float* mtxY_hat_d = NULL;

    bool debug = true;


    CHECK(cudaMalloc((void**)&mtxZ_d, numOfRow * numOfClm * sizeof(float)));
    CHECK(cudaMemcpy(mtxZ_d, mtxZ, numOfRow * numOfClm * sizeof(float), cudaMemcpyHostToDevice));

    if(debug){
        printf("\n\n~~mtxZ~~\n\n");
        print_mtx_clm_d(mtxZ_d, numOfRow, numOfClm);
    }


    orth(&mtxY_hat_d, mtxZ_d, numOfRow, numOfClm, crntRank);

    printf("\n\n~~mtxY_Hat~~\n\n");
    print_mtx_clm_d(mtxY_hat_d, numOfRow, crntRank);
    printf("\n\n~~Current Rarnk = %d~~\n\n", crntRank);

}// end of orth_test_Case4()

void orth_test_Case5()
{

    // Define the dense matrixB column major
    float mtxZ[] = {
        1.2, 0.9, 3.1, 2.3, 1.5, 0.7, 2.1, 3.3, 2.5, 1.9, 0.9, 1.0, 1.1, 1.2, 1.3, 
        2.3, 1.8, 4.2, 3.4, 2.6, 1.6, 3.0, 4.4, 3.6, 2.8, 1.8, 2.1, 2.4, 2.6, 2.8, 
        3.4, 2.7, 5.3, 4.5, 3.7, 2.5, 4.9, 5.5, 4.7, 3.7, 2.7, 3.2, 3.7, 4.0, 4.3, 
        4.5, 3.6, 6.4, 5.6, 4.8, 3.4, 5.8, 6.6, 5.8, 4.6, 3.6, 4.3, 5.0, 5.4, 5.8, 
        5.6, 4.5, 7.5, 6.7, 5.9, 4.3, 6.7, 7.7, 6.9, 5.5, 4.5, 5.4, 6.3, 6.8, 7.3, 
        6.7, 5.4, 8.6, 7.8, 6.0, 5.2, 7.6, 8.8, 7.0, 6.4, 5.4, 6.5, 7.6, 8.2, 8.8, 
        7.8, 6.3, 9.7, 8.9, 7.1, 6.1, 8.5, 9.9, 8.1, 7.3, 6.3, 7.6, 8.9, 9.6, 10.3,
        8.9, 7.2, 0.8, 9.0, 8.2, 7.0, 9.4, 0.0, 9.2, 8.2, 7.2, 8.7, 0.2, 0.0, 1.8, 
        9.0, 8.1, 1.9, 1.1, 9.3, 8.9, 0.3, 1.1, 0.3, 9.1, 8.1, 9.8, 1.5, 1.4, 2.4, 
        1.1, 9.0, 2.0, 2.2, 0.4, 9.8, 1.2, 2.2, 1.4, 0.0, 9.0, 0.9, 2.8, 2.8, 3.0, 
        2.2, 1.9, 3.1, 3.3, 1.5, 1.7, 2.1, 3.3, 2.5, 1.9, 0.9, 1.0, 1.1, 1.2, 1.3, 
        3.3, 2.8, 4.2, 4.4, 2.6, 2.6, 3.0, 4.4, 3.6, 2.8, 1.8, 2.1, 2.4, 2.6, 2.8, 
        4.4, 3.7, 5.3, 5.5, 3.7, 3.5, 4.9, 5.5, 4.7, 3.7, 2.7, 3.2, 3.7, 4.0, 4.3, 
        5.5, 4.6, 6.4, 6.6, 4.8, 4.4, 5.8, 6.6, 5.8, 4.6, 3.6, 4.3, 5.0, 5.4, 5.8, 
        6.6, 5.5, 7.5, 7.7, 5.9, 5.3, 6.7, 7.7, 6.9, 5.5, 4.5, 5.4, 6.3, 6.8, 7.3, 
        7.7, 6.4, 8.6, 8.8, 6.0, 6.2, 7.6, 8.8, 7.0, 6.4, 5.4, 6.5, 7.6, 8.2, 8.8, 
        8.8, 7.3, 9.7, 9.9, 7.1, 7.1, 8.5, 9.9, 8.1, 7.3, 6.3, 7.6, 8.9, 9.6, 10.3,
        9.9, 8.2, 0.8, 9.0, 8.2, 8.0, 9.4, 0.0, 9.2, 8.2, 7.2, 8.7, 0.2, 0.0, 1.8, 
        1.0, 9.1, 1.9, 2.2, 0.4, 9.9, 1.2, 2.2, 1.4, 0.0, 9.0, 0.9, 2.8, 2.8, 3.0, 
        2.1, 0.0, 2.0, 3.2, 1.5, 0.8, 2.1, 3.2, 2.3, 0.0, 0.9, 1.0, 1.1, 1.2, 1.3
    };

    int numOfRow = 20;
    int numOfClm = 15;
    int crntRank = 15;

    float* mtxZ_d = NULL;
    float* mtxY_hat_d = NULL;

    bool debug = true;


    CHECK(cudaMalloc((void**)&mtxZ_d, numOfRow * numOfClm * sizeof(float)));
    CHECK(cudaMemcpy(mtxZ_d, mtxZ, numOfRow * numOfClm * sizeof(float), cudaMemcpyHostToDevice));

    if(debug){
        printf("\n\n~~mtxZ~~\n\n");
        print_mtx_clm_d(mtxZ_d, numOfRow, numOfClm);
    }


    orth(&mtxY_hat_d, mtxZ_d, numOfRow, numOfClm, crntRank);

    printf("\n\n~~mtxY_Hat~~\n\n");
    print_mtx_clm_d(mtxY_hat_d, numOfRow, crntRank);
    printf("\n\n~~Current Rarnk = %d~~\n\n", crntRank);

}// end of orth_test_Case5()



#endif // ORTH_TEST_CASES_H
