#ifndef SVD_DECOMP_TEST_CASES_H
#define SVD_DECOMP_TEST_CASES_H


#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>



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
    float mtxU_h[] = {0.8165, 0.4082, 0.4082, 0, 0.7071, -0.7071};
    float sngVals_h[] = {1.7321, 1.000};
    float mtxVT_h[] = {-0.7071, 0.7071, 0.7071, 0.7071};


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

    if(debug){
        printf("\n\n👀Result of  SVD Decomp function👀\n\n");
        printf("\n\n~~mtxU_d\n");
        print_mtx_clm_d(mtxU_d, ROW_A, COL_A);
        printf("\n\n~~mtxD_d\n");
        print_mtx_clm_d(sngVals_d, COL_A, 1);
        printf("\n\n~~mtxVT_d\n");
        print_mtx_clm_d(mtxVT_d, COL_A, COL_A);
    }

    if(debug){
        printf("\n\n🧐🧐🧐From MATLAB and expected values🧐🧐🧐");
        printf("\n\n~~mtxU_h~~\n");
        print_mtx_clm_h(mtxU_h, ROW_A, COL_A);
        printf("\n\n~~sngVals_h~~\n");
        print_mtx_clm_h(sngVals_h, COL_A, 1);
        printf("\n\n~~mtxVT_h~~\n");
        print_mtx_clm_h(mtxVT_h, COL_A, COL_A);
        printf("\n\n= = = END OF CASE = = = = =\n\n");
    }

    checkCudaErrors(cusolverDnDestroy(cusolverHandler));
    checkCudaErrors(cublasDestroy(cublasHandler));

    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(mtxU_d));
    CHECK(cudaFree(sngVals_d));
    CHECK(cudaFree(mtxVT_d));
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

    float mtxU_h[] = {0, 0, 0, 0, 0, 0};
    float sngVals_h[] = {0, 0};
    float mtxVT_h[] = {0, 0, 0, 0};


    //For SVD functions in cuSolver
    const int N = 4; // N by N square matrix 
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

    if(debug){
        printf("\n\n👀Result of  SVD Decomp function👀\n\n");
        printf("\n\n~~mtxU_d\n");
        print_mtx_clm_d(mtxU_d, ROW_A, COL_A);
        printf("\n\n~~mtxD_d\n");
        print_mtx_clm_d(sngVals_d, COL_A, 1);
        printf("\n\n~~mtxVT_d\n");
        print_mtx_clm_d(mtxVT_d, COL_A, COL_A);
    }

    debug = false;
    if(debug){
        printf("\n\n🧐🧐🧐From MATLAB and expected values🧐🧐🧐");
        printf("\n\n~~mtxU_h~~\n");
        print_mtx_clm_h(mtxU_h, ROW_A, COL_A);
        printf("\n\n~~sngVals_h~~\n");
        print_mtx_clm_h(sngVals_h, COL_A, 1);
        printf("\n\n~~mtxVT_h~~\n");
        print_mtx_clm_h(mtxVT_h, COL_A, COL_A);
        printf("\n\n= = = END OF CASE = = = = =\n\n");
    }

    checkCudaErrors(cusolverDnDestroy(cusolverHandler));
    checkCudaErrors(cublasDestroy(cublasHandler));

    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(mtxU_d));
    CHECK(cudaFree(sngVals_d));
    CHECK(cudaFree(mtxVT_d));
} // end of case 2










#endif // SVD_DECOMP_TEST_CASES_H
