#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <cmath>
#include<sys/time.h>

// helper function CUDA error checking and initialization
#include "helper_cuda.h"  

#define CHECK(call){ \
    const cudaError_t cuda_ret = call; \
    if(cuda_ret != cudaSuccess){ \
        printf("Error: %s:%d,  ", __FILE__, __LINE__ );\
        printf("code: %d, reason: %s \n", cuda_ret, cudaGetErrorString(cuda_ret));\
        exit(-1); \
    }\
}




//Generate random SPD dense matrix
// N is matrix size
float* generateSPD_DenseMatrix(int N){
	float* mtx_h = NULL;
	float* mtx_d = NULL;
	float* mtxSPD_h = NULL;
	float* mtxSPD_d = NULL;

	//Using for cublas function
	const float alpha = 1.0f;
	const float beta = 0.0f;

	//Allocate memoery in Host
	mtx_h = (float*)malloc(sizeof(float)* (N*N));
	mtxSPD_h = (float*)malloc(sizeof(float)* (N*N));

	if(! mtx_h || ! mtxSPD_h){
		printf("\nFailed to allocate memory in host\n\n");
		return NULL;
	}

	// Seed the random number generator
	srand(static_cast<unsigned>(time(0)));

	// Generate and store to mtx_h in all elements random values between 0 and 1.
	for (int wkr = 0; wkr < N*N;  wkr++){
		if(wkr % N == 0){
			printf("\n");
		}
		mtx_h[wkr] = ((float)rand()/RAND_MAX);
		// printf("\nmtx_h[%d] = %f", wkr, mtx_h[wkr]);

	}



	//(1)Allocate memoery in device
	CHECK(cudaMalloc((void**)&mtx_d, sizeof(float) * (N*N)));
	CHECK(cudaMalloc((void**)&mtxSPD_d, sizeof(float) * (N*N)));

	//(2) Copy value from host to device
	CHECK(cudaMemcpy(mtx_d, mtx_h, sizeof(float)* (N*N), cudaMemcpyHostToDevice));

	//(3) Calculate SPD matrix <- A' * A
	// Create a cuBLAS handle
	cublasHandle_t handle;
	cublasCreate(&handle);
	checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, N, &alpha, mtx_d, N, mtx_d, N, &beta, mtxSPD_d, N));

	//(4) Copy value from device to host
	CHECK(cudaMemcpy(mtxSPD_h, mtxSPD_d, sizeof(float) * (N*N), cudaMemcpyDeviceToHost));
	
	//(5) Free memeory
	cudaFree(mtx_d);
	cudaFree(mtxSPD_d);
	cublasDestroy(handle);
	free(mtx_h);


	return mtxSPD_h;
} // enf of generateSPD_DenseMatrix


// N is matrix size
float* generate_TriDiagMatrix(int N){

	//Allocate memoery in Host
	float* mtx_h = (float*)calloc(N*N, sizeof(float));


	if(! mtx_h){
		printf("\nFailed to allocate memory in host\n\n");
		return NULL;
	}

	// Seed the random number generator
	srand(static_cast<unsigned>(time(0)));

	// Generate and store to mtx_h tridiagonal random values between 0 and 1.
	mtx_h[0] = ((float)rand()/RAND_MAX)+10.0f;
	mtx_h[1] = ((float)rand()/RAND_MAX);
	for (int wkr = 1; wkr < N -1 ;  wkr++){
		mtx_h[(wkr * N) + (wkr-1)] =((float)rand()/RAND_MAX);
		mtx_h[wkr * N + wkr] = ((float)rand()/RAND_MAX)+10.0f;
		mtx_h[(wkr * N) + (wkr+1)] = ((float)rand()/RAND_MAX);
	}
	mtx_h[(N*(N-1))+ (N-2)] = ((float)rand()/RAND_MAX);
	mtx_h[(N*(N-1)) + (N-1)] = ((float)rand()/RAND_MAX) + 10.0f;
	
	// printf("\nTridiagonal Mtx: \n");
	// for(int rw_wkr = 0; rw_wkr < N; rw_wkr++){
	// for(int clm_wkr = 0; clm_wkr < N; clm_wkr++){
	// 	printf("%f ", mtx_h[rw_wkr*N + clm_wkr]);
	// }
	// printf("\n");
	// }
	// printf("\n\n");

	//Scale down
	for (int i = 0; i < N * N; i++) {
        mtx_h[i] /= 10;
    }

	return mtx_h;
} // enf of generate_TriDiagMatrix




void validateSol(const float *mtxA_h, const float* x_h, float* rhs, int N){
    float rsum, diff, error = 0.0f;

    for (int rw_wkr = 0; rw_wkr < N; rw_wkr++){
        rsum = 0.0f;
        for(int clm_wkr = 0; clm_wkr < N; clm_wkr++){
            rsum += mtxA_h[rw_wkr*N + clm_wkr]* x_h[clm_wkr];
            // printf("\nrsum = %f", rsum);
        } // end of inner loop
        diff = fabs(rsum - rhs[rw_wkr]);
        if(diff > error){
            error = diff;
        }
        
    }// end of outer loop
    
    printf("\n\nTest Summary: Error amount = %f\n", error);

}// end of validateSol




//Orth functions
void SVD_Decmp(cusolverDnHandle_t cusolverHandler, int numOfRow, int numOfClm, int ldngDim, float* mtxA_d, float* mtxU_d, float* sngVals_d, float*mtxVT_d)
{	

	/*The devInfo is an integer pointer
    It points to device memory where cuSOLVER can store information 
    about the success or failure of the computation.*/
    int *devInfo = NULL;

    int lwork = 0;//Size of workspace
    //work_d is a pointer to device memory that serves as the workspace for the computation
    //Then passed to the cuSOLVER function performing the computation.
    float *work_d = NULL; // 
    float *rwork_d = NULL; // Place holder
    

    //Specifies options for computing all or part of the matrix U: = ‘A’: 
    //all m columns of U are returned in array
    signed char jobU = 'S';

    //Specifies options for computing all or part of the matrix V**T: = ‘A’: 
    //all N rows of V**T are returned in the array
    signed char jobVT = 'S';



	//(1) Allocate memoery for devInfo
	CHECK((cudaMalloc((void**)&devInfo, sizeof(int))));

	//(2) Calculate workspace for SVD decompositoin
	checkCudaErrors(cusolverDnSgesvd_bufferSize(cusolverHandler, numOfRow, numOfClm, &lwork));
    CHECK(cudaMalloc((void**)&work_d, lwork * sizeof(float)));


    //(3) Compute SVD decomposition
    checkCudaErrors(cusolverDnSgesvd(cusolverHandler, jobU, jobVT, numOfRow, numOfClm, mtxA_d, ldngDim, sngVals_d, mtxU_d,ldngDim, mtxVT_d, numOfClm, work_d, lwork, rwork_d, devInfo));

	return;
}

int setRank(float* sngVals_d, int currentRank, float threashold)
{
	return 0;
}

float* truncate_Den_Mtx(float* mtxV_d, int numOfN, int currentRank)
{
	return NULL;
}

void normalize_mtx_d(float* mtxY_d, int numOfRow, int numOfCol)
{
	return;
}

 __global__ void normalizeClmVec(float* mtxY_d, int numOfRow, int numOfCol)
 {
	return;
 }

#endif // HELPER_FUNCTIONS_H