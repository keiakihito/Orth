// includes, system
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

/*Using updated (v2) interfaces to cublas*/
#include<cublas_v2.h>
#include<cuda_runtime.h>
#include<cusparse.h>
#include <cusolverDn.h>
#include<sys/time.h>


//Utilities
#include "includes/helper_debug.h"
// helper function CUDA error checking and initialization
#include "includes/helper_cuda.h"  
#include "includes/helper_functions.h"
#include "test_cases/orth_test_cases.h"

#define CHECK(call){ \
    const cudaError_t cuda_ret = call; \
    if(cuda_ret != cudaSuccess){ \
        printf("Error: %s:%d,  ", __FILE__, __LINE__ );\
        printf("code: %d, reason: %s \n", cuda_ret, cudaGetErrorString(cuda_ret));\
        exit(-1); \
    }\
}

// Time tracker for each iteration
double myCPUTimer()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec/1.0e6);
}



//Functions signatures
void SVD_Decomp_Test();
void setRank_Test();
void truncate_Den_Mtx_Test();
void normalize_mtx_d_Test();
void orth_test();



int main(int argc, char** argv)
{
    // SVD_Decomp_Test();

    // setRank_Test();

    // truncate_Den_Mtx_Test();

    // normalize_mtx_d_Test();

    orth_test();


    
}// end of main



//Function definitions
void SVD_Decomp_Test()
{   
    printf("\n\n~~SVD_Decomp_Test()~~\n\n");
    
    printf("\n\n🔍🔍🔍Test case 1🔍🔍🔍\n");
    SVD_Decomp_Case1();

    printf("\n\n🔍🔍🔍Test case 2🔍🔍🔍\n");
    SVD_Decomp_Case2();

    printf("\n\n🔍🔍🔍Test case 3🔍🔍🔍\n");
    SVD_Decomp_Case3();

    printf("\n\n🔍🔍🔍Test case 4🔍🔍🔍\n");
    SVD_Decomp_Case4();

    printf("\n\n🔍🔍🔍Test case 5🔍🔍🔍\n");
    SVD_Decomp_Case5();




} // end of SVD_Decomp_test



void setRank_Test()
{
    printf("\n\n~~setRank_Test()~~\n\n");

    printf("\n\n🔍🔍🔍Test case 1🔍🔍🔍\n");
    setRank_Case1();

    printf("\n\n🔍🔍🔍Test case 2🔍🔍🔍\n");
    setRank_Case2();

    printf("\n\n🔍🔍🔍Test case 3🔍🔍🔍\n");
    setRank_Case3();

    printf("\n\n🔍🔍🔍Test case 4🔍🔍🔍\n");
    setRank_Case4();

    printf("\n\n🔍🔍🔍Test case 5🔍🔍🔍\n");
    setRank_Case5();


} // end of setRank_Test




void truncate_Den_Mtx_Test()
{   
    // printf("\n\n~~truncate_Den_Mtx_Test()~~\n\n");

    // printf("\n\n🔍🔍🔍Test case 1🔍🔍🔍\n");
    // truncate_Den_Mtx_Case1();

    // printf("\n\n🔍🔍🔍Test case 2🔍🔍🔍\n");
    // truncate_Den_Mtx_Case2();

    // printf("\n\n🔍🔍🔍Test case 3🔍🔍🔍\n");
    // truncate_Den_Mtx_Case3();

    // printf("\n\n🔍🔍🔍Test case 4🔍🔍🔍\n");
    // truncate_Den_Mtx_Case4();

    // printf("\n\n🔍🔍🔍Test case 5🔍🔍🔍\n");
    // truncate_Den_Mtx_Case5();

} // end of truncate_Den_Mtx_Test




void normalize_mtx_d_Test()
{
    printf("\n\n~~normalize_mtx_d_Test()~~\n\n");

    printf("\n\n🔍🔍🔍Test case 1🔍🔍🔍\n");
    normalize_Den_Mtx_Case1();

    // printf("\n\n🔍🔍🔍Test case 2🔍🔍🔍\n");
    // normalize_Den_Mtx_Case2();

    // printf("\n\n🔍🔍🔍Test case 3🔍🔍🔍\n");
    // normalize_Den_Mtx_Case3();

    // printf("\n\n🔍🔍🔍Test case 4🔍🔍🔍\n");
    // normalize_Den_Mtx_Case4();

    // printf("\n\n🔍🔍🔍Test case 5🔍🔍🔍\n");
    // normalize_Den_Mtx_Case5();


} // end of normalize_mtx_d_Test


void orth_test()
{
    printf("\n\n~~orth_Test()~~\n\n");

    // printf("\n\n🔍🔍🔍Test case 1🔍🔍🔍\n");
    // orth_test_Case1();
    
    printf("\n\n🔍🔍🔍Test case 2🔍🔍🔍\n");
    orth_test_Case2();

    // printf("\n\n🔍🔍🔍Test case 3🔍🔍🔍\n");
    // orth_test_Case3();

    // printf("\n\n🔍🔍🔍Test case 4🔍🔍🔍\n");
    // orth_test_Case4();

    // printf("\n\n🔍🔍🔍Test case 5🔍🔍🔍\n");
    // orth_test_Case5();
}