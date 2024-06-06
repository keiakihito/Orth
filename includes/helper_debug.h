#ifndef HELPER_DEBUG_H
#define HELPER_DEBUG_H

#include <iostream>
#include <cuda_runtime.h>

//Print vector loat
void print_vector(const float *d_val, int size) {
    // Allocate memory on the host
    float *check_r = (float *)malloc(sizeof(float) * size);

    if (check_r == NULL) {
        printf("Failed to allocate host memory");
        return;
    }

    // Copy data from device to host
    cudaError_t err = cudaMemcpy(check_r, d_val, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        free(check_r);
        return;
    }
    // Print the values to check them
    for (int i = 0; i < size; i++) {
            printf("%f \n", check_r[i]);
    }
    

    // Free allocated memory
    free(check_r);
} // print_vector


//Print vector integer
void print_vector(const int *d_val, int size) {
    // Allocate memory on the host
    int *check_r = (int*)malloc(sizeof(int) * size);

    if (check_r == NULL) {
        printf("Failed to allocate host memory");
        return;
    }

    // Copy data from device to host
    cudaError_t err = cudaMemcpy(check_r, d_val, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        free(check_r);
        return;
    }
    // Print the values to check them
    for (int i = 0; i < size; i++) {
            printf("%d \n", check_r[i]);
    }


    // Free allocated memory
    free(check_r);
}// end of print_vector



//Print matrix row major
void print_mtx_row_d(const float *mtx_d, int numOfRow, int numOfClm){
    //Allocate memory oh the host
    float *check_r = (float *)malloc(sizeof(float) * numOfRow * numOfClm);

    if (check_r == NULL) {
        printf("Failed to allocate host memory");
        return;
    }

    // Copy data from device to host
    cudaError_t err = cudaMemcpy(check_r, mtx_d, numOfRow * numOfClm * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        free(check_r);
        return;
    }

    for (int rwWkr = 0; rwWkr < numOfRow; rwWkr++){
        for(int clWkr = 0; clWkr < numOfClm; clWkr++){
            printf("%f ", check_r[rwWkr*numOfClm + clWkr]);
        }// end of column walker
        printf("\n");
    }// end of row walker
} // end of print_mtx_h


//Print matrix column major
void print_mtx_clm_d(const float *mtx_d, int numOfRow, int numOfClm){
    //Allocate memory oh the host
    float *check_r = (float *)malloc(sizeof(float) * numOfRow * numOfClm);

    if (check_r == NULL) {
        printf("Failed to allocate host memory");
        return;
    }

    // Copy data from device to host
    cudaError_t err = cudaMemcpy(check_r, mtx_d, numOfRow * numOfClm * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        free(check_r);
        return;
    }

    for (int rwWkr = 0; rwWkr < numOfRow; rwWkr++){
        for(int clWkr = 0; clWkr < numOfClm; clWkr++){
            printf("%f ", check_r[clWkr*numOfRow + rwWkr]);
        }// end of column walker
        printf("\n");
    }// end of row walker
} // end of print_mtx_h


//Print matrix row major
void print_mtx_row_h(const float *mtx_h, int numOfRow, int numOfClm){
    for (int rwWkr = 0; rwWkr < numOfRow; rwWkr++){
        for(int clWkr = 0; clWkr < numOfClm; clWkr++){
            printf("%f ", mtx_h[rwWkr*numOfClm + clWkr]);
        }// end of column walker
        printf("\n");
    }// end of row walker
} // end of print_mtx_h


//Print matrix column major
void print_mtx_clm_h(const float *mtx_h, int numOfRow, int numOfClm){
    for (int rwWkr = 0; rwWkr < numOfRow; rwWkr++){
        for(int clWkr = 0; clWkr < numOfClm; clWkr++){
            printf("%f ", mtx_h[clWkr*numOfRow + rwWkr]);
        }// end of column walker
        printf("\n");
    }// end of row walker
} // end of print_mtx_h


void validate(const float *mtxA_h, const float* x_h, float* rhs, int N){
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

}// end of validate

#endif // HELPER_DEBUG_H
