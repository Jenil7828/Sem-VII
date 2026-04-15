#include <stdio.h>
#include <cuda_runtime.h>

#define N 2

// ================= VECTOR ADD =================
__global__ void vectorAdd(int *A, int *B, int *C, int n) {
    int i = threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

// ================= MATRIX MULT =================
__global__ void matrixMul(int *A, int *B, int *C) {
    int row = threadIdx.y;
    int col = threadIdx.x;

    int sum = 0;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}

// ================= ERROR CHECK =================
void checkError() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("\nCUDA ERROR: %s\n", cudaGetErrorString(err));
    }
}

int main() {

    // -------- VECTOR ADD --------
    int n = 5;
    int size = n * sizeof(int);

    int A[5] = {1,2,3,4,5};
    int B[5] = {5,4,3,2,1};
    int C[5] = {0};   // initialize

    int *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    vectorAdd<<<1,n>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();
    checkError();   // 🔥 important

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Vector Addition Result:\n");
    for(int i=0;i<n;i++) printf("%d ", C[i]);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);


    // -------- MATRIX MULT --------
    int A_mat[N*N] = {1,2,3,4};
    int B_mat[N*N] = {1,0,0,1};
    int C_mat[N*N] = {0};   // initialize

    int *d_A_mat, *d_B_mat, *d_C_mat;
    int msize = N*N*sizeof(int);

    cudaMalloc((void**)&d_A_mat, msize);
    cudaMalloc((void**)&d_B_mat, msize);
    cudaMalloc((void**)&d_C_mat, msize);

    cudaMemcpy(d_A_mat, A_mat, msize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_mat, B_mat, msize, cudaMemcpyHostToDevice);

    dim3 threads(N,N);
    matrixMul<<<1,threads>>>(d_A_mat, d_B_mat, d_C_mat);
    cudaDeviceSynchronize();
    checkError();   // 🔥 important

    cudaMemcpy(C_mat, d_C_mat, msize, cudaMemcpyDeviceToHost);

    printf("\n\nMatrix Multiplication Result:\n");
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            printf("%d ", C_mat[i*N + j]);
        }
        printf("\n");
    }

    cudaFree(d_A_mat); cudaFree(d_B_mat); cudaFree(d_C_mat);

    return 0;
}