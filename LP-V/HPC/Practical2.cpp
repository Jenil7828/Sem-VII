// Write a program to implement Parallel Bubble Sort and Merge sort using OpenMP. 
// Use existing algorithms and measure the performance of sequential and parallel algorithms.

// How to run: g++ -fopenmp Practical2.cpp -o program

#include <iostream>
#include <omp.h>
#include <cstdlib>
using namespace std;

#define MAX 100000

// ================= UTILITY =================
void generateArray(int arr[], int n) {
    for (int i = 0; i < n; i++)
        arr[i] = rand() % 1000;
}

void copyArray(int src[], int dest[], int n) {
    for (int i = 0; i < n; i++)
        dest[i] = src[i];
}

// ================= SEQUENTIAL BUBBLE SORT =================
void bubbleSort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// ================= PARALLEL BUBBLE SORT =================
// Odd-Even Sort (Parallel version of Bubble Sort)
void parallelBubbleSort(int arr[], int n) {
    for (int i = 0; i < n; i++) {

        // Even phase
        #pragma omp parallel for
        for (int j = 0; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
        }

        // Odd phase
        #pragma omp parallel for
        for (int j = 1; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
        }
    }
}

// ================= MERGE FUNCTION =================
void merge(int arr[], int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    int L[MAX], R[MAX];

    for (int i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j])
            arr[k++] = L[i++];
        else
            arr[k++] = R[j++];
    }

    while (i < n1)
        arr[k++] = L[i++];
    while (j < n2)
        arr[k++] = R[j++];
}

// ================= SEQUENTIAL MERGE SORT =================
void mergeSort(int arr[], int l, int r) {
    if (l < r) {
        int m = (l + r) / 2;
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

// ================= PARALLEL MERGE SORT =================
void parallelMergeSort(int arr[], int l, int r) {
    if (l < r) {
        int m = (l + r) / 2;

        #pragma omp task
        parallelMergeSort(arr, l, m);

        #pragma omp task
        parallelMergeSort(arr, m + 1, r);

        #pragma omp taskwait
        merge(arr, l, m, r);
    }
}

// ================= MAIN =================
int main() {
    int n;
    cout << "Enter number of elements: ";
    cin >> n;

    int arr[MAX], arr1[MAX], arr2[MAX], arr3[MAX], arr4[MAX];

    generateArray(arr, n);

    copyArray(arr, arr1, n);
    copyArray(arr, arr2, n);
    copyArray(arr, arr3, n);
    copyArray(arr, arr4, n);

    double start, end;

    // Sequential Bubble Sort
    start = omp_get_wtime();
    bubbleSort(arr1, n);
    end = omp_get_wtime();
    cout << "\nSequential Bubble Sort Time: " << (end - start);

    // Parallel Bubble Sort
    start = omp_get_wtime();
    parallelBubbleSort(arr2, n);
    end = omp_get_wtime();
    cout << "\nParallel Bubble Sort Time: " << (end - start);

    // Sequential Merge Sort
    start = omp_get_wtime();
    mergeSort(arr3, 0, n - 1);
    end = omp_get_wtime();
    cout << "\nSequential Merge Sort Time: " << (end - start);

    // Parallel Merge Sort
    start = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        parallelMergeSort(arr4, 0, n - 1);
    }
    end = omp_get_wtime();
    cout << "\nParallel Merge Sort Time: " << (end - start);

    cout << endl;
    return 0;
}

// Sample Input:
// Enter number of elements: 1000
// Sample Output:
// Sequential Bubble Sort Time: 0.123456
// Parallel Bubble Sort Time: 0.012345
// Sequential Merge Sort Time: 0.012345
// Parallel Merge Sort Time: 0.001234
// Note: The actual times will vary based on the system and the random data generated.