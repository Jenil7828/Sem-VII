// Write a program to implement Parallel Bubble Sort and Merge sort using OpenMP. 
// Use existing algorithms and measure the performance of sequential and parallel algorithms.
// - Use std::vector for dynamic arrays
// How to compile: g++ -fopenmp Practical2.cpp -o program
// How to run: g++ -fopenmp Practical2.cpp -o program

#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
using namespace std;

void SequentialBubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n-1; i++) {
        for (int j = 0; j < n-i-1; j++) {
            if(arr[j] > arr[j+1]) {
                swap(arr[j], arr[j+1]);
            }
        }
    }
}

void ParallelBubbleSort(vector<int>& arr) {
    int n = arr.size();
    for(int i = 0; i < n-1; i++) {
        #pragma omp parallel for
        for(int j = 0; j < n-1; j++) {
            if(arr[j] > arr[j+1]) {
                swap(arr[j], arr[j+1]);
            }
        }

        #pragma omp parallel for
        for(int j = 0; j < n-1; j+=2) {
            if(arr[j] > arr[j+1]) {
                swap(arr[j], arr[j+1]);
            }
        }
    }
}

void merge(vector<int>& arr, int left, int mid, int right) {
    vector<int> temp(right - left + 1);
    int l= left;
    int r = mid + 1;
    int t = 0;
    while(l <= mid && r <= right) {
        if(arr[l] < arr[r]) {
            temp[t++] = arr[l++];
        }
        else {
            temp[t++] = arr[r++];
        }
    }
    while (l <= mid) {
        temp[t++] = arr[l++];
    }
    while (r <= right) {
        temp[t++] = arr[r++];
    }
    for(int pos = left, k = 0; pos <= right; pos++, k++) {
        arr[pos] = temp[k];
    }
}

void ParallelMergeSort(vector<int>& arr, int left, int right) {
    if(left >= right) return;
    int mid = (right + left) / 2;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            ParallelMergeSort(arr, left, mid);
        }
        #pragma omp section
        {
            ParallelMergeSort(arr, mid+1, right);
        }
    }
    merge(arr, left, mid, right);
}

void SequentialMergeSort(vector<int>& arr, int left, int right) {
    if(left >= right) return;
    int mid = (left + right) / 2;
    SequentialMergeSort(arr, left, mid);
    SequentialMergeSort(arr, mid+1, right);
    merge(arr, left, mid, right);
}

void printArray(const vector<int>& arr) {
    for(int num: arr) {
        cout<<num<<" ";
    }
    cout<<endl;
}

int main() {
    int n;
    cout<<"Enter number of elements: ";
    cin>>n;
    vector<int> arr(n);
    for(int i = 0; i < n; i++) {
        cout<<"Enter element "<<i+1<<": ";
        cin>>arr[i];
    }
    vector<int> arr1 = arr, arr2 = arr, arr3 = arr, arr4 = arr;
    double start, end;

    start = omp_get_wtime();
    SequentialBubbleSort(arr1);
    end = omp_get_wtime();
    cout<<"Sequential Bubble Sort: ";
    printArray(arr1);
    cout<<"Time taken: "<<end - start<<" seconds"<<endl;

    start = omp_get_wtime();
    ParallelBubbleSort(arr2);
    end = omp_get_wtime();
    cout<<"Parallel Bubble Sort: ";
    printArray(arr2);
    cout<<"Time taken: "<<end - start<<" seconds"<<endl;

    start = omp_get_wtime();
    SequentialMergeSort(arr3, 0, n-1);
    end = omp_get_wtime();
    cout<<"Sequential Merge Sort: ";
    printArray(arr3);
    cout<<"Time taken: "<<end - start<<" seconds"<<endl;

    start = omp_get_wtime();
    ParallelMergeSort(arr4, 0, n-1);
    end = omp_get_wtime();
    cout<<"Parallel Merge Sort: ";
    printArray(arr4);
    cout<<"Time taken: "<<end - start<<" seconds"<<endl;

    return 0;
}

// Sample Input:
// Enter number of elements: 10
// Enter element 1: 5
// Enter element 2: 2
// Enter element 3: 9
// Enter element 4: 1
// Enter element 5: 5
// Enter element 6: 6
// Enter element 7: 7
// Enter element 8: 3
// Enter element 9: 4
// Enter element 10: 8
// Sample Output:
// Sequential Bubble Sort: 1 2 3 4 5 5 6 7 8 9
// Time taken: 0.0002 seconds
// Parallel Bubble Sort: 1 2 3 4 5 5 6 7 8 9
// Time taken: 0.0001 seconds
// Sequential Merge Sort: 1 2 3 4 5 5 6 7 8 9
// Time taken: 0.0001 seconds
// Parallel Merge Sort: 1 2 3 4 5 5 6 7 8 9
// Time taken: 0.00005 seconds
// Note: The actual times will vary based on the system and the random data generated.