// Implement Min, Max, Sum and Average operations using Parallel Reduction.
// 

#include <iostream>
#include <omp.h>
using namespace std;

#define MAX 100000

// ================= MAIN =================
int main() {
    int n;
    cout << "Enter number of elements: ";
    cin >> n;

    int arr[MAX];

    // Input array
    cout << "Enter elements:\n";
    for (int i = 0; i < n; i++)
        cin >> arr[i];

    // ================= SEQUENTIAL =================
    int min_val = arr[0], max_val = arr[0];
    long long sum = 0;

    double start = omp_get_wtime();

    for (int i = 0; i < n; i++) {
        if (arr[i] < min_val)
            min_val = arr[i];

        if (arr[i] > max_val)
            max_val = arr[i];

        sum += arr[i];
    }

    double end = omp_get_wtime();

    cout << "\n--- Sequential Results ---\n";
    cout << "Min: " << min_val;
    cout << "\nMax: " << max_val;
    cout << "\nSum: " << sum;
    cout << "\nAverage: " << (double)sum / n;
    cout << "\nTime: " << (end - start) << " sec\n";

    // ================= PARALLEL =================
    int p_min = arr[0], p_max = arr[0];
    long long p_sum = 0;

    start = omp_get_wtime();

    #pragma omp parallel for reduction(+:p_sum) reduction(min:p_min) reduction(max:p_max)
    for (int i = 0; i < n; i++) {
        p_sum += arr[i];

        if (arr[i] < p_min)
            p_min = arr[i];

        if (arr[i] > p_max)
            p_max = arr[i];
    }

    end = omp_get_wtime();

    cout << "\n--- Parallel Results ---\n";
    cout << "Min: " << p_min;
    cout << "\nMax: " << p_max;
    cout << "\nSum: " << p_sum;
    cout << "\nAverage: " << (double)p_sum / n;
    cout << "\nTime: " << (end - start) << " sec\n";

    return 0;
}