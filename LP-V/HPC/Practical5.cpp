// Practical 5: Parallel Matrix Multiplication using OpenMP
// g++ -fopenmp -O2 Practical5.cpp -o Practical5
// Practical5 

#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

// Print matrix
void printMatrix(vector<vector<int>>& matrix) {

    for (int row = 0; row < matrix.size(); row++) {

        for (int col = 0; col < matrix[0].size(); col++) {

            cout << matrix[row][col] << " ";
        }

        cout << endl;
    }
}

int main() {

    int rowsA, colsA;
    int rowsB, colsB;

    cout << "Enter rows and cols of Matrix A: ";
    cin >> rowsA >> colsA;

    cout << "Enter rows and cols of Matrix B: ";
    cin >> rowsB >> colsB;

    // Matrix multiplication condition
    if (colsA != rowsB) {

        cout << "Multiplication not possible\n";
        return 0;
    }

    // Number of threads
    int threads;

    cout << "Enter number of threads: ";
    cin >> threads;

    omp_set_num_threads(threads);

    // Create matrices
    vector<vector<int>> A(
        rowsA,
        vector<int>(colsA)
    );

    vector<vector<int>> B(
        rowsB,
        vector<int>(colsB)
    );

    vector<vector<int>> C(
        rowsA,
        vector<int>(colsB, 0)
    );

    // Input Matrix A
    cout << "\nEnter Matrix A:\n";

    for (int row = 0; row < rowsA; row++) {

        for (int col = 0; col < colsA; col++) {

            cin >> A[row][col];
        }
    }

    // Input Matrix B
    cout << "\nEnter Matrix B:\n";

    for (int row = 0; row < rowsB; row++) {

        for (int col = 0; col < colsB; col++) {

            cin >> B[row][col];
        }
    }

    double start = omp_get_wtime();

    // Parallel region
    #pragma omp parallel for

    for (int row = 0; row < rowsA; row++) {

        int threadID = omp_get_thread_num();

        #pragma omp critical
        {
        cout << "Thread "
             << threadID
             << " processing row "
             << row
             << endl;
        }

        for (int col = 0; col < colsB; col++) {

            for (int k = 0; k < colsA; k++) {
                #pragma omp critical
                cout<<"Matrix C["<<row<<"]["<<col<<"] += "<<A[row][k]<<" * "<<B[k][col]<<"\n";
                C[row][col] +=
                    A[row][k] * B[k][col];
            }
        }
    }

    double end = omp_get_wtime();

    // Output matrix
    cout << "\nResult Matrix:\n";

    printMatrix(C);

    cout << "\nExecution Time: "
         << end - start
         << " seconds\n";

    return 0;
}
