// Design and implement Parallel Breadth First Search and Depth First Search based on existing
// algorithms using OpenMP. Use a Tree or an undirected graph for BFS and DFS .

// How to run: g++ -fopenmp Practical1.cpp -o program
#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>

using namespace std;

class Graph {
    int V;
    vector<vector<int>> adj;

public:
    Graph(int V) {
        this->V = V;
        adj.resize(V);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u); // undirected graph
    }

    // -------- PARALLEL BFS --------
    void parallelBFS(int start) {
        vector<bool> visited(V, false);
        queue<int> q;

        visited[start] = true;
        q.push(start);

        cout << "Parallel BFS: ";

        while (!q.empty()) {
            int size = q.size();

            #pragma omp parallel for
            for (int i = 0; i < size; i++) {
                int node = -1;

                #pragma omp critical
                {
                    if (!q.empty()) {
                        node = q.front();
                        q.pop();
                        cout << node << " ";
                    }
                }

                if (node == -1) continue;

                for (int neighbor : adj[node]) {
                    if (!visited[neighbor]) {
                        #pragma omp critical
                        {
                            if (!visited[neighbor]) {
                                visited[neighbor] = true;
                                q.push(neighbor);
                            }
                        }
                    }
                }
            }
        }
        cout << endl;
    }

    // -------- PARALLEL DFS --------
    void parallelDFSUtil(int node, vector<bool>& visited) {
        visited[node] = true;

        #pragma omp critical
        cout << node << " ";

        #pragma omp parallel for
        for (int i = 0; i < adj[node].size(); i++) {
            int neighbor = adj[node][i];

            if (!visited[neighbor]) {
                parallelDFSUtil(neighbor, visited);
            }
        }
    }

    void parallelDFS(int start) {
        vector<bool> visited(V, false);

        cout << "Parallel DFS: ";
        parallelDFSUtil(start, visited);
        cout << endl;
    }
};

int main() {
    int V, E;

    cout << "Enter number of vertices: ";
    cin >> V;

    Graph g(V);

    cout << "Enter number of edges: ";
    cin >> E;

    cout << "Enter edges (u v):\n";
    for (int i = 0; i < E; i++) {
        int u, v;
        cin >> u >> v;
        g.addEdge(u, v);
    }

    int start;
    cout << "Enter starting vertex: ";
    cin >> start;

    g.parallelBFS(start);
    g.parallelDFS(start);

    return 0;
}
//g++ -fopenmp file.cpp -o graph
//./graph

// Sample Input:
// Enter number of nodes: 5
// Enter number of edges: 6
// Enter edges (u v) for undirected graph:
// 0 1
// 0 2
// 1 3
// 1 4
// 2 4
// 3 4
// Enter starting node: 0
