#include <limits.h> 
#include <stdio.h> 
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
// #include <mpi.h>

#define seed 13517006

  
// A utility function to find the vertex with minimum distance value, from 
// the set of vertices not yet included in shortest path tree 
__device__
int minDistance(long *dist, bool *sptSet, int V, int src) {
    long min = LONG_MAX;
    int  min_index; 
  
    for (int v = 0; v < V; v++) {
        if (sptSet[src*V + v] == false && dist[src*V + v] <= min) {
            min = dist[src*V + v], min_index = v; 
        }
    }   
  
    return min_index; 
}

// Function that implements Dijkstra's single source shortest path algorithm 
// for a graph represented using adjacency matrix representation 
__device__
void dijkstra(int V, long *graph, long *(*dist), int src, bool* sptSet) { 
    for (int i = 0; i < V; i++) {
        (*dist)[src*V +i] = LONG_MAX;
        sptSet[src*V + i] = false;
    }

    (*dist)[src*V + src] = 0; 
    for (int count = 0; count < V - 1; count++) {
        int u = minDistance(*dist, sptSet, V, src); 

        sptSet[src*V + u] = true; 

        for (int v = 0; v < V; v++) {
            if (!sptSet[src*V + v] && graph[u*V + v] && (*dist)[src*V + u] != INT_MAX 
                && (*dist)[src*V + u] + graph[u*V + v] < (*dist)[src*V + v]) {
                (*dist)[src*V + v] = (*dist)[src*V + u] + graph[u*V + v];
            }
        }
    } 
} 

// Utility function to print a graph
__host__
void printGraph(long **graph, int V) {
    for (int i = 0; i < V; i++) {
        for (int j = 0; j< V ; j++) { 
            printf("%ld ", graph[i][j]);
        }
        printf("\n");
    }
}
  
// writeFile make a txt file to save result of dijkstra
__host__
void writeFile(long *graph, long time, int V) {
    FILE *fp;
    char filename[V/10 +6];
    sprintf(filename, "%d.txt", V);

    if((fp = fopen(filename, "wb")) == NULL) {
        printf("Failed to open file. \n");
    }


    for(int i = 0; i < V; i++){
        for(int j = 0; j < V; j++) {
            fprintf(fp, "%ld ", graph[i* V + j]);
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "Elapsed time : %ld micro-seconds", time);
    fprintf(fp, "\n");
    fclose(fp);
}

__global__
void cudaDijkstra(int V, long* dist, long* graph, bool* sptSet) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < V; i += stride)
        dijkstra(V, graph, &(dist), i, sptSet);     
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: number of node\n");
        exit(1);
    }

    int V = atoi(argv[1]);
    
    double time_spent = 0.0;

    long *graph;
    cudaMallocManaged(&graph, V*V*sizeof(long));
    
    long *dist;
    cudaMallocManaged(&dist, V*V*sizeof(long));

    bool *sptSet;
    cudaMallocManaged(&sptSet, V*V*sizeof(bool));

    srand(seed);
    for(int i = 0; i < V; i++) {
        for(int j = 0; j < V ; j++) { 
            if(i != j) {   
                graph[i*V+j] = rand();
            }       
            else {
                graph[i*V+j] = 0;
            }
        }
    }
    
    
    int blockSize = 256;
    int numBlocks = (V + blockSize - 1) / blockSize;
    clock_t begin = clock();
    cudaDijkstra<<<numBlocks, blockSize>>>(V, dist, graph, sptSet);
    cudaDeviceSynchronize();
    clock_t end = clock();

    time_spent += ((double)(end - begin) / CLOCKS_PER_SEC );
    time_spent *= 1000000;
    printf("Time elapsed is %f micro-seconds\n", time_spent);
    
    writeFile(dist, time_spent, V);
    cudaFree(dist);
    cudaFree(graph);
    return 0;
}