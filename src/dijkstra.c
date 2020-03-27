#include <limits.h> 
#include <stdio.h> 
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define seed 13517006

// Number of vertices in the graph 
int  V;
  
// A utility function to find the vertex with minimum distance value, from 
// the set of vertices not yet included in shortest path tree 
int minDistance(long *dist, bool *sptSet) {
    long min = LONG_MAX;
    int  min_index; 
  
    for (int v = 0; v < V; v++) {
        if (sptSet[v] == false && dist[v] <= min) {
            min = dist[v], min_index = v; 
        }
    }   
  
    return min_index; 
}
  
// Function that implements Dijkstra's single source shortest path algorithm 
// for a graph represented using adjacency matrix representation 
void dijkstra(long **graph, long *(*dist), int src) { 
    bool sptSet[V];

    for (int i = 0; i < V; i++) {
        (*dist)[i] = LONG_MAX;
        sptSet[i] = false;
    }

    (*dist)[src] = 0; 
    for (int count = 0; count < V - 1; count++) {
        int u = minDistance(*dist, sptSet); 

        sptSet[u] = true; 

        for (int v = 0; v < V; v++) {
            if (!sptSet[v] && graph[u][v] && (*dist)[u] != INT_MAX 
                && (*dist)[u] + graph[u][v] < (*dist)[v]) {
                (*dist)[v] = (*dist)[u] + graph[u][v];
            }
        }
    } 
} 

// Utility function to print a graph
void printGraph(long **graph) {
    for (int i = 0; i < V; i++) {
        for (int j = 0; j< V ; j++) { 
            printf("%ld ", graph[i][j]);
        }
        printf("\n");
    }
}
  
// writeFile make a txt file to save result of dijkstra
void writeFile(long **graph, long time) {
    FILE *fp;
    char filename[V/10 +6];
    sprintf(filename, "%d.txt", V);

    if((fp = fopen(filename, "wb")) == NULL) {
        printf("Failed to open file. \n");
    }
    
    for(int i = 0; i < V; i++){
        for(int j = 0; j < V; j++) {
            fprintf(fp, "%ld ", graph[i][j]);
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "Elapsed time : %ld micro-seconds", time);
    fprintf(fp, "\n");
    fclose(fp);
}

// Utility function to concat 2 array
void concat(long *(*a), int idxA, long *b, int lenB) {
    for(int i = 0; i < lenB; i++) {
        (*a)[i+idxA] = b[i];
    }
}

// Utility function to free a matrix
void freeMatrix(long ** mat) {
    for(long i = 0 ; i < V; i++) {
        free(mat[i]);
    }
    free(mat);
}

// sortSolution sort array received from MPI_Gather 
void sortSolution(long **(*dist), long *arr) {
    for(int i = V; i < (V+1)*V; i += V+1) {
        for(int j = 0; j < V; j++) {
            (*dist)[arr[i]][j] = arr[i-V+j];
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: number of node\n");
        exit(1);
    }

    V = atoi(argv[1]);
    
    double time_spent = 0.0;

    long **graph = (long**)malloc(sizeof(long*)*V);
    for(long i = 0; i < V; i++) {
        *(graph+i) = (long*)malloc(sizeof(long)*V);
    }
    
    long **dist = NULL;
    long *recvSolution = NULL;
    int *recvCount = NULL;
    int *displs = NULL;

    srand(seed);
    for(int i = 0; i < V; i++) {
        for(int j = 0; j < V ; j++) { 
            if(i != j) {   
                graph[i][j] = rand();
            }       
            else {
                graph[i][j] = 0;
            }
        }
    }
    
    MPI_Init(NULL, NULL);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    clock_t begin = clock();

    if(world_rank == 0) {
        dist = (long**)malloc(sizeof(long*)*V);
        for(long i =0; i < V; i++) {
            *(dist+i) = (long*)malloc(sizeof(long)*V);
        }

        recvSolution = (long*)malloc(sizeof(long*)*(V+1)*V);

        recvCount = malloc(sizeof(int)*world_size);
        displs = malloc(sizeof(int)*world_size);
        
        displs[0] = 0;
        for (int i = 0; i < world_size; i++) {
            recvCount[i] = V/world_size;
            if (V % world_size > i) {
                recvCount[i] += 1;    
            }
            recvCount[i] *= (V+1);

            if(i != world_size - 1) {
                displs[i+1] = displs[i] + recvCount[i];
            }
        }
    }
    
    int size_proc = V/world_size;
    if (V % world_size > world_rank) {
        size_proc += 1;    
    }

    long *sendSolution = (long*)malloc(sizeof(long)*(V + 1)*size_proc);

    for(int i = world_rank; i < V; i += world_size) {
        long *solutionPerSource = (long*)malloc(sizeof(long)*(V+1));
        
        dijkstra(graph, &solutionPerSource, i);
        solutionPerSource[V] = i;
        concat(&sendSolution, (i/world_size)*(V+1), solutionPerSource, V+1);
        
        free(solutionPerSource);
    }
       
    MPI_Gatherv(sendSolution, (V+1)*size_proc, MPI_LONG, recvSolution, recvCount, displs,
        MPI_LONG, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        sortSolution(&dist, recvSolution);
        
        clock_t end = clock();
        time_spent += ((double)(end - begin) / CLOCKS_PER_SEC );
        time_spent *= 1000000;
        printf("Time elapsed is %f micro-seconds\n", time_spent);
        
        writeFile(dist, time_spent);
        
        free(recvSolution);
        free(recvCount);
        free(displs);
        freeMatrix(dist);
    }
    
    free(sendSolution);
    freeMatrix(graph);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
