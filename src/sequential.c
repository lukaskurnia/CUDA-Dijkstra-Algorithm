// A C++ program for Dijkstra's single source shortest path algorithm. 
// The program is for adjacency matrix representation of the graph 
  
#include <limits.h> 
#include <stdio.h> 
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>

#define seed 13517006

// Number of vertices in the graph 
int  V;

  
// A utility function to find the vertex with minimum distance value, from 
// the set of vertices not yet included in shortest path tree 
int minDistance(long **dist, bool *sptSet, int src) {
    // Initialize min value 
    long min = LONG_MAX;
    int  min_index; 
  
    for (int v = 0; v < V; v++) {
        if (sptSet[v] == false && dist[src][v] <= min) {
            min = dist[src][v], min_index = v; 
        }
    }   
  
    return min_index; 
}
  
// Function that implements Dijkstra's single source shortest path algorithm 
// for a graph represented using adjacency matrix representation 
void dijkstra(long **graph, long **(*dist)) { 
    
    for (int src = 0; src < V; src++) {
        bool sptSet[V]; // sptSet[i] will be true if vertex i is included in shortest 
        // path tree or shortest distance from src to i is finalized 

        // Initialize all distances as INFINITE and stpSet[] as false 
        for (int i = 0; i < V; i++) {
            (*dist)[src][i] = LONG_MAX;
            sptSet[i] = false;
        }
            
        // Distance of source vertex from itself is always 0 
        (*dist)[src][src] = 0; 
    
        // Find shortest path for all vertices 
        for (int count = 0; count < V - 1; count++) {
            // Pick the minimum distance vertex from the set of vertices not 
            // yet processed. u is always equal to src in the first iteration. 
            int u = minDistance(*dist, sptSet, src); 
    
            // Mark the picked vertex as processed 
            sptSet[u] = true; 
    
            // Update dist value of the adjacent vertices of the picked vertex. 
            for (int v = 0; v < V; v++) {
                // Update dist[v] only if is not in sptSet, there is an edge from 
                // u to v, and total weight of path from src to v through u is 
                // smaller than current value of dist[v] 
                if (!sptSet[v] && graph[u][v] && (*dist)[src][u] != INT_MAX 
                    && (*dist)[src][u] + graph[u][v] < (*dist)[src][v]) {
                    (*dist)[src][v] = (*dist)[src][u] + graph[u][v];
                }
            }
        } 
    }
} 

void printGraph(long **graph) {
    for (int i = 0; i < V; i++) {
        for (int j = 0; j< V ; j++) { 
            printf("%ld ", graph[i][j]);
        }
        printf("\n");
    }
}
  

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

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: number of node\n");
        exit(1);
    }

    V = atoi(argv[1]);

    double time_spent = 0.0;

    long ** graph = (long**)malloc(sizeof(long*)*V);
    for(long i =0 ; i <V;i++)
        *(graph+i) = (long*)malloc(sizeof(long)*V);

    long ** dist = (long**)malloc(sizeof(long*)*V);
    for(long i =0 ; i <V;i++)
        *(dist+i) = (long*)malloc(sizeof(long)*V);

    srand(seed);
    for(int i = 0; i < V; i++) {
        for(int j = 0; j< V ; j++) { 
            if(i != j) {   
                graph[i][j] = rand();
            }       
            else {
                graph[i][j] = 0;
            }
        }
    }
  
    clock_t begin = clock();
    dijkstra(graph, &dist);
    clock_t end = clock();
    
    time_spent += ((double)(end - begin) / CLOCKS_PER_SEC );
    time_spent *= 1000000;
    printf("Time elapsed is %f micro-seconds\n", time_spent);
    writeFile(dist, time_spent);
  
    return 0;
}
