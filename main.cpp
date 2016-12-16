/*
The main file of the program. Contains methods to create row major, column major data along with kmeans
*/

#include <cstdlib>
#include <cmath>
#include <assert.h>
#include <iostream>
#include <ctime>
#include "kMeansCuda.h"


float* create_Col_Major_Data(int size_1, int size_2, bool cudaMalloc)
{
   
    float* arr;
    if (cudaMalloc)
        CHECK_CUDA(cudaMallocHost(&arr, size_1*size_2*sizeof(float), cudaHostAllocDefault));
    else
        arr = (float*)malloc(size_1*size_2*sizeof(float));
    for (int i = 0; i < size_1; ++i)
        for (int j = 0; j < size_2; ++j)
        {
            arr[size_1*j + i] = i*100 + j;
        }
    return arr;
}

float** create_Row_Major_Data(int size_1, int size_2)
{
    float** ret = (float**)malloc(size_1*sizeof(float*));
    ret[0] = (float*)malloc(size_1*size_2*sizeof(float));
    for (int i = 1; i < size_1; ++i)
    {
        ret[i] = ret[i-1] + size_2;
    }
    for (int i = 0; i < size_1; ++i)
        for (int j = 0; j < size_2; ++j)
        {
            ret[i][j] = i*100 + j;
        }
    return ret;
}

float* copy_to_host_clusters(float* host_data, int num_of_objs, int num_of_dim, int num_of_clusters, int*& membership)
{
    float* devData, *dev_clusters, *host_Clusters;
    CHECK_CUDA(cudaMalloc(&devData, num_of_objs*num_of_dim*sizeof(float)));
    CHECK_CUDA(cudaMemcpy(devData, host_data, num_of_objs*num_of_dim*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&dev_clusters, num_of_clusters*num_of_dim*sizeof(float)));
    if (membership)
        membership = new int[num_of_objs];
    
    cuda::kMeans(devData, num_of_dim, num_of_objs, num_of_clusters, 0, 500, membership, dev_clusters);
    host_Clusters = new float[num_of_clusters*num_of_dim*sizeof(float)];
    
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(host_Clusters, dev_clusters, num_of_clusters*num_of_dim*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(devData));
    CHECK_CUDA(cudaFree(dev_clusters));
    
    return host_Clusters;
}

float** copy_from_host(float** host_data, int num_of_objs, int num_of_dim, int num_of_clusters, int*& membership)
{
    int loops;
    membership = new int[num_of_objs];
    return cuda::kMeansHost(host_data, num_of_dim, num_of_objs, num_of_clusters, 0, membership, &loops);
}


void k_means()
{
    const int size_1 = 5000, size_2 = 5000, num_of_clusters = 10;
	std::cout<<"\t\tData set size:"<<size_1<<std::endl;
	std::cout<<"\t\tNumber of clusters:"<<num_of_clusters<<std::endl;
    float* dataCm = create_Col_Major_Data(size_1, size_2, true);
    float** dataRm = create_Row_Major_Data(size_1, size_2);
    int* membership1, *membership2;
    float *clusters1, **clusters2;
    const int TIMES = 2;
    
    {
        clock_t begin = clock();
        for (int i = 0; i < TIMES; ++i)
            clusters1 = copy_to_host_clusters(dataCm, size_1, size_2, num_of_clusters, membership1);
        double elapsed_secs = double(clock() - begin) / CLOCKS_PER_SEC;
		
		
    }
    
    {
        clock_t begin = clock();
        for (int i = 0; i < TIMES; ++i)
            clusters2 = copy_from_host(dataRm, size_1, size_2, num_of_clusters, membership2);
        double elapsed_secs = double(clock() - begin) / CLOCKS_PER_SEC;
		
        std::cout << "\t\tElapsed Time: " << elapsed_secs << " secs" << std::endl;
    }
    
    delete[] membership1;
    delete[] membership2;
    delete[] clusters1;
    free(clusters2[0]);
    free(clusters2);
    CHECK_CUDA(cudaFreeHost(dataCm));
    free(dataRm[0]);
    free(dataRm);
}

int main(int argc, char** argv)
{
   
	printf("\t\tInitializing K-means\n");
    k_means();
	std::cout<<"\t\tPress any key to exit";
	getchar();
    return 0;
}

