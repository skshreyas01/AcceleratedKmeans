#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "kMeansCuda.h"

namespace cuda 
{

void get_kernel_config_given_ratios(int size_1, int size_2, dim3& size_grid, dim3& size_block
                , int& row_per_thread, int& col_per_thread
                , int n_thread_x_ratio, int n_thread_y_ratio)
{
    size_block.x = std::min(size_1, n_thread_x_ratio);
    size_block.y = std::min(size_2, n_thread_y_ratio);
    size_block.z = 1;
    size_grid.x = size_grid.y = size_grid.z = 1;
    col_per_thread = row_per_thread = 1;
    
    if (size_1 > n_thread_x_ratio || size_2 > n_thread_y_ratio)
    {
        int ratio = size_1/n_thread_x_ratio, k;
        for (k = 1; (1 << k) <= ratio; ++k)
        {
            row_per_thread = (2 << (k/2));
        }
        size_grid.x = (size_1 + size_block.x*row_per_thread - 1) / (size_block.x*row_per_thread);

        ratio = size_2/n_thread_y_ratio;
        for (k = 1; (1 << k) <= ratio; ++k)
        {
            col_per_thread = (2 << (k/2));
        }
        size_grid.y = (size_2 + size_block.y*col_per_thread - 1) / (size_block.y*col_per_thread);
    }
    assert(size_grid.x*size_block.x*row_per_thread >= size_1);
    assert(size_grid.y*size_block.y*col_per_thread >= size_2);
}

void get_kernel_config(int size_1, int size_2, dim3& size_grid, dim3& size_block
                    , int& row_per_thread, int& col_per_thread)
{
    
    int n_thread_x, n_thread_y;
    if (size_1 / size_2 >= 2)
    {
        n_thread_x = 64; n_thread_y = 16;
    }
    else if (size_2 / size_1 >= 2)
    {
        n_thread_x = 16; n_thread_y = 64;
    }
    else
    {
        n_thread_x = n_thread_y = 32;
    }
    get_kernel_config_given_ratios(size_1, size_2, size_grid, size_block
            , row_per_thread, col_per_thread, n_thread_x, n_thread_y);
}

/******************************************************************************/


static inline int next_power(int n) {
    n--;

    n = n >>  1 | n;
    n = n >>  2 | n;
    n = n >>  4 | n;
    n = n >>  8 | n;
    n = n >> 16 | n;
//  n = n >> 32 | n;    //  For 64-bit ints

    return ++n;
}

/*Euclid Distance Calculation*/
__host__ __device__ inline static
float euclid_distance(int    num_coords,
                    int    num_of_objs,
                    int    num_of_clusters,
                    float *objects,    
                    float *clusters,   
                    int    objectId,
                    int    clusterId)
{
    int i;
    float ans=0.0;

    for (i = 0; i < num_coords; i++) {
        ans += (objects[num_of_objs * i + objectId] - clusters[num_of_clusters * i + clusterId]) *
               (objects[num_of_objs * i + objectId] - clusters[num_of_clusters * i + clusterId]);
    }

    return(ans);
}

/*Finding nearest cluster*/
__global__ static
void find_nearest_cluster(int num_coords,
                          int num_of_objs,
                          int num_of_clusters,
                          float *objects,           //  [num_coords][num_of_objs]
                          float *device_clusters,    //  [num_coords][num_of_clusters]
                          int *membership,          //  [num_of_objs]
                          int *intermediates)
{
    extern __shared__ char shared_memory[];

    unsigned char *membership_changed = (unsigned char *)shared_memory;
#if BLOCK_SHARED_MEM_OPTIMIZATION
    float *clusters = (float *)(shared_memory + blockDim.x);
#else
    float *clusters = device_clusters;
#endif

    membership_changed[threadIdx.x] = 0;

#if BLOCK_SHARED_MEM_OPTIMIZATION
  
    for (int i = threadIdx.x; i < num_of_clusters; i += blockDim.x) {
        for (int j = 0; j < num_coords; j++) {
            clusters[num_of_clusters * j + i] = device_clusters[num_of_clusters * j + i];
        }
    }
    __syncthreads();
#endif

    int objectId = blockDim.x * blockIdx.x + threadIdx.x;

    if (objectId < num_of_objs) {
        int   index, i;
        float dist, minimum_distance;

		/*finding id of cluster which has minimum distance to object*/
        index    = 0;
        minimum_distance = euclid_distance(num_coords, num_of_objs, num_of_clusters,
                                 objects, clusters, objectId, 0);

        for (i=1; i<num_of_clusters; i++) {
            dist = euclid_distance(num_coords, num_of_objs, num_of_clusters,
                                 objects, clusters, objectId, i);
            /* no need square root */
            if (dist < minimum_distance) { /* find the min and its array index */
                minimum_distance = dist;
                index    = i;
            }
        }

        if (membership[objectId] != index) {
            membership_changed[threadIdx.x] = 1;
        }
		
		/*assigning array membership to object id*/
        membership[objectId] = index;

        __syncthreads();    //  For membership_changed[]

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                membership_changed[threadIdx.x] +=
                    membership_changed[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            intermediates[blockIdx.x] = membership_changed[0];
        }
    }
}

__global__ static
void compute_delta(int *device_intermediate,
                   int num_Intermediate,    //  The actual number of intermediates
                   int num_Intermediate2)   //  The next power of two
{
   
    extern __shared__ unsigned int intermediates[];

    //  Copying into shared memory.
    intermediates[threadIdx.x] =
        (threadIdx.x < num_Intermediate) ? device_intermediate[threadIdx.x] : 0;

    __syncthreads();

    for (unsigned int s = num_Intermediate2 / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            intermediates[threadIdx.x] += intermediates[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        device_intermediate[0] = intermediates[0];
    }
}

#define malloc2D(name, xDim, yDim, type) do {               \
    name = (type **)malloc(xDim * sizeof(type *));          \
    assert(name != NULL);                                   \
    name[0] = (type *)malloc(xDim * yDim * sizeof(type));   \
    assert(name[0] != NULL);                                \
    for (size_t i = 1; i < xDim; i++)                       \
        name[i] = name[i-1] + yDim;                         \
} while (0)



/* returns a cluster array of centers  */
float** kMeansHost(float **objects,     
                   int     num_coords,   
                   int     num_of_objs,     
                   int     num_of_clusters, 
                   float   threshold,   
                   int    *membership,  
                   int    *loop_iterations)
{
    int      i, j, index, loop=0;
    int     *new_cluster_size; 
    float    delta;          
    float  **dim_Objects;
    float  **clusters;       
    float  **dim_clusters;
    float  **new_clusters;    

    float *device_Objects;
    float *device_clusters;
    int *device_membership;
    int *device_intermediate;

    
    malloc2D(dim_Objects, num_coords, num_of_objs, float);
    for (i = 0; i < num_coords; i++) {
        for (j = 0; j < num_of_objs; j++) {
            dim_Objects[i][j] = objects[j][i];
        }
    }

    /* picking as initial center clusters*/
    malloc2D(dim_clusters, num_coords, num_of_clusters, float);
    for (i = 0; i < num_coords; i++) {
        for (j = 0; j < num_of_clusters; j++) {
            dim_clusters[i][j] = dim_Objects[i][j];
        }
    }

    /* initializing array membership */
    for (i=0; i<num_of_objs; i++) membership[i] = -1;

   
    new_cluster_size = (int*) calloc(num_of_clusters, sizeof(int));
    assert(new_cluster_size != NULL);

    malloc2D(new_clusters, num_coords, num_of_clusters, float);
    memset(new_clusters[0], 0, num_coords * num_of_clusters * sizeof(float));

    
    const unsigned int num_Threads_Per_Cluster_Block = 128;
    const unsigned int num_Cluster_Blocks =
        (num_of_objs + num_Threads_Per_Cluster_Block - 1) / num_Threads_Per_Cluster_Block;
#if BLOCK_SHARED_MEM_OPTIMIZATION
    const unsigned int cluster_Block_Shared_DataSize =
        num_Threads_Per_Cluster_Block * sizeof(unsigned char) +
        num_of_clusters * num_coords * sizeof(float);

    cudadevice_Prop device_Prop;
    int device_Num;
    cudaGetDevice(&device_Num);
    cudaGetdevice_Properties(&device_Prop, device_Num);

    if (cluster_Block_Shared_DataSize > device_Prop.sharedMemPerBlock) {
        err("WARNING:CUDA hardware has insufficient block shared memory. "
            "You need to recompile with BLOCK_SHARED_MEM_OPTIMIZATION=0. "
            "\n");
    }
#else
    const unsigned int cluster_Block_Shared_DataSize =
        num_Threads_Per_Cluster_Block * sizeof(unsigned char);
#endif

    const unsigned int num_Reduction_Threads =
        next_power(num_Cluster_Blocks);
    const unsigned int reduction_Block_Shared_Data_Size =
        num_Reduction_Threads * sizeof(unsigned int);

    CHECK_CUDA(cudaMalloc(&device_Objects, num_of_objs*num_coords*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_clusters, num_of_clusters*num_coords*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_membership, num_of_objs*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&device_intermediate, num_Reduction_Threads*sizeof(unsigned int)));

    CHECK_CUDA(cudaMemcpy(device_Objects, dim_Objects[0],
              num_of_objs*num_coords*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_membership, membership,
              num_of_objs*sizeof(int), cudaMemcpyHostToDevice));

    do {
        CHECK_CUDA(cudaMemcpy(device_clusters, dim_clusters[0],
                  num_of_clusters*num_coords*sizeof(float), cudaMemcpyHostToDevice));

        find_nearest_cluster
            <<< num_Cluster_Blocks, num_Threads_Per_Cluster_Block, cluster_Block_Shared_DataSize >>>
            (num_coords, num_of_objs, num_of_clusters,
             device_Objects, device_clusters, device_membership, device_intermediate);

        cudaDeviceSynchronize();
        CHECK_CUDA(cudaGetLastError());

        compute_delta <<< 1, num_Reduction_Threads, reduction_Block_Shared_Data_Size >>>
            (device_intermediate, num_Cluster_Blocks, num_Reduction_Threads);

        cudaDeviceSynchronize();
        CHECK_CUDA(cudaGetLastError());

        int d;
        CHECK_CUDA(cudaMemcpy(&d, device_intermediate,
                  sizeof(int), cudaMemcpyDeviceToHost));
        delta = (float)d;

        CHECK_CUDA(cudaMemcpy(membership, device_membership,
                  num_of_objs*sizeof(int), cudaMemcpyDeviceToHost));

        for (i=0; i<num_of_objs; i++) {
           
            index = membership[i];

       
            new_cluster_size[index]++;
            for (j=0; j<num_coords; j++)
                new_clusters[j][index] += objects[i][j];
        }

        
        for (i=0; i<num_of_clusters; i++) {
            for (j=0; j<num_coords; j++) {
                if (new_cluster_size[i] > 0)
                    dim_clusters[j][i] = new_clusters[j][i] / new_cluster_size[i];
                new_clusters[j][i] = 0.0;   /* set back to 0 */
            }
            new_cluster_size[i] = 0;   /* set back to 0 */
        }

        delta /= num_of_objs;
    } while (delta > threshold && loop++ < 500);

    *loop_iterations = loop + 1;


    malloc2D(clusters, num_of_clusters, num_coords, float);
    for (i = 0; i < num_of_clusters; i++) {
        for (j = 0; j < num_coords; j++) {
            clusters[i][j] = dim_clusters[j][i];
        }
    }

    CHECK_CUDA(cudaFree(device_Objects));
    CHECK_CUDA(cudaFree(device_clusters));
    CHECK_CUDA(cudaFree(device_membership));
    CHECK_CUDA(cudaFree(device_intermediate));

    free(dim_Objects[0]);
    free(dim_Objects);
    free(dim_clusters[0]);
    free(dim_clusters);
    free(new_clusters[0]);
    free(new_clusters);
    free(new_cluster_size);

    return clusters;
}



__global__ static
void update_cluster(const float* objects, const int* membership, float* clusters
                    , const int nCoords, const int nObjs, const int nClusters
                    , const int row_per_thread, const int col_per_thread)
{
    for (int cIdx = 0; cIdx < col_per_thread; ++cIdx)
    {
        int c = cIdx * gridDim.y * blockDim.y + blockIdx.y * blockDim.y + threadIdx.y;
        if (c >= nClusters)
            break;
        
        for (int rIdx = 0; rIdx < row_per_thread; ++rIdx)
        {
            int r = rIdx * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
            if (r >= nCoords)
                break;

            float sumVal(0);
            int clusterCount(0);
            for (int i = 0; i < nObjs; ++i)
            {
                if (membership[i] == c)
                {
                    sumVal += objects[r*nObjs + i];
                    clusterCount++;
                }
            }
            if (clusterCount > 0)
                clusters[nClusters*r+c] = sumVal / clusterCount;
        }
    }
}

__global__ static
void copy_rows(const float* src, const int size_1, const int size_2
                , const int copied_Rows, float* dest
                , const int row_per_thread, const int col_per_thread)
{
    for (int rIdx = 0; rIdx < row_per_thread; ++rIdx)
    {
        int r = rIdx * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
        if (r >= copied_Rows)
            break;
            
        for (int cIdx = 0; cIdx < col_per_thread; ++cIdx)
        {
            int c = cIdx * gridDim.y * blockDim.y + blockIdx.y * blockDim.y + threadIdx.y;
            if (c >= size_2)
                break;
            dest[c*copied_Rows+r] = src[c*size_1+r];
        }
    }
}

int kMeans(float *device_Objects,      
                   int     num_coords,    
                   int     num_of_objs,      
                   int     num_of_clusters,  
                   float   threshold,    
                   int     maxLoop,      
                   int    *membership,   
                   float  *device_clusters)
{
    int loop(0);
    float    delta;          
    int *device_membership;
    int *device_intermediate;

    CHECK_PARAM(device_clusters, "device_clusters cannot be NULL");
    
    
    const unsigned int num_Threads_Per_Cluster_Block = 128;
    const unsigned int num_Cluster_Blocks =
        (num_of_objs + num_Threads_Per_Cluster_Block - 1) / num_Threads_Per_Cluster_Block;
#if BLOCK_SHARED_MEM_OPTIMIZATION
    const unsigned int cluster_Block_Shared_DataSize =
        num_Threads_Per_Cluster_Block * sizeof(unsigned char) +
        num_of_clusters * num_coords * sizeof(float);

    cudadevice_Prop device_Prop;
    int device_Num;
    cudaGetDevice(&device_Num);
    cudaGetdevice_Properties(&device_Prop, device_Num);

    if (cluster_Block_Shared_DataSize > device_Prop.sharedMemPerBlock) {
        err("WARNING: CUDA has insufficient block shared memory. "
            "You need to recompile with BLOCK_SHARED_MEM_OPTIMIZATION=0. "
            "\n");
    }
#else
    const unsigned int cluster_Block_Shared_DataSize =
        num_Threads_Per_Cluster_Block * sizeof(unsigned char);
#endif

    const unsigned int num_Reduction_Threads = next_power(num_Cluster_Blocks);
    const unsigned int reduction_Block_Shared_Data_Size = num_Reduction_Threads * sizeof(unsigned int);

    CHECK_CUDA(cudaMalloc(&device_membership, num_of_objs*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&device_intermediate, num_Reduction_Threads*sizeof(unsigned int)));

   
    if (membership)
    {
        for (int i=0; i<num_of_objs; i++) 
            membership[i] = -1;
        CHECK_CUDA(cudaMemcpy(device_membership, membership,
              num_of_objs*sizeof(int), cudaMemcpyHostToDevice));
    }
    else
    {
        int* host_Membership = (int*)malloc(num_of_objs*sizeof(int));
        CHECK_PARAM(host_Membership, "memory allocation failed");
        for (int i=0; i<num_of_objs; i++) 
            host_Membership[i] = -1;
        CHECK_CUDA(cudaMemcpy(device_membership, host_Membership,
              num_of_objs*sizeof(int), cudaMemcpyHostToDevice));
        free(host_Membership);
    }

    dim3 size_grid, size_block;
    int row_per_thread, col_per_thread;
        
    //initializing centroids
    get_kernel_config(num_of_clusters, num_coords, size_grid, size_block, row_per_thread, col_per_thread);
    copy_rows<<<size_grid, size_block>>>(device_Objects, num_of_objs, num_coords
            , num_of_clusters, device_clusters, row_per_thread, col_per_thread);
    
    do
    {
        find_nearest_cluster
            <<< num_Cluster_Blocks, num_Threads_Per_Cluster_Block, cluster_Block_Shared_DataSize >>>
            (num_coords, num_of_objs, num_of_clusters,
             device_Objects, device_clusters, device_membership, device_intermediate);

        

        compute_delta <<< 1, num_Reduction_Threads, reduction_Block_Shared_Data_Size >>>
            (device_intermediate, num_Cluster_Blocks, num_Reduction_Threads);

       

        get_kernel_config(num_coords, num_of_clusters, size_grid, size_block, row_per_thread, col_per_thread);
        
        update_cluster <<< size_grid, size_block >>> (device_Objects, device_membership
                    , device_clusters, num_coords, num_of_objs, num_of_clusters, row_per_thread, col_per_thread);
        
        cudaDeviceSynchronize();
        CHECK_CUDA(cudaGetLastError());
        
        
        int d;
        CHECK_CUDA(cudaMemcpy(&d, device_intermediate,
                  sizeof(int), cudaMemcpyDeviceToHost));
        delta = (float)d/num_of_objs;
    } 
    while (delta > threshold && loop++ < maxLoop);

    if (membership)
    {
        CHECK_CUDA(cudaMemcpy(membership, device_membership, 
              num_of_objs*sizeof(int), cudaMemcpyDeviceToHost));
    }
    CHECK_CUDA(cudaFree(device_membership));
    CHECK_CUDA(cudaFree(device_intermediate));

    return (loop + 1);
}

}