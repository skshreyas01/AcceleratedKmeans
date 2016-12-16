#ifndef KMEANSCUDA_H
#define	KMEANSCUDA_H

#include <sstream>
#include <stdexcept>
#include <cuda_runtime.h>

namespace cuda
{

inline void checkCudaError(cudaError_t err
                    , char const * file, unsigned int line)
{
    if (err != cudaSuccess)
    {
        std::stringstream ss;
        ss << "CUDA error " << err << " at " << file << ":" << line;
        throw std::runtime_error(ss.str());
    }
}

inline void check(bool bTrue, const char* msg
                     , char const * file, unsigned int line)
{
    if (!bTrue)
    {
        std::stringstream ss;
        ss << "Error: \"" << msg << "\" at " << file << ":" << line;
        throw std::runtime_error(ss.str());
    }
}

#define CHECK_PARAM(x, msg)   cuda::check((x), (msg), __FILE__, __LINE__)
#define CHECK_CUDA(cudaError) cuda::checkCudaError((cudaError), __FILE__, __LINE__)


int kMeans(float *deviceObjects,     
               int     numCoords,    
               int     numObjs,      
               int     numClusters,  
               float   threshold,    
               int     maxLoop,      
               int    *membership,   
               float  *deviceClusters);


float** kMeansHost(float **objects,      
                   int     numCoords,    
                   int     numObjs,      
                   int     numClusters,  
                   float   threshold,    
                   int    *membership,   
                   int    *loop_iterations);
}

#endif	

