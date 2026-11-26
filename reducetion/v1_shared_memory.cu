#include<cuda_runtime.h>
__global__ void reduce2(float *partial,float *output,int numblocks)
{
    __shared__ float sdata[256];
    int tid=threadIdx.x;
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<numblocks)
        sdata[tid]=partial[i];
    else
        sdata[tid]=0;
    __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1)
    {
        if(tid<s)
            sdata[tid]+=sdata[tid+s];
        __syncthreads();
    }
    if(tid==0)
        *output=sdata[0];
}
__global__ void reduce(const float *input,float *partial,int n)
{
    __shared__ float sdata[256];
    int tid=threadIdx.x;
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)
        sdata[tid]=input[i];
    else
        sdata[tid]=0;
    __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1)
    {
        if(tid<s)
            sdata[tid]+=sdata[tid+s];
        __syncthreads();
    }
    if(tid==0)
        partial[blockIdx.x]=sdata[0];
}
extern "C" void solve(const float* input,float *output,int n)
{
    int numblocks=(n+255)/256;
    float *partial;
    cudaMalloc(&partial, numblocks * sizeof(float));
    reduce<<<numblocks,256>>>(input,partial,n);
    reduce2<<<1,256>>>(partial,output,numblocks);
    cudaDeviceSynchronize();
    cudaFree(partial);
}