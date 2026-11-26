#include<cuda_runtime.h>
__device__ float warp_reduce(float value)
{
    for(int offset=16;offset>0;offset/=2)
    {
        __shfl_down_sync(0xffffffff,value,offset);
    }
    return value;
}
__global__ void reduce2(float * partial,float *output,int numblocks)
{
    __shared__ float sdata[8];
    int tid=threadIdx.x;
    float value=(tid<numblocks)?partial[tid]:0.0f;
    value=warp_reduce(value);
    int laneId=tid%32;
    int warpId=tid/32;
    if(laneId==0)
    {
        sdata[warpId]=value;
    }
    __syncthreads();
    if(warpId==0)
    {
        value=(tid<8)?sdata[tid]:0.0f;
        value=warp_reduce(value);
        if(laneId==0)*output=value;
    }
}
__global__ void reduce(const float* input,float *partial,int N)
{
    __shared__ float sdata[8];
    int tid=threadIdx.x;
    int idx=blockIdx.x*blockDim.x+tid;
    int stride=blockDim.x*gridDim.x;
    float value=0.0f;
    for(int i=idx;i<N;i+=stride)
    {
        value+=input[i];
    }
    value=warp_reduce(value);
    int laneId=tid%32;
    int warpId=tid/32;
    if(laneId==0)
    {
        sdata[warpId]=value;
    }
    __syncthreads();
    if(warpId==0)
    {
        value=(laneId<8)?sdata[laneId]:0.0f;
        value=warp_reduce(value);
        if(laneId==0)partial[blockIdx.x]=value;
    }
}
extern "C" void solve(const float* input,float *output,int N)
{
    int numblocks=(N+255)/256;
    numblocks=(numblocks>256)?256:numblocks;
    float *partial;
    cudaMalloc(&partial,sizeof(float)*numblocks);
    reduce<<<numblocks,256>>>(input,partial,N);
    reduce2<<<1,256>>>(partial,output,numblocks);
    cudaDeviceSynchronize();
    cudaFree(partial);
}