#include<cuda_runtime.h>
__device__ float warp_reduce(float value)
{
    for(int offset=16;offset>0;offset/=2)
    {
        value+=__shfl_down_sync(0xffffffff,value,offset);
    }
    return value;
}
__global__ void reduce2(float *partial,float *output,int numblocks)
{
    __shared__ float sdata[8];
    int tid=threadIdx.x;
    float value=(tid<numblocks)?partial[tid]:0.0f;
    value=warp_reduce(value);
    int warpid=tid/32;
    int laneid=tid%32;
    if(laneid==0)
    {
        sdata[warpid]=value;
    }
    __syncthreads();
    if(warpid==0)
    {
        value=(tid<8)?sdata[tid]:0.0f;
        value=warp_reduce(value);
        if(tid==0)*output=value;
    }
}
__global__ void reduce(const float *input,float *partial,int N)
{
    __shared__ float sdata[8];
    int tid=threadIdx.x;
    int idx=blockDim.x*blockIdx.x+tid;
    float value=(idx<N)?input[idx]:0.0f;
    value=warp_reduce(value);
    int laneid=tid%32;
    int warpid=tid/32;
    if(laneid==0)
    {
        sdata[warpid]=value;
    }
    __syncthreads();
    if(warpid==0)
    {
        value=(laneid<8)?sdata[laneid]:0.0f;
        value=warp_reduce(value);
        if(laneid==0)partial[blockIdx.x]=value;
    }
}
extern "C" void solve(const float *input,float *output,int N)
{
    int numblocks=(N+255)/256;
    float * partial;
    cudaMalloc(&partial,sizeof(float)*numblocks);
    reduce<<<numblocks,256>>>(input,partial,N);
    reduce2<<<1,256>>>(partial,output,numblocks);
    cudaDeviceSynchronize();
    cudaFree(partial);
}