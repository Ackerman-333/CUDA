#include<cuda_runtime.h>
__device__ float warp_reduce(float value)
{
    for(int offset=16;offset>0;offset/=2)
    {
        value+=__shfl_down_sync(0xffffffff,value,offset);
    }
    return value;
}
__global__ void reduce(const float* input,float *output,int N)
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
        if(laneId==0)atomicAdd(output,value);
    }
}
extern "C" void solve(const float* input,float *output,int N)
{
    int numblocks=(N+255)/256;
    numblocks=(numblocks>256)?256:numblocks;
    cudaMemset(output,0,sizeof(float));
    reduce<<<numblocks,256>>>(input,output,N);
    cudaDeviceSynchronize();
}