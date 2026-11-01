#include "ops.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cub/device/device_scan.cuh>
#include <algorithm>
#include <climits>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string error_text;
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void blas_check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS error " + std::to_string(int(status)));
}
template<class T> struct Buffer {
    T* p = nullptr;
    explicit Buffer(size_t n) { if (n) check(cudaMalloc(reinterpret_cast<void**>(&p), n*sizeof(T))); }
    ~Buffer() { if (p) cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void put(const T* data, size_t n) { if (n) check(cudaMemcpy(p,data,n*sizeof(T),cudaMemcpyHostToDevice)); }
    void get(T* data, size_t n) const { if (n) check(cudaMemcpy(data,p,n*sizeof(T),cudaMemcpyDeviceToHost)); }
    void zero(size_t n) { if (n) check(cudaMemset(p,0,n*sizeof(T))); }
};
struct Blas {
    cublasHandle_t handle = nullptr;
    explicit Blas(bool needed) {
        if (needed) {
            blas_check(cublasCreate(&handle));
            auto status = cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
            if (status != CUBLAS_STATUS_SUCCESS) { cublasDestroy(handle); handle=nullptr; blas_check(status); }
        }
    }
    ~Blas() { if (handle) cublasDestroy(handle); }
};
template<class F> int protect(F f) {
    try { error_text.clear(); f(); check(cudaGetLastError()); check(cudaDeviceSynchronize()); return 0; }
    catch (const std::exception& e) { error_text=e.what(); return -1; }
}
__global__ void similarity_basic(const float* q,const float* db,float* s,int nq,int n,int d) {
    size_t p=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(p>=size_t(nq)*n) return;
    int i=int(p/n), j=int(p%n);
    float v=0;
    for(int a=0;a<d;++a) v+=q[size_t(i)*d+a]*db[size_t(j)*d+a];
    s[p]=v;
}

__global__ void similarity_tiled(const float* q,const float* db,float* s,int nq,int n,int d) {
    __shared__ float a[16][16],b[16][17];
    int x=threadIdx.x,y=threadIdx.y;
    int row=blockIdx.y*16+y,col=blockIdx.x*16+x;
    float value=0;
    for(int t=0;t<d;t+=16) {
        a[y][x]=(row<nq && t+x<d)?q[size_t(row)*d+t+x]:0;
        b[y][x]=(col<n && t+y<d)?db[size_t(col)*d+t+y]:0;
        __syncthreads();
        #pragma unroll
        for(int j=0;j<16;++j) value+=a[y][j]*b[j][x];
        __syncthreads();
    }
    if(row<nq && col<n) s[size_t(row)*n+col]=value;
}

void similarity(const float* q,const float* db,float* s,int nq,int n,int d,int mode,Blas& blas) {
    if(mode==2) {
        float alpha=1,beta=0;
        blas_check(cublasSgemm(blas.handle,CUBLAS_OP_T,CUBLAS_OP_N,n,nq,d,&alpha,db,d,q,d,&beta,s,n));
    } else if(mode==1) {
        similarity_tiled<<<dim3((n+15)/16,(nq+15)/16),dim3(16,16)>>>(q,db,s,nq,n,d);
    } else {
        similarity_basic<<<unsigned((size_t(nq)*n+255)/256),256>>>(q,db,s,nq,n,d);
    }
    check(cudaGetLastError());
}

__global__ void initialize_top(float* values,int* ids,size_t count) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count) { values[i]=-INFINITY; ids[i]=-1; }
}

__global__ void merge_top(float* s,float* previous,int* previous_ids,float* out,int* ids,
                           int n,int k,int offset) {
    __shared__ float values[128];
    __shared__ int identifiers[128],positions[128];
    int q=blockIdx.x,t=threadIdx.x;
    for(int rank=0;rank<k;++rank) {
        float best=-INFINITY; int id=INT_MAX,pos=-1;
        for(int j=t;j<n+k;j+=128) {
            float v=(j<n)?s[size_t(q)*n+j]:previous[size_t(q)*k+j-n];
            int candidate=(j<n)?offset+j:previous_ids[size_t(q)*k+j-n];
            if(candidate>=0 && isfinite(v) && (v>best || (v==best && candidate<id))) {
                best=v; id=candidate; pos=j;
            }
        }
        values[t]=best; identifiers[t]=id; positions[t]=pos;
        __syncthreads();
        for(int stride=64;stride;stride/=2) {
            if(t<stride && (values[t+stride]>values[t] ||
                (values[t+stride]==values[t] && identifiers[t+stride]<identifiers[t]))) {
                values[t]=values[t+stride]; identifiers[t]=identifiers[t+stride]; positions[t]=positions[t+stride];
            }
            __syncthreads();
        }
        if(!t) {
            out[size_t(q)*k+rank]=values[0];
            ids[size_t(q)*k+rank]=(positions[0]<0)?-1:identifiers[0];
            int p=positions[0];
            if(p>=0 && p<n) s[size_t(q)*n+p]=-INFINITY;
            else if(p>=n) previous[size_t(q)*k+p-n]=-INFINITY;
        }
        __syncthreads();
    }
}

__device__ int gray_at(const unsigned char* image,int w,int y,int x) {
    const auto* p=image+(size_t(y)*w+x)*3;
    return (77*int(p[0])+150*int(p[1])+29*int(p[2]))>>8;
}

__global__ void descriptors_basic(const unsigned char* images,float* out,int n,int h,int w,int bins) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    int hist[128]={0};
    const auto* image=images+size_t(i)*h*w*3;
    for(int p=0;p<h*w;++p) for(int c=0;c<3;++c) ++hist[c*bins+int(image[p*3+c])*bins/256];
    for(int y=1;y<h-1;++y) for(int x=1;x<w-1;++x) {
        int gx=-gray_at(image,w,y-1,x-1)+gray_at(image,w,y-1,x+1)-2*gray_at(image,w,y,x-1)+2*gray_at(image,w,y,x+1)-gray_at(image,w,y+1,x-1)+gray_at(image,w,y+1,x+1);
        int gy=-gray_at(image,w,y-1,x-1)-2*gray_at(image,w,y-1,x)-gray_at(image,w,y-1,x+1)+gray_at(image,w,y+1,x-1)+2*gray_at(image,w,y+1,x)+gray_at(image,w,y+1,x+1);
        ++hist[3*bins+min(bins-1,(abs(gx)+abs(gy))*bins/2041)];
    }
    for(int j=0;j<4*bins;++j) out[size_t(i)*4*bins+j]=float(hist[j]);
}

__global__ void descriptors_tiled(const unsigned char* images,float* out,int h,int w,int bins) {
    __shared__ int gray[18][18],hist[128];
    int image_id=blockIdx.z,tx=threadIdx.x,ty=threadIdx.y,t=ty*16+tx;
    int ox=blockIdx.x*16,oy=blockIdx.y*16;
    const auto* image=images+size_t(image_id)*h*w*3;
    for(int j=t;j<4*bins;j+=256) hist[j]=0;
    for(int j=t;j<18*18;j+=256) {
        int y=oy+j/18-1,x=ox+j%18-1;
        gray[j/18][j%18]=(y>=0 && y<h && x>=0 && x<w)?gray_at(image,w,y,x):0;
    }
    __syncthreads();
    int x=ox+tx,y=oy+ty;
    if(x<w && y<h) {
        for(int c=0;c<3;++c) atomicAdd(&hist[c*bins+int(image[(size_t(y)*w+x)*3+c])*bins/256],1);
        if(x>0 && x<w-1 && y>0 && y<h-1) {
            int a=ty+1,b=tx+1;
            int gx=-gray[a-1][b-1]+gray[a-1][b+1]-2*gray[a][b-1]+2*gray[a][b+1]-gray[a+1][b-1]+gray[a+1][b+1];
            int gy=-gray[a-1][b-1]-2*gray[a-1][b]-gray[a-1][b+1]+gray[a+1][b-1]+2*gray[a+1][b]+gray[a+1][b+1];
            atomicAdd(&hist[3*bins+min(bins-1,(abs(gx)+abs(gy))*bins/2041)],1);
        }
    }
    __syncthreads();
    for(int j=t;j<4*bins;j+=256) atomicAdd(out+size_t(image_id)*4*bins+j,float(hist[j]));
}

__global__ void normalize_descriptors(float* out,int n,int h,int w,int bins) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    float norm=0;
    for(int j=0;j<4*bins;++j) {
        float v=out[size_t(i)*4*bins+j]/float(j<3*bins?h*w:(h-2)*(w-2));
        out[size_t(i)*4*bins+j]=v; norm+=v*v;
    }
    float inv=rsqrtf(norm);
    for(int j=0;j<4*bins;++j) out[size_t(i)*4*bins+j]*=inv;
}

__global__ void count_labels(const int* labels,int* counts,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) atomicAdd(counts+labels[i],1);
}

__global__ void scatter_members(const int* labels,int* cursors,int* permutation,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) permutation[atomicAdd(cursors+labels[i],1)]=i;
}
}

extern "C" const char* hpc_cuda_error() {return error_text.c_str();}

extern "C" int hpc_cuda_devices() {int count=0;auto s=cudaGetDeviceCount(&count);if(s!=cudaSuccess) {error_text=cudaGetErrorString(s);return 0;}return count;}

extern "C" int hpc_cuda_set_device(int device) {return protect([&]{check(cudaSetDevice(device));});}

extern "C" int hpc_cuda_search(const float* q,const float* db,int nq,int n,int d,int k,int chunk,int mode,int* ids,float* scores) {
    return protect([&]{
        if(nq<1 || n<1 || d<1 || k<1 || k>n || chunk<1 || mode<0 || mode>2) throw std::invalid_argument("Invalid CUDA search arguments");
        int capacity=std::min(n,chunk);
        Buffer<float> dq(size_t(nq)*d),ddb(size_t(capacity)*d),s(size_t(nq)*capacity),a(size_t(nq)*k),b(size_t(nq)*k);
        Buffer<int> ia(size_t(nq)*k),ib(size_t(nq)*k);
        dq.put(q,size_t(nq)*d);
        initialize_top<<<unsigned((size_t(nq)*k+255)/256),256>>>(a.p,ia.p,size_t(nq)*k);
        float *old=a.p,*next=b.p;int *old_ids=ia.p,*next_ids=ib.p;
        Blas blas(mode==2);
        for(int start=0;start<n;start+=capacity) {
            int count=std::min(capacity,n-start);
            ddb.put(db+size_t(start)*d,size_t(count)*d);
            similarity(dq.p,ddb.p,s.p,nq,count,d,mode,blas);
            merge_top<<<nq,128>>>(s.p,old,old_ids,next,next_ids,count,k,start);
            check(cudaGetLastError());std::swap(old,next);std::swap(old_ids,next_ids);
        }
        check(cudaMemcpy(scores,old,size_t(nq)*k*sizeof(float),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(ids,old_ids,size_t(nq)*k*sizeof(int),cudaMemcpyDeviceToHost));
    });
}

extern "C" int hpc_cuda_descriptors(const unsigned char* images,int n,int h,int w,int bins,int mode,float* output) {
    return protect([&]{
        if(n<1 || h<3 || w<3 || bins<2 || bins>32 || mode<0 || mode>2) throw std::invalid_argument("Invalid CUDA descriptor arguments");
        Buffer<unsigned char> data(size_t(n)*h*w*3);Buffer<float> out(size_t(n)*4*bins);
        data.put(images,size_t(n)*h*w*3);out.zero(size_t(n)*4*bins);
        if(mode) descriptors_tiled<<<dim3((w+15)/16,(h+15)/16,n),dim3(16,16)>>>(data.p,out.p,h,w,bins);
        else descriptors_basic<<<(n+63)/64,64>>>(data.p,out.p,n,h,w,bins);
        normalize_descriptors<<<(n+127)/128,128>>>(out.p,n,h,w,bins);
        out.get(output,size_t(n)*4*bins);
    });
}

extern "C" int hpc_cuda_inverted_index(const int* labels,int n,int clusters,int* offsets,int* permutation) {
    return protect([&]{
        if(n<0 || clusters<1 || clusters==INT_MAX) throw std::invalid_argument("Invalid CUDA index shape");
        for(int i=0;i<n;++i) if(labels[i]<0 || labels[i]>=clusters) throw std::invalid_argument("Invalid cluster label");
        Buffer<int> data(n),counts(size_t(clusters)+1),prefix(size_t(clusters)+1),cursors(clusters),order(n);
        data.put(labels,n);counts.zero(size_t(clusters)+1);
        if(n) count_labels<<<(n+255)/256,256>>>(data.p,counts.p,n);
        size_t scratch_bytes=0;
        check(cub::DeviceScan::ExclusiveSum(nullptr,scratch_bytes,counts.p,prefix.p,clusters+1));
        Buffer<unsigned char> scratch(scratch_bytes);
        check(cub::DeviceScan::ExclusiveSum(scratch.p,scratch_bytes,counts.p,prefix.p,clusters+1));
        check(cudaMemcpy(cursors.p,prefix.p,size_t(clusters)*sizeof(int),cudaMemcpyDeviceToDevice));
        if(n) scatter_members<<<(n+255)/256,256>>>(data.p,cursors.p,order.p,n);
        prefix.get(offsets,size_t(clusters)+1);order.get(permutation,n);
    });
}
