#include "ops.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#ifdef HPC_OPENMP
#include <omp.h>
#endif

namespace {
thread_local std::string last_error;
int workers(int requested, bool parallel) {
#ifdef HPC_OPENMP
    omp_set_dynamic(0);
    return parallel ? std::max(1, requested) : 1;
#else
    if (parallel) throw std::runtime_error("OpenMP was not compiled into this library");
    return 1;
#endif
}
float dot(const float* a, const float* b, int d) {
    float sum = 0;
#pragma omp simd reduction(+:sum)
    for (int j = 0; j < d; ++j) sum += a[j] * b[j];
    return sum;
}
template<class F> int protect(F operation) {
    try { last_error.clear(); operation(); return 0; }
    catch (const std::exception& e) { last_error = e.what(); return -1; }
}
}


extern "C" const char* hpc_error() { return last_error.c_str(); }

extern "C" int hpc_has_openmp() {
#ifdef HPC_OPENMP
    return 1;
#else
    return 0;
#endif
}

extern "C" int hpc_search(const float* q, const float* db, int nq, int n, int dim,
                          int k, int chunk, int threads, int parallel, int* ids, float* scores) {
    return protect([&] {
        if (nq < 1 || n < 1 || dim < 1 || k < 1 || k > n || chunk < 1 || threads < 1)
            throw std::invalid_argument("Invalid search shape, k, chunk, or threads");
        int nt = workers(threads, parallel);
        // pair(score, -id): smaller IDs win equal-score ties; min-heap exposes worst match.
        using Match = std::pair<float, int>;
#pragma omp parallel for num_threads(nt) if(parallel) schedule(static)
        for (int i = 0; i < nq; ++i) {
            std::priority_queue<Match, std::vector<Match>, std::greater<Match>> heap;
            for (int start = 0; start < n; start += chunk) {
                int end = std::min(n, start + chunk);
                for (int j = start; j < end; ++j) {
                    Match item{dot(q+size_t(i)*dim, db+size_t(j)*dim, dim), -j};
                    if (int(heap.size()) < k) heap.push(item);
                    else if (item > heap.top()) { heap.pop(); heap.push(item); }
                }
            }
            for (int r = k-1; r >= 0; --r) {
                ids[size_t(i)*k+r] = -heap.top().second;
                scores[size_t(i)*k+r] = heap.top().first;
                heap.pop();
            }
        }
    });
}

extern "C" int hpc_descriptors(const unsigned char* images, int n, int h, int w,
                               int bins, int threads, int parallel, float* out) {
    return protect([&] {
        if (n < 1 || h < 3 || w < 3 || bins < 2 || bins > 32 || threads < 1)
            throw std::invalid_argument("Images must be RGB, at least 3x3, with 2..32 histogram bins");
        int nt = workers(threads, parallel);
#pragma omp parallel for num_threads(nt) if(parallel) schedule(static)
        for (int i = 0; i < n; ++i) {
            const auto* image = images + size_t(i)*h*w*3;
            std::vector<int> hist(4*bins, 0);
            auto gray = [&](int y, int x) {
                const auto* p = image + (size_t(y)*w+x)*3;
                return (77*int(p[0]) + 150*int(p[1]) + 29*int(p[2])) >> 8;
            };
            for (int p = 0; p < h*w; ++p)
                for (int c = 0; c < 3; ++c) ++hist[c*bins + int(image[p*3+c])*bins/256];
            for (int y = 1; y < h-1; ++y) for (int x = 1; x < w-1; ++x) {
                int gx = -gray(y-1,x-1)+gray(y-1,x+1)-2*gray(y,x-1)+2*gray(y,x+1)-gray(y+1,x-1)+gray(y+1,x+1);
                int gy = -gray(y-1,x-1)-2*gray(y-1,x)-gray(y-1,x+1)+gray(y+1,x-1)+2*gray(y+1,x)+gray(y+1,x+1);
                int bin = std::min(bins-1, (std::abs(gx)+std::abs(gy))*bins/2041);
                ++hist[3*bins+bin];
            }
            float norm = 0;
            for (int j = 0; j < 4*bins; ++j) {
                float value = float(hist[j]) / (j < 3*bins ? h*w : (h-2)*(w-2));
                out[size_t(i)*4*bins+j] = value;
                norm += value*value;
            }
            norm = 1 / std::sqrt(norm);
            for (int j = 0; j < 4*bins; ++j) out[size_t(i)*4*bins+j] *= norm;
        }
    });
}

extern "C" int hpc_inverted_index(const int* labels, int n, int clusters, int* offsets, int* permutation) {
    return protect([&] {
        if (n < 0 || clusters < 1) throw std::invalid_argument("Invalid index size");
        std::fill(offsets, offsets+clusters+1, 0);
        for (int i = 0; i < n; ++i) {
            if (labels[i] < 0 || labels[i] >= clusters) throw std::invalid_argument("Invalid cluster label");
            ++offsets[labels[i]+1];
        }
        for (int c = 0; c < clusters; ++c) offsets[c+1] += offsets[c]; // Exclusive prefix scan of counts.
        std::vector<int> cursor(offsets, offsets+clusters);
        for (int i = 0; i < n; ++i) permutation[cursor[labels[i]]++] = i;
    });
}
