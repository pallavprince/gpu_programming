#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>

#define THREADS_PER_BLOCK	1024
#define DATA_SIZE		(500 * 1024 * 1024)

typedef float precision_t;

struct mem_pointers{
	int	Arr_dim;
	precision_t	*h_in;
	precision_t	*h_out;
	precision_t	*d_in;
	precision_t	*d_out;
	precision_t	*cpu_out;
};

__host__ void cpu_reduce(struct mem_pointers *mem)
{
	double sum = 0;
	int idx;

	for (idx = 0; idx < mem->Arr_dim; idx++)
		sum += mem->h_in[idx];
	*(mem->cpu_out) = sum;
	return;
}

#define gpuErrchk(ans)				\
{						\
	gpuAssert((ans), __FILE__, __LINE__);	\
}

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr,"GPUassert: %s %s %d\n",
			cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#if 0
__global__ void gpu_reduce_kernel_seq_2synct(precision_t *d_in,
					     precision_t *d_out, int Arr_dim)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	precision_t load;
	int id;

	for (id = Arr_dim / 2; id; id >>= 1) {
		if (idx < id) {
			load = d_in[idx];
			d_in[idx] = load + d_in[idx + id];
		}
		__syncthreads();
	}
	if (idx == 0)
		*d_out = d_in[idx];
}
#endif

#if 0
__global__ void gpu_reduce_kernel_seq_1synct(precision_t *d_in,
					     precision_t *d_out, int Arr_dim)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	precision_t load;
	int id;

	for (id = Arr_dim / 2; id; id >>= 1) {
		if (idx < id)
			d_in[idx] = d_in[idx] + d_in[idx + id];
		__syncthreads();
	}

	if (idx == 0)
		*d_out = d_in[idx];
}
#endif

__global__ void gpu_reduce_kernel_interleaved(precision_t *d_in,
					      precision_t *d_out,
					      int Arr_dim)
{
	extern __shared__ precision_t sh_in[];
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;
	int sid, indx;

	sh_in[tid] = d_in[idx];
	__syncthreads();
	for (sid = 1; sid < blockDim.x; sid <<= 1) {
		indx = 2 * sid * tid;
		if (indx < blockDim.x) {
			sh_in[indx] += sh_in[indx + sid];
		}
		__syncthreads();
	}
	if (tid == 0)
		atomicAdd(d_out, sh_in[0]);
}

__global__ void gpu_reduce_kernel_seqential(precision_t *d_in,
					    precision_t *d_out,
					    int Arr_dim)
{
	extern __shared__ precision_t sh_in[];
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;
	int sid;

	sh_in[tid] = d_in[idx];
	__syncthreads();
	for (sid = blockDim.x / 2; sid ; sid >>= 1) {
		if ( tid < sid)
			sh_in[tid] += sh_in[tid + sid];
		__syncthreads();
	}

	if (tid == 0)
		atomicAdd(d_out, sh_in[0]);
}

__global__ void gpu_reduce_kernel_reverse_seqential(precision_t *d_in,
						    precision_t *d_out,
						    int Arr_dim)
{
	extern __shared__ precision_t sh_in[];
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;
	int sid;

	sh_in[tid] = d_in[idx];
	__syncthreads();
	for (sid = blockDim.x / 2; sid; sid >>= 1) {
		if ( tid < sid)
			sh_in[tid] += sh_in[ 2 * sid - tid - 1];
		__syncthreads();
	}

	if (tid == 0)
		atomicAdd(d_out, sh_in[0]);
}

static double cpu_get_time(struct timespec *st, struct timespec *en)
{
	double st_nsec, en_nsec;

	st_nsec = st->tv_nsec + st->tv_sec * 1000000000ULL;
	en_nsec = en->tv_nsec + en->tv_sec * 1000000000ULL;
	return (en_nsec - st_nsec) / 1000;
}

static int alloc_host_buffers(struct mem_pointers *mem)
{
	mem->h_in = (precision_t *)calloc(mem->Arr_dim, sizeof(precision_t));
	if (!mem->h_in)
		return -ENOMEM;
	mem->h_out = (precision_t *)calloc(3, sizeof(precision_t));
	if (!mem->h_out)
		goto out1;
	mem->cpu_out = (precision_t *)calloc(1, sizeof(precision_t));
	if (!mem->cpu_out)
		goto out2;
	return 0;
out2:
	free(mem->h_out);
out1:
	free(mem->h_in);
	return -ENOMEM;
}

static void free_host_buffers(struct mem_pointers *mem)
{
	free(mem->cpu_out);
	free(mem->h_out);
	free(mem->h_in);
}

static int alloc_device_buffers(struct mem_pointers *mem, precision_t nblks)
{
	if (cudaMalloc(&mem->d_in,
		       (mem->Arr_dim * sizeof(precision_t))) != cudaSuccess)
		return -ENOMEM;
	if (cudaMalloc(&mem->d_out, 3 * (sizeof(precision_t))) != cudaSuccess)
		goto out1;
	return 0;
out1:
	cudaFree(mem->d_in);
	return -ENOMEM;
}

static void free_device_buffers(struct mem_pointers *mem)
{
	cudaFree(mem->d_out);
	cudaFree(mem->d_in);
}

static void init_host_input(struct mem_pointers *mem)
{
	int idx;

	for (idx = 0; idx < mem->Arr_dim; idx++)
		mem->h_in[idx] = 100 * drand48() / random(); 
}

int main(int argc, char *argv[])
{
	precision_t d_time1, d_time2, h_time;
	struct timespec h_start, h_end;
	cudaEvent_t d_start, d_end;
	struct mem_pointers mem;
	int ret, nth, nblks;
	int shm_sz;

	mem.Arr_dim = DATA_SIZE;
	nth = THREADS_PER_BLOCK;
	nblks = (mem.Arr_dim + nth - 1) / nth;
	shm_sz = nth * sizeof(precision_t);

	if (alloc_host_buffers(&mem))
		return -ENOMEM;
	ret = alloc_device_buffers(&mem, nblks);
	if (ret)
		goto out;
	init_host_input(&mem);

	clock_gettime(CLOCK_REALTIME, &h_start);
	cpu_reduce(&mem);
	clock_gettime(CLOCK_REALTIME, &h_end);

	ret = cudaMemset(mem.d_out, 0, sizeof(precision_t));
	if (ret != cudaSuccess) {
		ret = -EINVAL;
		goto out1;
	}

	cudaEventCreate(&d_start);
	cudaEventCreate(&d_end);
	cudaEventRecord(d_start);

	cudaMemcpy(mem.d_in, mem.h_in, mem.Arr_dim * sizeof(precision_t),
		   cudaMemcpyHostToDevice);

	printf("Calling cuda kernel, nblk = %d, nth = %d\n", nblks, nth);
	gpu_reduce_kernel_interleaved<<<nblks, nth, shm_sz>>>(mem.d_in,
							      &mem.d_out[0],
							      mem.Arr_dim);

	gpuErrchk(cudaPeekAtLastError());
	cudaMemcpy(&mem.h_out[0], &mem.d_out[0],
		   sizeof(precision_t), cudaMemcpyDeviceToHost);
	cudaEventRecord(d_end);
        cudaEventSynchronize(d_end);
        cudaEventElapsedTime((float *)&d_time1, d_start, d_end);
	d_time1 *= 1000;

	cudaEventRecord(d_start);
	gpu_reduce_kernel_seqential<<<nblks, nth, shm_sz>>>(mem.d_in,
							    &mem.d_out[1],
							    mem.Arr_dim);

	gpuErrchk(cudaPeekAtLastError());
	cudaMemcpy(&mem.h_out[1], &mem.d_out[1],
		   sizeof(precision_t), cudaMemcpyDeviceToHost);
	cudaEventRecord(d_end);
        cudaEventSynchronize(d_end);
        cudaEventElapsedTime((float *)&d_time2, d_start, d_end);
	d_time2 *= 1000;

	cudaEventRecord(d_start);
	gpu_reduce_kernel_reverse_seqential<<<nblks, nth, shm_sz>>>(mem.d_in,
							      &mem.d_out[2],
							      mem.Arr_dim);
	gpuErrchk(cudaPeekAtLastError());

	h_time = cpu_get_time(&h_start, &h_end);
#if 0
	if (*(mem.cpu_out) != *(mem.h_out)) {
		printf("FAIL at cpu: %f, gpu %f\n", *(mem.cpu_out), *(mem.h_out));
		goto out1;
	}
#endif
	printf("Input size %s: %d\n\n",
	       (sizeof(precision_t) == sizeof(float)) ? "floats" : "doubles",
	       mem.Arr_dim);

	printf("Results for interleaved memory access\n");
	printf("CPU sum: %lf; GPU sum: %lf\n", *(mem.cpu_out), mem.h_out[0]);
	printf("Cpu time: %fus; Gpu time: %fus\n", h_time, d_time1);
	printf("Speedup: Cpu/Gpu = %f\n\n", h_time / d_time1);

	printf("Results for sequential memory access\n");
	printf("CPU sum: %lf; GPU sum = %lf\n", *(mem.cpu_out), mem.h_out[1]);
	printf("cpu time: %fus; gpu time: %fus\n", h_time, d_time2);
	printf("Speedup: cpu/gpu = %f\n\n", h_time / d_time2);
	

out1:
	free_device_buffers(&mem);
out:
	free_host_buffers(&mem);
	return ret;
}
