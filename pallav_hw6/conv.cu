#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>

#define RANDOM_GEN		15
#define THREADS_PER_BLOCK	256
#define DATA_SIZE		(32768 * 1024)
#define FILTER_SIZE		17

typedef float precision_t;

struct mem_pointers{
	int	dim_a1;
	int	dim_a2;
	precision_t	*m1;
	precision_t	*n1;
	
	precision_t	*m2;
	precision_t	*n2;

	precision_t	*m3;
	precision_t	*n3;
	precision_t	*Cm;
};

__host__ void cpu_conv(struct mem_pointers *mem)
{
	int rb = (mem->dim_a2 - 1) / 2; /* dim_y should be odd and non-zero */
	int lb = -rb, idx, idy;

	for (idx = rb; idx < mem->dim_a1 - rb; ++idx)
		for (idy = lb; idy <= rb; ++idy)
			mem->Cm[idx] += mem->m1[idx + idy] * mem->m2[idy + rb];
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

__global__ void gpu_conv_shared_kernel(precision_t *n1, precision_t *n2, precision_t *n3,
				       int dim_a1, int dim_a2)
{
	__shared__ precision_t sa[THREADS_PER_BLOCK + FILTER_SIZE];
	__shared__ precision_t sb[FILTER_SIZE];

	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int radius = (dim_a2 - 1) / 2;
	int lidx, indx;
	precision_t sum;

	lidx = threadIdx.x + radius;
	if (threadIdx.x < dim_a2)
		sb[threadIdx.x] = n2[threadIdx.x];

	if (idx < dim_a1){
		sa[lidx] = n1[idx];
		if (threadIdx.x < radius)
		{
			if (idx >= radius)
				sa[threadIdx.x] = n1[idx - radius];
			if ((idx + THREADS_PER_BLOCK) < dim_a1)
				sa[THREADS_PER_BLOCK + lidx] = n1[idx + THREADS_PER_BLOCK];
		}
	}
	__syncthreads();

	if ((idx < (dim_a1 - radius)) && (idx >= radius))
	{
		sum = 0;
		for (indx = -radius; indx <= radius; indx++)
			sum += sa[lidx + indx] * sb[indx + radius];
		n3[idx] = sum;
	}
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
	mem->m1 = (precision_t *)calloc(mem->dim_a1, sizeof(precision_t));
	if (!mem->m1)
		return -ENOMEM;
	mem->m2 = (precision_t *)calloc(mem->dim_a2, sizeof(precision_t));
	if (!mem->m2)
		goto out1;
	mem->m3 = (precision_t *)calloc(mem->dim_a1, sizeof(precision_t));
	if (!mem->m3)
		goto out2;
	mem->Cm = (precision_t *)calloc(mem->dim_a1, sizeof(precision_t));
	if (!mem->Cm)
		goto out3;
	return 0;
out3:
	free(mem->m3);
out2:
	free(mem->m2);
out1:
	free(mem->m1);
	return -ENOMEM;
}

static void free_host_buffers(struct mem_pointers *mem)
{
	free(mem->Cm);
	free(mem->m3);
	free(mem->m2);
	free(mem->m1);
}

static int alloc_device_buffers(struct mem_pointers *mem)
{
	if (cudaMalloc(&mem->n1, (mem->dim_a1 * sizeof(precision_t))) != cudaSuccess)
		return -ENOMEM;
	if (cudaMalloc(&mem->n2, (mem->dim_a2 * sizeof(precision_t))) != cudaSuccess)
		goto out1;
	if (cudaMalloc(&mem->n3, (mem->dim_a1 * sizeof(precision_t))) != cudaSuccess)
		goto out2;
	return 0;
out2:
	cudaFree(mem->n2);
out1:
	cudaFree(mem->n1);
	return -ENOMEM;
}

static void free_device_buffers(struct mem_pointers *mem)
{
	cudaFree(mem->n3);
	cudaFree(mem->n2);
	cudaFree(mem->n1);
}

static void init_host_input(struct mem_pointers *mem)
{
	int idx;

	for (idx = 0; idx < mem->dim_a1; idx++)
		mem->m1[idx] = rand() % RANDOM_GEN;
	for (idx = 0; idx < mem->dim_a2; idx++)
		mem->m2[idx] = 1;
}

int main(int argc, char *argv[])
{
	struct timespec h_start, h_end;
	cudaEvent_t d_start, d_end;
	int ret, idx, nth, nblks;
	struct mem_pointers mem;
	precision_t d_time, h_time;

	mem.dim_a1 = DATA_SIZE;
	mem.dim_a2 = FILTER_SIZE;

	if (alloc_host_buffers(&mem))
		return -ENOMEM;
	ret = alloc_device_buffers(&mem);
	if (ret)
		goto out;
	init_host_input(&mem);

	clock_gettime(CLOCK_REALTIME, &h_start);
	cpu_conv(&mem);
	clock_gettime(CLOCK_REALTIME, &h_end);

	ret = cudaMemset(mem.n3, 0, mem.dim_a1 * sizeof(precision_t));
	if (ret != cudaSuccess) {
		ret = -EINVAL;
		goto out1;
	}

	cudaEventCreate(&d_start);
	cudaEventCreate(&d_end);

	nth = THREADS_PER_BLOCK;
	nblks = (mem.dim_a1 + nth - 1) / nth;
	cudaEventRecord(d_start);
	cudaMemcpy(mem.n1, mem.m1, mem.dim_a1 * sizeof(precision_t),
		   cudaMemcpyHostToDevice);
	cudaMemcpy(mem.n2, mem.m2, mem.dim_a2 * sizeof(precision_t),
		   cudaMemcpyHostToDevice);
	gpu_conv_shared_kernel<<<nblks,nth>>>(mem.n1, mem.n2,
					      mem.n3, mem.dim_a1, mem.dim_a2);
	gpuErrchk(cudaPeekAtLastError());
	cudaMemcpy(mem.m3, mem.n3,
		   mem.dim_a1 * sizeof(precision_t), cudaMemcpyDeviceToHost);
	cudaEventRecord(d_end);
        cudaEventSynchronize(d_end);
        cudaEventElapsedTime((float *)&d_time, d_start, d_end);
	d_time *= 1000;
	h_time = cpu_get_time(&h_start, &h_end);

	for (idx = 0; idx < mem.dim_a1; idx++){
		if (mem.cpu_c[idx] != mem.m3[idx]) {
			printf("FAIL at %d, cpu: %f, gpu %f\n",
				idx, mem.cpu_c[idx], mem.m3[idx]);
			goto out1;
		}
	}
	printf("Input size = %d single precision data:\ncpu time: %fus, gpu time: %fus\n",
		mem.dim_a1, h_time, d_time);
	printf("Speedup: cpu/gpu = %f\n", h_time / d_time);
out1:
	free_device_buffers(&mem);
out:
	free_host_buffers(&mem);
	return ret;
}
