#include <stdio.h>
#include <stdlib.h>

#define  my_assert(expression)	\
	if (expression)		\
		asm("trap;")

#define gpuErrchk(ans)					\
	{						\
		gpuAssert((ans), __FILE__, __LINE__);	\
	}

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void my_divergent_kernel(int *x, int *y, int *z)
{
	my_assert((*x == *y || *x == *z));
	if (*x == *y)
		*z = *x;
	else
		*z = *y;
}

__global__ void my_non_divergent_kernel(int *x, int *y, int *z)
{
	int predicate;

	my_assert((*x == *y || *x == *z));
	predicate = (*x == *y);
	if (predicate)
		*z = *x;
	if (!predicate)
		*z = *y;
}

static int process_args(int argc, char *argv[])
{
	int nth;

	if (argc != 2) {
		printf("%s <number of threads>\n", argv[0]);
		return 0;
	}
	nth = atoi(argv[1]);
	if (nth > 1024) {
		printf("More than 1024 threads are not supported\n");
		return 0;
	}
	return nth;
}

int main(int argc, char *argv[])
{
	int hx = 3, hy = 4, hz = 5;
	cudaEvent_t start,stop;
	int *dx, *dy, *dz, nth;
	dim3 grid(1,1,1);
	int ox, oy, oz;
	float ms;

	nth = process_args(argc, argv);
	if (!nth)
		exit(-1);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMalloc(&dx, sizeof(int));
	cudaMalloc(&dy, sizeof(int));
	cudaMalloc(&dz, sizeof(int));

	printf("Original values: x = %d, y = %d, z = %d\n", hx, hy, hz);
	cudaMemcpy(dx, &hx, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dy, &hy, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dz, &hz, sizeof(int), cudaMemcpyHostToDevice);

	cudaEventRecord(start);
	my_divergent_kernel<<<grid, nth>>>(dx, dy, dz);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms,start,stop);
	gpuErrchk(cudaPeekAtLastError());

	/* cudaDeviceSynchronize(); not needed due to implicit barrier */
	cudaMemcpy(&ox, dx, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&oy, dy, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&oz, dz, sizeof(int), cudaMemcpyDeviceToHost);
	printf("Exchanged values using divergent kernel: x = %d, y = %d, z = %d\n",
		ox, oy, oz);
	printf("divergent kernel runtime = %f msec\n\n", ms);


	printf("Original values: x = %d, y = %d, z = %d\n", hx, hy, hz);
	cudaMemcpy(dx, &hx, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dy, &hy, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dz, &hz, sizeof(int), cudaMemcpyHostToDevice);

	cudaEventRecord(start);
	my_non_divergent_kernel<<<grid, nth>>>(dx, dy, dz);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms,start,stop);
	gpuErrchk(cudaPeekAtLastError());

	/*cudaDeviceSynchronize(); not needed due to implicit barrier */
	cudaMemcpy(&ox, dx, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&oy, dy, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&oz, dz, sizeof(int), cudaMemcpyDeviceToHost);
	printf("Exchanged values using non-divergent kernel: x = %d, y = %d, z = %d\n",
		ox, oy, oz);
	printf("non-divergent kernel runtime = %f msec\n", ms);

	cudaEventDestroy(stop);
	cudaEventDestroy(start);
	cudaFree(dx);
	cudaFree(dy);
	cudaFree(dz);
	return 0;
}
