#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <errno.h>
#include <stdbool.h>
#include <unistd.h>

#ifdef USE_AVX
#include <immintrin.h>
#endif

static inline void __set_input(double *in, unsigned long size,
			       unsigned int seed)
{
	unsigned int s, r;

	srand(seed);
	for (int idx = 0; idx < size; idx++) {
		r = rand_r(&s);
		in[idx] = (double)(r * (idx + 1)) / (r + idx + 1);
	}
}

void print_results(double *c, unsigned long size)
{
	for (int idx = 0; idx < size; idx++) {
		printf(" %6.6lf", c[idx]);
		if (idx && !idx % 15)
			printf("\n");
	}
	printf("\n");
}

void print_compute_time(struct timeval *st, struct timeval *end)
{
	long double sec, usec, delta;

	sec = end->tv_sec - st->tv_sec;
	usec = end->tv_usec - st->tv_usec;
	delta = sec * 1000000UL + usec;

	printf("Compute Time: %Lf usec\n", (long double)(delta));
}

#ifndef USE_AVX
int add_sequential(double *a, double *b, double *c,
		   unsigned long size)
{
	for (int idx = 0; idx < size; idx++)
		c[idx] = a[idx] + b[idx];
	return 0;
}

#else
int add_avx(double *a, double *b, double *c, unsigned long size)
{
	for (int idx = 0; idx < size; idx += 4) {
		__m256d A = _mm256_load_pd(&a[idx]);
		__m256d B = _mm256_load_pd(&b[idx]);
		__m256d C = _mm256_add_pd(A, B);
		_mm256_store_pd(&c[idx], C);
	}

	return 0;
}
#endif

int alloc_memory(double **a, double **b, double **c, long size)
{
#ifndef USE_AVX 
	*a = calloc(sizeof(double), size);
	if (!*a)
		return -ENOMEM;
	*b = calloc(sizeof(double), size);
	if (!*b)
		return -ENOMEM;
	*c = calloc(sizeof(double), size);
	if (!*c)
		return -ENOMEM;
#else
	*a = _mm_malloc(size * sizeof(double), 32);
	if (!*a)
		return -ENOMEM;
	*b = _mm_malloc(size * sizeof(double), 32);
	if (!*b)
		return -ENOMEM;
	*c = _mm_malloc(size * sizeof(double), 32);
	if (!*c)
		return -ENOMEM;
#endif
	return 0;
}

void free_alloc(double *a, double *b, double *c)
{
	free(c);
	free(b);
	free(a);
}


int main (int argc, char* argv[])
{
	double *a;
	double *b;
	double *c;
	int ret = 0;
	unsigned long size;
	bool verbose = false;
	struct timeval tv_start, tv_end;

	if (argc != 2) {
		printf("Usage: %s <input_size>\n", argv[0]);
		return -EINVAL;
	}

	size = atol(argv[1]);
	if (size <= 0) {
		printf("Invalid input");
		return -EINVAL;

	}
	if (size <= 20)
		verbose = true;

	ret = alloc_memory(&a, &b, &c, size);
	if (ret)
		goto out;

	__set_input(a, size, 89764125);
	__set_input(b, size, 34521769);

	gettimeofday(&tv_start, NULL);
#ifndef USE_AVX
	ret = add_sequential(a, b, c, size);
#else
	ret = add_avx(a, b, c, size);
#endif
	gettimeofday(&tv_end, NULL);

	if (verbose) {
		printf("Input A[%ld], Input B[%ld], Sum[%ld]\n",
			size, size, size);
		print_results(a, size);
		print_results(b, size);
		print_results(c, size);
	}

	print_compute_time(&tv_start, &tv_end);
out:
	free_alloc(a, b, c);
	return ret;
}
