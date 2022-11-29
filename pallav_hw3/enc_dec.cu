#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <time.h>

struct file_details {
	int	file_info;
	size_t	file_sz;
	char	*file_ptr;
	char	*file_name;
};

char range[] = "abcdefghijklmnopqrstuvwxyz";
#define PRINT_THRESHOLD	200

static void *alloc_buf(int size, int alloc_int)
{
	int unit;

	unit = alloc_int ? sizeof(int) : sizeof(char);
	return calloc(size, unit);
}
static int alloc_all_buff(int size, char **en, char **d_en, int **key)
{
	*key = (int *)alloc_buf(size, true);
	if (!*key)
		goto out;
	*en = (char *)alloc_buf(size, false);
	if (!*en) {
		free(*key);
		goto out;
	}

	*d_en = (char *)alloc_buf(size, false);
	if (!*d_en) {
		free(*en);
		free(*key);
		goto out;
	}

	return 0;
out:
	return -ENOMEM;
}

static void free_all_buff(char *en, char *d_en, int *key)
{
	free(d_en);
	free(en);
	free(key);
}

static int write_file(struct file_details *file_info, void *in,
		      char *op, bool is_key)
{
	int of_file_info, cnt;
	char *file_name;

	cnt = asprintf(&file_name, "%s.%s", file_info->file_name, op);
	if (cnt < 0)
		return -EINVAL;
	of_file_info = open(file_name, O_CREAT|O_RDWR, 0666);
	if (of_file_info < 0)
		return -EBADF;
	cnt = is_key ? file_info->file_sz * sizeof(int) : file_info->file_sz;
	cnt = write(of_file_info, (char *)in, cnt);
	close(of_file_info);
	return cnt;
}

static int save_all_output(struct file_details *file_info, char *en,
			   char *d_en, int *key, bool is_gpu)
{
	int idx, cnt;
	char name[15];

	if (file_info->file_sz <= PRINT_THRESHOLD) {
		printf("input:\n");
		printf("%s", file_info->file_ptr);
		printf("\nEncrypted:\n");
		for (idx = 0; idx < file_info->file_sz; idx++)
			printf("%c", en[idx]);
		printf("\nDecrypted:\n");
		for (idx = 0; idx < file_info->file_sz; idx++)
			printf("%c", d_en[idx]);
		printf("\nKey:\n");
		for (idx = 0; idx < file_info->file_sz; idx++)
			printf("%2d", key[idx]);
		printf("\n");
	}
	sprintf(name, "%s", (is_gpu) ? "encrypted.gpu" : "encrypted");
	cnt = write_file(file_info, en, name, false);
	if (cnt < 0)
		printf("Failed to write encrypted file\n");
	sprintf(name, "%s", (is_gpu) ? "decrypted.gpu" : "decrypted");
	cnt = write_file(file_info, d_en, name, false);
	if (cnt < 0)
		printf("Failed to writed decrypted file\n");

	sprintf(name, "%s", (is_gpu) ? "key.gpu" : "key");
	cnt = write_file(file_info, key, name, true);
	if (cnt < 0)
		printf("Failed to write the key file\n");

	return 0;
}

static int cpu_encrypt(char *input, int *key, char *outp, size_t in_size)
{
	int idx, len, norm;
	char mod;

	len = strlen(range);
	for (idx = 0; idx < in_size; idx++) {
		if (!(input[idx] > 96 && input[idx] < 123)) {
			outp[idx] = input[idx];
			continue;
		}
		norm = input[idx] - 'a' + 1 + idx;
		mod = norm % len;
		outp[idx] = mod + 'a';
		key[idx] = norm / len;
	}

	return 0;
}

static int cpu_decrypt(char *input, int *key, char *outp, size_t in_size)
{
	int idx, norm, len, mod;
	char op;

	len = strlen(range);
	for (idx = 0; idx < in_size; idx++) {
		if (!(input[idx] > 96 && input[idx] < 123)) {
			outp[idx] = input[idx];
			continue;
		}
		mod = input[idx] - 'a';
		norm = mod + len * key[idx];
		op = (norm - 1 - idx) % len + 'a';
		outp[idx] = op;
		/* printf("in = %c, ch = %c\n", input[idx], op); */
	}

	return 0;
}

static void cpu_show_time(struct timespec *st, struct timespec *md,
			  struct timespec *en)
{
	double st_nsec, mid_nsec, en_nsec;

	st_nsec = st->tv_nsec + st->tv_sec * 1000000000ULL;
	mid_nsec = md->tv_nsec + md->tv_sec * 1000000000ULL;
	en_nsec = en->tv_nsec + en->tv_sec * 1000000000ULL;

	printf("Encryption time %lf usec\n", (mid_nsec - st_nsec) / 1000);
	printf("Decryption time %lf usec\n", (en_nsec - mid_nsec) / 1000);
}

static int cpu_encap_decap(struct file_details *f_det)
{
	struct timespec start, mid, end;
	char *encryp, *decryp;
	int *key;

	if (alloc_all_buff(f_det->file_sz, &encryp, &decryp, &key))
		goto out;

	clock_gettime(CLOCK_REALTIME, &start);
	cpu_encrypt(f_det->file_ptr, key, encryp, f_det->file_sz);
	clock_gettime(CLOCK_REALTIME, &mid);
	cpu_decrypt(encryp, key, decryp, f_det->file_sz);
	clock_gettime(CLOCK_REALTIME, &end);

	cpu_show_time(&start, &mid, &end);
	save_all_output(f_det, encryp, decryp, key, false);

	free_all_buff(encryp, decryp, key);
	return 0;
out:
	return -1;
}

#ifile_infoef USE_CUDA
struct mem_pointers {
	char	*d_in;
	char	*d_en;
	char	*h_en;
	char	*d_decap;
	char	*h_decap;
	int	*d_key;
	int	*h_key;
	int	*d_size;
	int	h_size;
};

#define gpuErrchk(ans)				\
{                                               \
	gpuAssert((ans), __FILE__, __LINE__);   \
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

__global__ void gpu_encryption_kernel(char *d_in, char *d_en,
				      int *d_key, int *d_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int len = 26, pred;
	char input;

	if (tid > *d_size)
		return;

	input = d_in[tid];
	pred = (input > 96 && input < 123);
	if (pred) {
		d_en[tid] = (input - 'a' + 1 + tid) % len + 'a';
		d_key[tid] = (input - 'a' + 1 + tid) / len;
	}
	if (!pred)
		d_en[tid] = input;
}

__global__ void gpu_decryption_kernel(char *d_in, int *d_key,
				      char *d_decap, int *d_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int len = 26, pred;
	char input;

	if (tid > *d_size)
		return;

	input = d_in[tid];
	pred = (input > 96 && input < 123);
	if (pred)
		d_decap[tid] = ((input - 'a' +
					len * d_key[tid] -
					1 - tid) % len + 'a');
	if (!pred)
		d_decap[tid] = input;
}
static void gpu_show_time(float *m1, float *m2)
{
	printf("Encryption time %lf usec\n", ((*m1) * 1000));
	printf("Decryption time %lf usec\n", ((*m2) * 1000));
}

static int gpu_alloc_all_buff(int size, struct mem_pointers *mp)
{
	if (alloc_all_buff(size, &mp->h_en, &mp->h_decap, &mp->h_key))
		return -ENOMEM;
	if(cudaMalloc(&mp->d_in, size) != cudaSuccess)
		goto out;
	if (cudaMalloc(&mp->d_en, size) != cudaSuccess)
		goto out1;
	if (cudaMalloc(&mp->d_decap, size) != cudaSuccess)
		goto out2;
	if (cudaMalloc(&mp->d_key, size * sizeof(int)) != cudaSuccess)
		goto out3;
	if (cudaMalloc(&mp->d_size, sizeof(int)) != cudaSuccess)
		goto out4;

	return 0;
out4:
	cudaFree(mp->d_key);
out3:
	cudaFree(mp->d_decap);
out2:
	cudaFree(mp->d_en);
out1:
	cudaFree(mp->d_in);
out:
	free_all_buff(mp->h_en, mp->h_decap, mp->h_key);
	return -ENOMEM;
}

static void gpu_free_all_buff(struct mem_pointers *mp)
{
	cudaFree(mp->d_key);
	cudaFree(mp->d_decap);
	cudaFree(mp->d_en);
	free_all_buff(mp->h_en, mp->h_decap, mp->h_key);
}

static int gpu_encap_decap(struct file_details *f_det)
{
	struct mem_pointers mem;
	cudaEvent_t start,stop;
	int ret, nblk, nth;
	float ms1, ms2;

	ret = gpu_alloc_all_buff(f_det->file_sz, &mem);
	if (ret)
		return ret;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	mem.h_size = f_det->file_sz;

	nth = mem.h_size;
	nblk = 1;
	if (mem.h_size > 1024) {
		nth = 1024;
		nblk = mem.h_size / nth;
		if (mem.h_size % nth)
			nblk++;
	}
	cudaMemcpy(mem.d_size, &mem.h_size, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(mem.d_in, f_det->file_ptr, mem.h_size, cudaMemcpyHostToDevice);

	cudaEventRecord(start);
	gpu_encryption_kernel<<<nblk, nth>>>(mem.d_in, mem.d_en, mem.d_key, mem.d_size);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms1,start,stop);
	gpuErrchk(cudaPeekAtLastError());
	cudaMemcpy(mem.h_en, mem.d_en, mem.h_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(mem.h_key, mem.d_key, mem.h_size * sizeof(int), cudaMemcpyDeviceToHost);
	cudaEventRecord(start);
	gpu_decryption_kernel<<<nblk, nth>>>(mem.d_en, mem.d_key, mem.d_decap, mem.d_size);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms2,start,stop);
	gpuErrchk(cudaPeekAtLastError());
	cudaMemcpy(mem.h_decap, mem.d_decap, mem.h_size, cudaMemcpyDeviceToHost);
	save_all_output(f_det, mem.h_en, mem.h_decap, mem.h_key, true);
	gpu_show_time(&ms1, &ms2);
	gpu_free_all_buff(&mem);

	return 0;
}
#endif

static int process_args(int argc, char *argv[], struct file_details *f_det)
{
	struct stat stat;
	int ret = 0, cnt;

	if (argc != 2) {
		printf("%s <input file path>\n", argv[0]);
		return -EINVAL;
	}

	cnt = asprintf(&f_det->file_name, "%s", argv[1]);
	if (cnt < 0)
		return -EINVAL;
	f_det->file_info = open(argv[1], O_RDWR);
	if (f_det->file_info <= 0)
		return -EBADF;

	ret = fstat(f_det->file_info, &stat);
	if (ret < 0) {
		close(f_det->file_info);
		return ret;
	}
	f_det->file_sz = stat.st_size;
	printf("size = %d\n", (int)stat.st_size);

	f_det->file_ptr = (char *)mmap(NULL, stat.st_size, PROT_READ|PROT_WRITE,
				    MAP_SHARED, f_det->file_info, 0);
	if (f_det->file_ptr == MAP_FAILED)
		ret = -ENOMEM;

	close(f_det->file_info);
	return ret;
}
int main(int argc, char *argv[])
{
	struct file_details f_det;
	int ret = 0;

	ret = process_args(argc, argv, &f_det);
	if (ret < 0)
		goto out;

	ret = cpu_encap_decap(&f_det);
	if (ret < 0)
		goto out1;
#ifile_infoef USE_CUDA
	ret = gpu_encap_decap(&f_det);
	if (ret < 0)
		goto out1;
#endif
out1:
	free(f_det.file_name);
	munmap(f_det.file_ptr, f_det.file_sz);
out:
	return ret;
}