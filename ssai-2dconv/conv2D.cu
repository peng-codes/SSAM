#include "./common.h"
#include "./cudaLib.cuh"
#include <ctime>

namespace CONV2D {
	static const int WARP_SIZE = 32;
#define MAD(__x, __y, __z) ((__x)*(__y) + (__z))

	template<typename T, int BLOCK_SIZE, int PROCESS_DATA_COUNT, int FILTER_WIDTH, int FILTER_HEIGHT>
	__global__ void kernel_convolution2D(
		const T* __restrict__ src, T* dst, int width, int widthStride, int height,
		const T* __restrict__ weight) {
#if 1
		const int WARP_COUNT = BLOCK_SIZE >> 5;
		const int laneId = threadIdx.x & 31;
		const int warpId = threadIdx.x >> 5;
		const int WARP_PROCESS_DATA_COUNT = WARP_SIZE - FILTER_WIDTH + 1;
		const int BLOCK_PROCESS_DATA_COUNT = WARP_PROCESS_DATA_COUNT*WARP_COUNT;
		const int DATA_CACHE_SIZE = PROCESS_DATA_COUNT + FILTER_HEIGHT - 1;

		T data[DATA_CACHE_SIZE];

		int tidx = BLOCK_PROCESS_DATA_COUNT*blockIdx.x + WARP_PROCESS_DATA_COUNT*warpId + laneId - FILTER_WIDTH / 2;
		int tidy = PROCESS_DATA_COUNT*blockIdx.y - FILTER_HEIGHT / 2;

		__shared__ T smem[FILTER_HEIGHT][FILTER_WIDTH];
		T* psmem = &smem[0][0];
		for (int i=threadIdx.x; i < FILTER_HEIGHT*FILTER_WIDTH; i += blockDim.x)
				psmem[i] = weight[i];
		__syncthreads();

		int index = widthStride*tidy + tidx;
#pragma unroll
		for (int s = 0; s < DATA_CACHE_SIZE; s++) {
			int _tidy = tidy + s;
			if (tidx >= 0 && tidx < width && _tidy >= 0 && _tidy < height) {
				data[s] = src[index];
			}
			else {
				data[s] = 0;
			}
			index += widthStride;
		}
#pragma unroll
		for (int i = 0; i < PROCESS_DATA_COUNT; i++) {
			T sum = 0;
#pragma unroll
			for (int m = 0; m < FILTER_WIDTH; m++) {
				if (m > 0) {
					sum = __my_shfl_up(sum, 1);
				}
#pragma unroll
				for (int n = 0; n < FILTER_HEIGHT; n++) {
					sum = MAD(data[i + n], smem[n][m], sum);
				}
			}
			data[i] = sum;
		}

		index = widthStride*(tidy + FILTER_HEIGHT / 2) + tidx - (FILTER_WIDTH - 1)/2;
#pragma unroll
		for (int i = 0; i < PROCESS_DATA_COUNT; i++) {
			if (laneId >= FILTER_WIDTH - 1 && tidx - (FILTER_WIDTH - 1) / 2 < width && tidy + FILTER_HEIGHT / 2 + i < height) {
				dst[index] = data[i];
			}
			index += widthStride;
		}
		/**/
#endif
	}

	template<class DataType, int FILTER_WIDTH, int FILTER_HEIGHT, int PROCESS_DATA_COUNT, int BLOCK_SIZE>
	static float Test_conv2D(int width, int height) {
		//typedef double DataType;
		printf("file_%s : func_%s : line_%d, using dtype=%s\n", __FILE__, __FUNCTION__, __LINE__, sizeof(DataType) == 8 ? "double" : "float32");

		const int WARP_COUNT = BLOCK_SIZE >> 5;
		const int WARP_PROCESS_DATA_COUNT = WARP_SIZE - FILTER_WIDTH + 1;
		const int BLOCK_PROCESS_DATA_COUNT = WARP_PROCESS_DATA_COUNT*WARP_COUNT;

		const int nRepeatCount = 100;
		float inc = 0;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		//StopWatchWin watch;
		DataT<DataType> img;
		char szPath[1024] = "";
		sprintf(szPath, "../data/Lena%dx%d.raw", width, height);
		bool bRtn = false;
		bRtn = img.Load_uchar(szPath, width, height);
		//sprintf(szPath, "../data/Lena%dx%d.txt", width, height);
		//img.SaveText(szPath);
		if (!bRtn) {
			printf("Load failed : %s, generate random data\n", szPath);
			img.MallocBuffer(width, height);
			for (int i = 0; i < img.width*img.height; i++) {
				img.data[i] = std::rand() % 256;
			}
		}
		else {
			printf("Load success : %s\n", szPath);
		}
		DevData<DataType> devSrc(width, height), devDst(width, height);
		devSrc.CopyFromHost(img.data, img.width, img.width, img.height);
		DataT<DataType> imgDst;
		imgDst.MallocBuffer(width, height);

		dim3 block_size(BLOCK_SIZE, 1);
		dim3 grid_size(UpDivide(width, BLOCK_PROCESS_DATA_COUNT), UpDivide(height, PROCESS_DATA_COUNT));

		DataType filter[FILTER_HEIGHT][FILTER_WIDTH] = {{ 1.0,  0.0, },{ 0,  -1.0, },};

		std::srand(std::time(nullptr));
		DataType* flt = &filter[0][0];
		for (int i = 0; i < FILTER_HEIGHT*FILTER_WIDTH; i++) {
			flt[i] = std::rand() % 21 - 10;
		}
		//flt[FILTER_HEIGHT*FILTER_WIDTH / 2] = FILTER_HEIGHT*FILTER_WIDTH - 1;

		DevData<DataType> devFilter(FILTER_HEIGHT*FILTER_WIDTH);
		devFilter.CopyFromHost(&filter[0][0], FILTER_WIDTH*FILTER_HEIGHT, FILTER_WIDTH*FILTER_HEIGHT, 1);

		cudaEventRecord(start, 0);
		//watch.start();
		//Conv2D(devSrc.GetData(), devDst.GetData(), width, devSrc.DataPitch(), height, devFilter.GetData(), FILTER_WIDTH, FILTER_HEIGHT);
		for (int s = 0; s < nRepeatCount; s++) {
			kernel_convolution2D<DataType, BLOCK_SIZE, PROCESS_DATA_COUNT, FILTER_WIDTH, FILTER_HEIGHT> << <grid_size, block_size >> > (devSrc.GetData(), devDst.GetData(), width, devSrc.DataPitch(), height, devFilter.GetData());
			cudaDeviceSynchronize();
		}
		//watch.stop();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		CUDA_CHECK_ERROR;

		devDst.CopyToHost(imgDst.data, imgDst.width, imgDst.width, imgDst.height);

		cudaEventElapsedTime(&inc, start, stop);
		//inc = watch.getAverageTime();
		inc /= (float)nRepeatCount;
		printf("%dx%d , %dx%d , proc_count=%d, cache=%d, BLOCK_SIZE=%d, %f ms , %f fps\n", width, height, FILTER_WIDTH, FILTER_HEIGHT, PROCESS_DATA_COUNT, PROCESS_DATA_COUNT + FILTER_HEIGHT - 1, BLOCK_SIZE, inc, 1000.0 / inc);
		sprintf(szPath, "../data/Lena_proc_%dx%d(%dx%d).raw", width, height, FILTER_WIDTH, FILTER_HEIGHT);
		//imgDst.SaveRaw(szPath);

		sprintf(szPath, "../data/Lena_proc_%dx%d.txt", width, height);
		//imgDst.SaveText(szPath);

		DataT<DataType> imgVerify;
		imgVerify.MallocBuffer(width, height);
		Convolution(img.data, imgVerify.data, width, height, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
		sprintf(szPath, "../data/Lena_proc_verify_%dx%d.txt", width, height);
		//imgVerify.SaveText(szPath);

		double dif = 0;
		for (int i = 0; i < img.width*img.height; i++) {
			dif += abs(imgVerify.data[i] - imgDst.data[i]);
		}
		printf("verify dif =%f\n", dif);
		sprintf(szPath, "../data/Lena_proc_verify_%dx%d.txt", width, height);
		//imgVerify.SaveText(szPath);
		sprintf(szPath, "../data/Lena_proc_verify(%dx%d)_%dx%d.raw", FILTER_WIDTH, FILTER_HEIGHT, width, height);
		//imgVerify.SaveRaw(szPath);

		FILE* fp = fopen("log.conv2D.csv", "at");
		if (fp) {
			fprintf(fp, "%dx%d, %d_%d, %d, %dx%d, %f\n", width, height, PROCESS_DATA_COUNT, PROCESS_DATA_COUNT + FILTER_HEIGHT - 1, BLOCK_SIZE, FILTER_WIDTH, FILTER_HEIGHT, inc);
			fclose(fp);
		}
		return inc;
	}
};


template<typename DataType>
void Test_2D(int argc, char** argv) {
	DISPLAY_FUNCTION("evaluate");
	int width = 8192;
	int height = 8192;
	int fsize = 3;
	if (argc >= 3) width = height = atoi(argv[2]);
	if (argc >= 4) fsize = atoi(argv[3]);

	const int PROCESS_DATA_COUNT = 4;
	const int BLOCK_SIZE = 128;
	if (fsize == 2) CONV2D::Test_conv2D<DataType, 2, 2, PROCESS_DATA_COUNT, BLOCK_SIZE>(width, height);
	if (fsize == 3) CONV2D::Test_conv2D<DataType, 3, 3, PROCESS_DATA_COUNT, BLOCK_SIZE>(width, height);
	if (fsize == 4) CONV2D::Test_conv2D<DataType, 4, 4, PROCESS_DATA_COUNT, BLOCK_SIZE>(width, height);
	if (fsize == 5) CONV2D::Test_conv2D<DataType, 5, 5, PROCESS_DATA_COUNT, BLOCK_SIZE>(width, height);
	if (fsize == 6) CONV2D::Test_conv2D<DataType, 6, 6, PROCESS_DATA_COUNT, BLOCK_SIZE>(width, height);
	if (fsize == 7) CONV2D::Test_conv2D<DataType, 7, 7, PROCESS_DATA_COUNT, BLOCK_SIZE>(width, height);
	if (fsize == 8) CONV2D::Test_conv2D<DataType, 8, 8, PROCESS_DATA_COUNT, BLOCK_SIZE>(width, height);
	if (fsize == 9) CONV2D::Test_conv2D<DataType, 9, 9, PROCESS_DATA_COUNT, BLOCK_SIZE>(width, height);
	if (fsize == 10) CONV2D::Test_conv2D<DataType, 10, 10, PROCESS_DATA_COUNT, BLOCK_SIZE>(width, height);
	if (fsize == 11) CONV2D::Test_conv2D<DataType, 11, 11, PROCESS_DATA_COUNT, BLOCK_SIZE>(width, height);
	if (fsize == 12) CONV2D::Test_conv2D<DataType, 12, 12, PROCESS_DATA_COUNT, BLOCK_SIZE>(width, height);
	if (fsize == 13) CONV2D::Test_conv2D<DataType, 13, 13, PROCESS_DATA_COUNT, BLOCK_SIZE>(width, height);
	if (fsize == 14) CONV2D::Test_conv2D<DataType, 14, 14, PROCESS_DATA_COUNT, BLOCK_SIZE>(width, height);
	if (fsize == 15) CONV2D::Test_conv2D<DataType, 15, 15, PROCESS_DATA_COUNT, BLOCK_SIZE>(width, height);
	if (fsize == 16) CONV2D::Test_conv2D<DataType, 16, 16, PROCESS_DATA_COUNT, BLOCK_SIZE>(width, height);
	if (fsize == 17) CONV2D::Test_conv2D<DataType, 17, 17, PROCESS_DATA_COUNT, BLOCK_SIZE>(width, height);
	if (fsize == 18) CONV2D::Test_conv2D<DataType, 18, 18, PROCESS_DATA_COUNT, BLOCK_SIZE>(width, height);
	if (fsize == 19) CONV2D::Test_conv2D<DataType, 19, 19, PROCESS_DATA_COUNT, BLOCK_SIZE>(width, height);
	if (fsize == 20) CONV2D::Test_conv2D<DataType, 20, 20, PROCESS_DATA_COUNT, BLOCK_SIZE>(width, height);
	if (fsize == 21) CONV2D::Test_conv2D<DataType, 21, 21, PROCESS_DATA_COUNT, BLOCK_SIZE>(width, height);
	if (fsize == 22) CONV2D::Test_conv2D<DataType, 22, 22, PROCESS_DATA_COUNT, BLOCK_SIZE>(width, height);
	if (fsize == 23) CONV2D::Test_conv2D<DataType, 23, 23, PROCESS_DATA_COUNT, BLOCK_SIZE>(width, height);
	if (fsize == 24) CONV2D::Test_conv2D<DataType, 24, 24, PROCESS_DATA_COUNT, BLOCK_SIZE>(width, height);
	if (fsize == 25) CONV2D::Test_conv2D<DataType, 25, 25, PROCESS_DATA_COUNT, BLOCK_SIZE>(width, height);
	if (fsize == 26) CONV2D::Test_conv2D<DataType, 26, 26, PROCESS_DATA_COUNT, BLOCK_SIZE>(width, height);
	if (fsize == 27) CONV2D::Test_conv2D<DataType, 27, 27, PROCESS_DATA_COUNT, BLOCK_SIZE>(width, height);
	if (fsize == 28) CONV2D::Test_conv2D<DataType, 28, 28, PROCESS_DATA_COUNT, BLOCK_SIZE>(width, height);

#if 0
	int width = 8192;
	if (argc > 1) width = atoi(argv[1]);
	int height = width;
	const int PROCESS_DATA_COUNT = 4;
	{
		{
			const int BLOCK_SIZE = 128;
			{ const int FilterSize = 3;  CONV2D::Test_conv2D<FilterSize, FilterSize, PROCESS_DATA_COUNT, BLOCK_SIZE>(width, height); }
			{ const int FilterSize = 5;  CONV2D::Test_conv2D<FilterSize, FilterSize, PROCESS_DATA_COUNT, BLOCK_SIZE>(width, height); }
			{ const int FilterSize = 7;  CONV2D::Test_conv2D<FilterSize, FilterSize, PROCESS_DATA_COUNT, BLOCK_SIZE>(width, height); }
			{ const int FilterSize = 8;  CONV2D::Test_conv2D<FilterSize, FilterSize, PROCESS_DATA_COUNT, BLOCK_SIZE>(width, height); }
			{ const int FilterSize = 9;  CONV2D::Test_conv2D<FilterSize, FilterSize, PROCESS_DATA_COUNT, BLOCK_SIZE>(width, height); }
		}
	}
#endif
}

int main(int argc, char** argv) {
	for (int i = 0; i < argc; i++) {
		printf("%s ", argv[i]);
		if (i == argc - 1) printf("\n");
	}
	if (argc < 4) {
		printf("error, argc number!");
	}
	const char* pdtype = argv[1];
	if (strcmp(pdtype, "float") == 0)
		Test_2D<float>(argc, argv);
	else
		Test_2D<double>(argc, argv);
	return 0;
}



