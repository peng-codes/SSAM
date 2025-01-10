#include "../ssai-2dconv/common.h"
#include "../ssai-2dconv/cudaLib.cuh"

namespace stencil2d_v2 {
#define MAD(__x, __y, __z)   ((__x)*(__y) + (__z))
	static const int WARP_SIZE = 32;
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

		const int process_count = BLOCK_PROCESS_DATA_COUNT*blockIdx.x + WARP_PROCESS_DATA_COUNT*warpId;
		if (process_count >= width)
			return;
		int tidx = process_count + laneId - FILTER_WIDTH / 2;
		int tidy = PROCESS_DATA_COUNT*blockIdx.y - FILTER_HEIGHT / 2;


		__shared__ T smem[FILTER_HEIGHT][FILTER_WIDTH];
		T* psmem = &smem[0][0];
		for (int i = threadIdx.x; i < FILTER_HEIGHT*FILTER_WIDTH; i += blockDim.x)
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
		//		T* p = &data[0];
#pragma unroll
		for (int i = 0; i < PROCESS_DATA_COUNT; i++) {
			T sum = 0;
			sum = data[i + 1] * smem[1][0];

			sum = __my_shfl_up(sum, 1);
			sum += data[i + 0] * smem[0][1];
			sum += data[i + 1] * smem[1][1];
			sum += data[i + 2] * smem[2][1];

			sum = __my_shfl_up(sum, 1);
			sum += data[i + 2] * smem[1][2];

//#pragma unroll
//			for (int m = 0; m < FILTER_WIDTH; m++) {
//				if (m > 0) {
//					sum = __my_shfl_up(sum, 1);
//				}
//#pragma unroll
//				for (int n = 0; n < FILTER_HEIGHT; n++) {
//					//					int a = data[i + 0];
//
//					//sum += data[i + n] * smem[n][m];
//					sum = MAD(data[i + n], smem[n][m], sum);
//				}
//			}
			data[i] = sum;
		}

		index = widthStride*(tidy + FILTER_HEIGHT / 2) + tidx - (FILTER_WIDTH - 1) / 2;
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
	template<typename DataType, int PROCESS_DATA_COUNT, int BLOCK_SIZE>
	static float Test_j2d5pt(int width, int height) {
		const int FILTER_WIDTH = 3;
		const int FILTER_HEIGHT = 3;

		const int WARP_COUNT = BLOCK_SIZE >> 5;
		const int WARP_PROCESS_DATA_COUNT = WARP_SIZE - FILTER_WIDTH + 1;
		const int BLOCK_PROCESS_DATA_COUNT = WARP_PROCESS_DATA_COUNT*WARP_COUNT;

		const int nRepeatCount = 1;
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

#if 1
		DataType filter[FILTER_HEIGHT][FILTER_WIDTH] = {
			{ 0.0, },
			{ 0,  },
			//			{ -1.0,  -2.0, -1.0, },
		};
#endif

		DataType* flt = &filter[0][0];
		filter[1][0] = std::rand() % 21 - 10;
		filter[0][1] = std::rand() % 21 - 10;
		filter[1][1] = std::rand() % 21 - 10;
		filter[2][1] = std::rand() % 21 - 10;
		filter[1][2] = std::rand() % 21 - 10;

		//for (int i = 0; i < FILTER_HEIGHT*FILTER_WIDTH; i++) {
		//	flt[i] = std::rand() % 21 - 10;
		//}
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
		printf("%dx%d , %dx%d , datatype=%s, proc_count=%d, cache=%d, BLOCK_SIZE=%d, %f ms , %f fps\n", 
			width, height, FILTER_WIDTH, FILTER_HEIGHT, typeid(img.data[0]).name(), PROCESS_DATA_COUNT, PROCESS_DATA_COUNT + FILTER_HEIGHT - 1, BLOCK_SIZE, inc, 1000.0 / inc);
		sprintf(szPath, "../data/Lena_proc_%dx%d.raw", width, height);
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

//template<typename T>
//static int stencil_5pt_v2(int argc, char** argv) {
//	DISPLAY_FUNCTION("");
//	int size = 8192; if (argc > 1) size = atoi(argv[1]);
//	const int P = 4;
//	const int B = 128;
//	stencil2d_v2::Test_j2d5pt<T, P, B>(size, size);
//	return 0;
//}

//int stencil_5pt_double(int argc, char** argv) {
//	DISPLAY_FUNCTION("");
//	int size = 8192; if (argc > 1) size = atoi(argv[1]);
//	const int P = 4;
//	const int B = 128;
//	stencil2d_v2::Test_j2d5pt<double, P, B>(size, size);
//	return 0;
//}
//int stencil_5pt_float(int argc, char** argv) {
//	DISPLAY_FUNCTION("");
//	int size = 8192; if (argc > 1) size = atoi(argv[1]);
//	const int P = 4;
//	const int B = 128;
//	stencil2d_v2::Test_j2d5pt<float, P, B>(size, size);
//	return 0;
//}
