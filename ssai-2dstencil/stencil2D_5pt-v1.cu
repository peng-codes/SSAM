#include "../ssai-2dconv/common.h"
#include "../ssai-2dconv/cudaLib.cuh"


namespace stencil2d_v1 {
	static const int WARP_SIZE = 32;

	template<typename T, int BLOCK_SIZE, int PROCESS_DATA_COUNT>
	__global__ void j2d5pt(const T* __restrict__ src, T* dst, int width, int height, T fc, T fn, T fs, T fw, T fe) 
	{
		const int FILTER_WIDTH  = 3;
		const int FILTER_HEIGHT = 3;
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

		{
			int index = width*tidy + tidx;
			if (tidx < 0)            index -= tidx;
			else if (tidx >= width)  index -= tidx - width + 1;
			if (tidy < 0)            index -= tidy*width;
			else if (tidy >= height) index -= (tidy - height + 1)*width;

#pragma unroll
			for (int s = 0; s < DATA_CACHE_SIZE; s++) {
				int _tidy = tidy + s;
				data[s] = src[index];
				if (_tidy >= 0 && _tidy < height - 1) {
					//data[s] = src[index];
					index += width;
				}
				//else {
				//	data[s] = 0;
				//}
			}
		}
#if 1		
		#pragma unroll
		for (int i = 0; i < PROCESS_DATA_COUNT; i++) {
			T sum = 0;
			sum = data[i + 1] * fe;

			sum = __my_shfl_down(sum, 1);
			sum += data[i + 0] * fn;
			sum += data[i + 1] * fc;
			sum += data[i + 2] * fs;

			sum = __my_shfl_down(sum, 1);
			sum += data[i + 1] * fw;

			data[i] = sum;
		}
#else
#pragma unroll
		for (int i = 0; i < PROCESS_DATA_COUNT; i += 4) {
			T sum0 = 0;
			T sum1 = 0;
			T sum2 = 0;
			T sum3 = 0;
			sum0 = data[i + 1] * fe; 
			sum1 = data[i + 1 + 1] * fe;
			sum2 = data[i + 1 + 2] * fe;
			sum3 = data[i + 1 + 3] * fe;

			sum0 = __my_shfl_down(sum0, 1);
			sum1 = __my_shfl_down(sum1, 1);
			sum2 = __my_shfl_down(sum2, 1);
			sum3 = __my_shfl_down(sum3, 1);

			sum0 += data[i + 0 + 0] * fn;
			sum1 += data[i + 0 + 1] * fn;
			sum2 += data[i + 0 + 2] * fn;
			sum3 += data[i + 0 + 3] * fn;


			sum0 += data[i + 1 + 0] * fc;
			sum1 += data[i + 1 + 1] * fc;
			sum2 += data[i + 1 + 2] * fc;
			sum3 += data[i + 1 + 3] * fc;


			sum0 += data[i + 2 + 0] * fs;
			sum1 += data[i + 2 + 1] * fs;
			sum2 += data[i + 2 + 2] * fs;
			sum3 += data[i + 2 + 3] * fs;

			sum0 = __my_shfl_down(sum0, 1);
			sum1 = __my_shfl_down(sum1, 1);
			sum2 = __my_shfl_down(sum2, 1);
			sum3 = __my_shfl_down(sum3, 1);

			sum0 += data[i + 1 + 0] * fw;
			sum1 += data[i + 1 + 1] * fw;
			sum2 += data[i + 1 + 2] * fw;
			sum3 += data[i + 1 + 3] * fw;

			data[i + 0] = sum0;
			data[i + 1] = sum1;
			data[i + 2] = sum2;
			data[i + 3] = sum3;
		}
#endif
		if (laneId >= WARP_SIZE - FILTER_WIDTH + 1)
			return;

		int _x = tidx + FILTER_WIDTH / 2;
		int _y = tidy + FILTER_HEIGHT / 2;
		int index = width*_y + _x;
		if (_x >= width)
			return;
#pragma unroll
		for (int i = 0; i < PROCESS_DATA_COUNT; i++) {
			if (_y + i < height) {
				dst[index] = data[i];
				index += width;
			}
		}
	}

	template<class DataType, int PROCESS_DATA_COUNT, int BLOCK_SIZE>
	static float Test_j2d5pt(int width, int height) {
		std::srand(std::time(nullptr));

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
		bool bRtn = img.Load_uchar(szPath, width, height);
		//sprintf(szPath, "../data/Lena%dx%d.txt", width, height);
		//img.SaveText(szPath);
		if (!bRtn) {
			printf("Load failed : %s, generate random data\n", szPath);
			img.MallocBuffer(width, height);
			for (int i = 0; i < img.width*img.height; i++) {
				img.data[i] = std::rand() % 256;
				img.data[i] = i/img.width;
			}
		}
		else {
			printf("Load success : %s\n", szPath);
		}

		DevBuffer<DataType> devSrc(width, height), devDst(width, height);
		devSrc.CopyFromHost(img.data, img.width, img.width, img.height);
		DataT<DataType> imgDst;
		imgDst.MallocBuffer(width, height);

		dim3 block_size(BLOCK_SIZE, 1);
		dim3 grid_size(UpDivide(width, BLOCK_PROCESS_DATA_COUNT), UpDivide(height, PROCESS_DATA_COUNT));

		DataType filter[FILTER_HEIGHT][FILTER_WIDTH] = {
			{ 0,  1, 0, },
			{ 4, 1, 2, },
			{ 0,  3, 0, },
		};
		filter[1][1] = std::rand() % 20 - 10;
		filter[0][1] = std::rand() % 20 - 10;
		filter[2][1] = std::rand() % 20 - 10;
		filter[1][0] = std::rand() % 20 - 10;
		filter[1][2] = std::rand() % 20 - 10;

		DataType fc = filter[1][1];
		DataType fn = filter[0][1];
		DataType fs = filter[2][1];
		DataType fw = filter[1][0];
		DataType fe = filter[1][2];

		cudaEventRecord(start, 0);
		for (int s = 0; s < nRepeatCount; s++) {
			j2d5pt<DataType, BLOCK_SIZE, PROCESS_DATA_COUNT> <<<grid_size, block_size >>> 
				(devSrc.GetData(), devDst.GetData(), width, height, fc, fn, fs, fw, fe);
		}
		cudaDeviceSynchronize();
		//watch.stop();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		CUDA_CHECK_ERROR;

		devDst.CopyToHost(imgDst.data, imgDst.width, imgDst.width, imgDst.height);

		cudaEventElapsedTime(&inc, start, stop);
		//inc = watch.getAverageTime();
		inc /= (float)nRepeatCount;
		printf("%dx%d , %dx%d , proc_count=%d, cache=%d, BLOCK_SIZE=%d, %f ms , %f fps\n", width, height, FILTER_WIDTH, FILTER_HEIGHT, PROCESS_DATA_COUNT, PROCESS_DATA_COUNT + FILTER_HEIGHT - 1, BLOCK_SIZE, inc, 1000.0 / inc);
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
			int x = i % img.width;
			int y = i / img.width;
			if (x > FILTER_WIDTH/2 && x < width - FILTER_WIDTH/2 && y > FILTER_HEIGHT/2 && y < height - FILTER_HEIGHT/2)
				dif += abs(imgVerify.data[i] - imgDst.data[i]);
		}
		printf("verify dif =%f\n", dif);
		sprintf(szPath, "../data/Lena_proc_verify_%dx%d.txt", width, height);
		//imgVerify.SaveText(szPath);
		sprintf(szPath, "../data/Lena_proc_verify(%dx%d)_%dx%d.raw", FILTER_WIDTH, FILTER_HEIGHT, width, height);
		//imgVerify.SaveRaw(szPath);
#if 0
		FILE* fp = fopen("log.conv2D.csv", "at");
		if (fp) {
			fprintf(fp, "%dx%d, %d_%d, %d, %dx%d, %f\n", width, height, PROCESS_DATA_COUNT, PROCESS_DATA_COUNT + FILTER_HEIGHT - 1, BLOCK_SIZE, FILTER_WIDTH, FILTER_HEIGHT, inc);
			fclose(fp);
		}
		return inc;
#endif
	}
};

int stencil_5pt_v1(int argc, char** argv) {
	DISPLAY_FUNCTION("");
	printf("datatype=double\n");
	int size = 8192; if (argc > 1) size = atoi(argv[1]);
	const int P = 4;
	const int B = 128;
	stencil2d_v1::Test_j2d5pt<double, P, B>(size, size);
	return 0;
}

int stencil_5pt_double(int argc, char** argv) {
	DISPLAY_FUNCTION("");
	int size = 8192; if (argc > 1) size = atoi(argv[1]);
	const int P = 4;
	const int B = 128;
	stencil2d_v1::Test_j2d5pt<double, P, B>(size, size);
	return 0;
}
int stencil_5pt_float(int argc, char** argv) {
	DISPLAY_FUNCTION("");
	int size = 8192; if (argc > 1) size = atoi(argv[1]);
	const int P = 4;
	const int B = 128;
	stencil2d_v1::Test_j2d5pt<float, P, B>(size, size);
	return 0;
}


