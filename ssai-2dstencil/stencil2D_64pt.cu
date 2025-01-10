#include "../ssai-2dconv/common.h"
#include "../ssai-2dconv/cudaLib.cuh"

#define    _F00         ((T)1.27449)   //1
#define    _F01         ((T)-0.000136017)   //2
#define    _F02         ((T)0.000714)   //3
#define    _F03         ((T)-0.002856)   //4
#define    _F05         ((T)0.002856)   //5
#define    _F06         ((T)-0.000714)   //6
#define    _F07         ((T)0.000136017)   //7
#define    _F08         ((T)-1.27449)   //8
#define    _F10         ((T)-0.000136017)   //9
#define    _F11         ((T)0.00145161)   //10
#define    _F12         ((T)-0.00762)   //11
#define    _F13         ((T)0.03048)   //12
#define    _F15         ((T)-0.03048)   //13
#define    _F16         ((T)0.00762)   //14
#define    _F17         ((T)-0.00145161)   //15
#define    _F18         ((T)0.000136017)   //16
#define    _F20         ((T)0.000714)   //17
#define    _F21         ((T)-0.00762)   //18
#define    _F22         ((T)0.04)   //19
#define    _F23         ((T)-10.16)   //20
#define    _F25         ((T)0.16)   //21
#define    _F26         ((T)-0.04)   //22
#define    _F27         ((T)0.00762)   //23
#define    _F28         ((T)-0.000714)   //24
#define    _F30         ((T)-0.002856)   //25
#define    _F31         ((T)0.03048)   //26
#define    _F32         ((T)-10.16)   //27
#define    _F33         ((T)0.64)   //28
#define    _F35         ((T)-0.64)   //29
#define    _F36         ((T)0.16)   //30
#define    _F37         ((T)-0.03048)   //31
#define    _F38         ((T)0.002856)   //32
#define    _F50         ((T)0.002856)   //33
#define    _F51         ((T)-0.03048)   //34
#define    _F52         ((T)0.16)   //35
#define    _F53         ((T)-0.64)   //36
#define    _F55         ((T)0.64)   //37
#define    _F56         ((T)-10.16)   //38
#define    _F57         ((T)0.03048)   //39
#define    _F58         ((T)-0.002856)   //40
#define    _F60         ((T)-0.000714)   //41
#define    _F61         ((T)0.00762)   //42
#define    _F62         ((T)-0.04)   //43
#define    _F63         ((T)0.16)   //44
#define    _F65         ((T)-10.16)   //45
#define    _F66         ((T)0.04)   //46
#define    _F67         ((T)-0.00762)   //47
#define    _F68         ((T)0.000714)   //48
#define    _F70         ((T)0.000136017)   //49
#define    _F71         ((T)-0.00145161)   //50
#define    _F72         ((T)0.00762)   //51
#define    _F73         ((T)-0.03048)   //52
#define    _F75         ((T)0.03048)   //53
#define    _F76         ((T)-0.00762)   //54
#define    _F77         ((T)0.00145161)   //55
#define    _F78         ((T)-0.000136017)   //56
#define    _F80         ((T)-1.27449)   //57
#define    _F81         ((T)0.000136017)   //58
#define    _F82         ((T)-0.000714)   //59
#define    _F83         ((T)0.002856)   //60
#define    _F85         ((T)-0.002856)   //61
#define    _F86         ((T)0.000714)   //62
#define    _F87         ((T)-0.000136017)   //63
#define    _F88         ((T)1.27449)   //64
namespace stencil2d_64pt {
	static const int WARP_SIZE = 32;
	static const int FILTER_WIDTH = 9;
	static const int FILTER_HEIGHT = 9;

	template<typename T, int BLOCK_SIZE, int PROCESS_DATA_COUNT>
	__global__ void j2d64pt(const T* __restrict__ src, T* dst, int width, int height)
	{
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
	
		#pragma unroll
		for (int i = 0; i < PROCESS_DATA_COUNT; i++) {
			T sum = 0;
			sum += data[i + 0] * _F08;
			sum += data[i + 1] * _F18;
			sum += data[i + 2] * _F28;
			sum += data[i + 3] * _F38;
			sum += data[i + 5] * _F58;
			sum += data[i + 6] * _F68;
			sum += data[i + 7] * _F78;
			sum += data[i + 8] * _F88;

			sum = __my_shfl_down(sum, 1);
			sum += data[i + 0] * _F07;
			sum += data[i + 1] * _F17;
			sum += data[i + 2] * _F27;
			sum += data[i + 3] * _F37;
			sum += data[i + 5] * _F57;
			sum += data[i + 6] * _F67;
			sum += data[i + 7] * _F77;
			sum += data[i + 8] * _F87;

			sum = __my_shfl_down(sum, 1);
			sum += data[i + 0] * _F06;
			sum += data[i + 1] * _F16;
			sum += data[i + 2] * _F26;
			sum += data[i + 3] * _F36;
			sum += data[i + 5] * _F56;
			sum += data[i + 6] * _F66;
			sum += data[i + 7] * _F76;
			sum += data[i + 8] * _F86;

			sum = __my_shfl_down(sum, 1);
			sum += data[i + 0] * _F05;
			sum += data[i + 1] * _F15;
			sum += data[i + 2] * _F25;
			sum += data[i + 3] * _F35;
			sum += data[i + 5] * _F55;
			sum += data[i + 6] * _F65;
			sum += data[i + 7] * _F75;
			sum += data[i + 8] * _F85;

			//sum = __my_shfl_down(sum, 1);

			sum = __my_shfl_down(sum, 2);
			sum += data[i + 0] * _F03;
			sum += data[i + 1] * _F13;
			sum += data[i + 2] * _F23;
			sum += data[i + 3] * _F33;
			sum += data[i + 5] * _F53;
			sum += data[i + 6] * _F63;
			sum += data[i + 7] * _F73;
			sum += data[i + 8] * _F83;

			sum = __my_shfl_down(sum, 1);
			sum += data[i + 0] * _F02;
			sum += data[i + 1] * _F12;
			sum += data[i + 2] * _F22;
			sum += data[i + 3] * _F32;
			sum += data[i + 5] * _F52;
			sum += data[i + 6] * _F62;
			sum += data[i + 7] * _F72;
			sum += data[i + 8] * _F82;

			sum = __my_shfl_down(sum, 1);
			sum += data[i + 0] * _F01;
			sum += data[i + 1] * _F11;
			sum += data[i + 2] * _F21;
			sum += data[i + 3] * _F31;
			sum += data[i + 5] * _F51;
			sum += data[i + 6] * _F61;
			sum += data[i + 7] * _F71;
			sum += data[i + 8] * _F81;

			sum = __my_shfl_down(sum, 1);
			sum += data[i + 0] * _F00;
			sum += data[i + 1] * _F10;
			sum += data[i + 2] * _F20;
			sum += data[i + 3] * _F30;
			sum += data[i + 5] * _F50;
			sum += data[i + 6] * _F60;
			sum += data[i + 7] * _F70;
			sum += data[i + 8] * _F80;

			data[i] = sum;
		}

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
	static float Test(int width, int height) {
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
				//img.data[i] = i/img.width;
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

		DataType filter[FILTER_HEIGHT][FILTER_WIDTH] =
		{
			{ 1.27449, -0.000136017, 0.000714, -0.002856, 0, 0.002856, -0.000714, 0.000136017, -1.27449, },
			{ -0.000136017, 0.00145161, -0.00762, 0.03048, 0, -0.03048, 0.00762, -0.00145161, 0.000136017, },
			{ 0.000714, -0.00762, 0.04, -10.16, 0, 0.16, -0.04, 0.00762, -0.000714, },
			{ -0.002856, 0.03048, -10.16, 0.64, 0, -0.64, 0.16, -0.03048, 0.002856, },
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, },
			{ 0.002856, -0.03048, 0.16, -0.64, 0, 0.64, -10.16, 0.03048, -0.002856, },
			{ -0.000714, 0.00762, -0.04, 0.16, 0, -10.16, 0.04, -0.00762, 0.000714, },
			{ 0.000136017, -0.00145161, 0.00762, -0.03048, 0, 0.03048, -0.00762, 0.00145161, -0.000136017, },
			{ -1.27449, 0.000136017, -0.000714, 0.002856, 0, -0.002856, 0.000714, -0.000136017, 1.27449, },
		};

		cudaEventRecord(start, 0);
		for (int s = 0; s < nRepeatCount; s++) {
			j2d64pt<DataType, BLOCK_SIZE, PROCESS_DATA_COUNT> <<<grid_size, block_size >>> 
				(devSrc.GetData(), devDst.GetData(), width, height);
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
		printf("verify dif =%f, avg-dif=%e\n", dif, dif/img.width/img.height);
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

int stencil_64pt_double(int argc, char** argv) {
	DISPLAY_FUNCTION("");
	printf("datatype=double\n");
	int size = 8192; if (argc > 1) size = atoi(argv[1]);
	const int P = 6;
	const int B = 128;
	stencil2d_64pt::Test<double, P, B>(size, size);
	return 0;
}

int stencil_64pt_float(int argc, char** argv) {
	DISPLAY_FUNCTION("");
	printf("datatype=double\n");
	int size = 8192; if (argc > 1) size = atoi(argv[1]);
	const int P = 4;
	const int B = 128;
	stencil2d_64pt::Test<float, P, B>(size, size);
	return 0;
}
