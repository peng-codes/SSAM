#include "../ssai-2dconv/common.h"
#include "../ssai-2dconv/cudaLib.cuh"

#define    _F00         ((T)3.18622)   //1
#define    _F01         ((T)4.5339)   //2
#define    _F02         ((T)-0.000357)   //3
#define    _F03         ((T)0.002856)   //4
#define    _F04         ((T)-0.00508225)   //5
#define    _F05         ((T)0.002856)   //6
#define    _F06         ((T)-0.000357)   //7
#define    _F07         ((T)4.5339)   //8
#define    _F08         ((T)3.18622)   //9
#define    _F10         ((T)4.5339)   //10
#define    _F11         ((T)0.00064516)   //11
#define    _F12         ((T)-0.00508)   //12
#define    _F13         ((T)0.04064)   //13
#define    _F14         ((T)-0.0723189)   //14
#define    _F15         ((T)0.04064)   //15
#define    _F16         ((T)-0.00508)   //16
#define    _F17         ((T)0.00064516)   //17
#define    _F18         ((T)4.5339)   //18
#define    _F20         ((T)-0.000357)   //19
#define    _F21         ((T)-0.00508)   //20
#define    _F22         ((T)0.04)   //21
#define    _F23         ((T)-0.32)   //22
#define    _F24         ((T)0.56944)   //23
#define    _F25         ((T)-0.32)   //24
#define    _F26         ((T)0.04)   //25
#define    _F27         ((T)-0.00508)   //26
#define    _F28         ((T)-0.000357)   //27
#define    _F30         ((T)0.002856)   //28
#define    _F31         ((T)0.04064)   //29
#define    _F32         ((T)-0.32)   //30
#define    _F33         ((T)2.56)   //31
#define    _F34         ((T)-4.55552)   //32
#define    _F35         ((T)2.56)   //33
#define    _F36         ((T)-0.32)   //34
#define    _F37         ((T)0.04064)   //35
#define    _F38         ((T)0.002856)   //36
#define    _F40         ((T)-0.00508225)   //37
#define    _F41         ((T)-0.0723189)   //38
#define    _F42         ((T)0.56944)   //39
#define    _F43         ((T)-4.55552)   //40
#define    _F44         ((T)8.10655)   //41
#define    _F45         ((T)-4.55552)   //42
#define    _F46         ((T)0.56944)   //43
#define    _F47         ((T)-0.0723189)   //44
#define    _F48         ((T)-0.00508225)   //45
#define    _F50         ((T)0.002856)   //46
#define    _F51         ((T)0.04064)   //47
#define    _F52         ((T)-0.32)   //48
#define    _F53         ((T)2.56)   //49
#define    _F54         ((T)-4.55552)   //50
#define    _F55         ((T)2.56)   //51
#define    _F56         ((T)-0.32)   //52
#define    _F57         ((T)0.04064)   //53
#define    _F58         ((T)0.002856)   //54
#define    _F60         ((T)-0.000357)   //55
#define    _F61         ((T)-0.00508)   //56
#define    _F62         ((T)0.04)   //57
#define    _F63         ((T)-0.32)   //58
#define    _F64         ((T)0.56944)   //59
#define    _F65         ((T)-0.32)   //60
#define    _F66         ((T)0.04)   //61
#define    _F67         ((T)-0.00508)   //62
#define    _F68         ((T)-0.000357)   //63
#define    _F70         ((T)4.5339)   //64
#define    _F71         ((T)0.00064516)   //65
#define    _F72         ((T)-0.00508)   //66
#define    _F73         ((T)0.04064)   //67
#define    _F74         ((T)-0.0723189)   //68
#define    _F75         ((T)0.04064)   //69
#define    _F76         ((T)-0.00508)   //70
#define    _F77         ((T)0.00064516)   //71
#define    _F78         ((T)4.5339)   //72
#define    _F80         ((T)3.18622)   //73
#define    _F81         ((T)4.5339)   //74
#define    _F82         ((T)-0.000357)   //75
#define    _F83         ((T)0.002856)   //76
#define    _F84         ((T)-0.00508225)   //77
#define    _F85         ((T)0.002856)   //78
#define    _F86         ((T)-0.000357)   //79
#define    _F87         ((T)4.5339)   //80
#define    _F88         ((T)3.18622)   //81

namespace stencil2d_81 {
	static const int WARP_SIZE = 32;
	static const int FILTER_WIDTH = 9;
	static const int FILTER_HEIGHT = 9;

	template<typename T, int BLOCK_SIZE, int PROCESS_DATA_COUNT>
	__global__ void j2d81pt(const T* __restrict__ src, T* dst, int width, int height)
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
			sum += data[i + 4] * _F48;
			sum += data[i + 5] * _F58;
			sum += data[i + 6] * _F68;
			sum += data[i + 7] * _F78;
			sum += data[i + 8] * _F88;

			sum = __my_shfl_down(sum, 1);
			sum += data[i + 0] * _F07;
			sum += data[i + 1] * _F17;
			sum += data[i + 2] * _F27;
			sum += data[i + 3] * _F37;
			sum += data[i + 4] * _F47;
			sum += data[i + 5] * _F57;
			sum += data[i + 6] * _F67;
			sum += data[i + 7] * _F77;
			sum += data[i + 8] * _F87;

			sum = __my_shfl_down(sum, 1);
			sum += data[i + 0] * _F06;
			sum += data[i + 1] * _F16;
			sum += data[i + 2] * _F26;
			sum += data[i + 3] * _F36;
			sum += data[i + 4] * _F46;
			sum += data[i + 5] * _F56;
			sum += data[i + 6] * _F66;
			sum += data[i + 7] * _F76;
			sum += data[i + 8] * _F86;

			sum = __my_shfl_down(sum, 1);
			sum += data[i + 0] * _F05;
			sum += data[i + 1] * _F15;
			sum += data[i + 2] * _F25;
			sum += data[i + 3] * _F35;
			sum += data[i + 4] * _F45;
			sum += data[i + 5] * _F55;
			sum += data[i + 6] * _F65;
			sum += data[i + 7] * _F75;
			sum += data[i + 8] * _F85;

			sum = __my_shfl_down(sum, 1);
			sum += data[i + 0] * _F04;
			sum += data[i + 1] * _F14;
			sum += data[i + 2] * _F24;
			sum += data[i + 3] * _F34;
			sum += data[i + 4] * _F44;
			sum += data[i + 5] * _F54;
			sum += data[i + 6] * _F64;
			sum += data[i + 7] * _F74;
			sum += data[i + 8] * _F84;

			sum = __my_shfl_down(sum, 1);
			sum += data[i + 0] * _F03;
			sum += data[i + 1] * _F13;
			sum += data[i + 2] * _F23;
			sum += data[i + 3] * _F33;
			sum += data[i + 4] * _F43;
			sum += data[i + 5] * _F53;
			sum += data[i + 6] * _F63;
			sum += data[i + 7] * _F73;
			sum += data[i + 8] * _F83;

			sum = __my_shfl_down(sum, 1);
			sum += data[i + 0] * _F02;
			sum += data[i + 1] * _F12;
			sum += data[i + 2] * _F22;
			sum += data[i + 3] * _F32;
			sum += data[i + 4] * _F42;
			sum += data[i + 5] * _F52;
			sum += data[i + 6] * _F62;
			sum += data[i + 7] * _F72;
			sum += data[i + 8] * _F82;

			sum = __my_shfl_down(sum, 1);
			sum += data[i + 0] * _F01;
			sum += data[i + 1] * _F11;
			sum += data[i + 2] * _F21;
			sum += data[i + 3] * _F31;
			sum += data[i + 4] * _F41;
			sum += data[i + 5] * _F51;
			sum += data[i + 6] * _F61;
			sum += data[i + 7] * _F71;
			sum += data[i + 8] * _F81;

			sum = __my_shfl_down(sum, 1);
			sum += data[i + 0] * _F00;
			sum += data[i + 1] * _F10;
			sum += data[i + 2] * _F20;
			sum += data[i + 3] * _F30;
			sum += data[i + 4] * _F40;
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
			{ 3.18622, 4.5339, -0.000357, 0.002856, -0.00508225, 0.002856, -0.000357, 4.5339, 3.18622, },
			{ 4.5339, 0.00064516, -0.00508, 0.04064, -0.0723189, 0.04064, -0.00508, 0.00064516, 4.5339, },
			{ -0.000357, -0.00508, 0.04, -0.32, 0.56944, -0.32, 0.04, -0.00508, -0.000357, },
			{ 0.002856, 0.04064, -0.32, 2.56, -4.55552, 2.56, -0.32, 0.04064, 0.002856, },
			{ -0.00508225, -0.0723189, 0.56944, -4.55552, 8.10655, -4.55552, 0.56944, -0.0723189, -0.00508225, },
			{ 0.002856, 0.04064, -0.32, 2.56, -4.55552, 2.56, -0.32, 0.04064, 0.002856, },
			{ -0.000357, -0.00508, 0.04, -0.32, 0.56944, -0.32, 0.04, -0.00508, -0.000357, },
			{ 4.5339, 0.00064516, -0.00508, 0.04064, -0.0723189, 0.04064, -0.00508, 0.00064516, 4.5339, },
			{ 3.18622, 4.5339, -0.000357, 0.002856, -0.00508225, 0.002856, -0.000357, 4.5339, 3.18622, },
		};

		cudaEventRecord(start, 0);
		for (int s = 0; s < nRepeatCount; s++) {
			j2d81pt<DataType, BLOCK_SIZE, PROCESS_DATA_COUNT> <<<grid_size, block_size >>> 
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


int stencil_81pt_double(int argc, char** argv) {
	DISPLAY_FUNCTION("");
	printf("datatype=double\n");
	int size = 8192; if (argc > 1) size = atoi(argv[1]);
	const int P = 4;
	const int B = 128;
	stencil2d_81::Test<double, P, B>(size, size);
	return 0;
}

int stencil_81pt_float(int argc, char** argv) {
	DISPLAY_FUNCTION("");
	printf("datatype=float\n");
	int size = 8192; if (argc > 1) size = atoi(argv[1]);
	const int P = 4;
	const int B = 128;
	stencil2d_81::Test<float, P, B>(size, size);
	return 0;
}
