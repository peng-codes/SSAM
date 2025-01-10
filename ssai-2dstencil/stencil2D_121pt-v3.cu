#include "../ssai-2dconv/common.h"
#include "../ssai-2dconv/cudaLib.cuh"

#define    _F00         ((T)0.112)   //1
#define    _F01         ((T)0.224)   //2
#define    _F02         ((T)0.336)   //3
#define    _F03         ((T)0.448)   //4
#define    _F04         ((T)0.56)   //5
#define    _F05         ((T)0.672)   //6
#define    _F06         ((T)0.784)   //7
#define    _F07         ((T)0.896)   //8
#define    _F08         ((T)1.008)   //9
#define    _F09         ((T)1.12)   //10
#define    _F010         ((T)1.232)   //11
#define    _F10         ((T)1.344)   //12
#define    _F11         ((T)1.456)   //13
#define    _F12         ((T)1.568)   //14
#define    _F13         ((T)1.68)   //15
#define    _F14         ((T)1.792)   //16
#define    _F15         ((T)1.904)   //17
#define    _F16         ((T)2.016)   //18
#define    _F17         ((T)2.128)   //19
#define    _F18         ((T)2.24)   //20
#define    _F19         ((T)2.352)   //21
#define    _F110         ((T)2.464)   //22
#define    _F20         ((T)2.576)   //23
#define    _F21         ((T)2.688)   //24
#define    _F22         ((T)2.8)   //25
#define    _F23         ((T)2.912)   //26
#define    _F24         ((T)3.024)   //27
#define    _F25         ((T)3.136)   //28
#define    _F26         ((T)3.248)   //29
#define    _F27         ((T)3.36)   //30
#define    _F28         ((T)3.472)   //31
#define    _F29         ((T)3.584)   //32
#define    _F210         ((T)3.696)   //33
#define    _F30         ((T)3.808)   //34
#define    _F31         ((T)3.92)   //35
#define    _F32         ((T)4.032)   //36
#define    _F33         ((T)4.144)   //37
#define    _F34         ((T)4.256)   //38
#define    _F35         ((T)4.368)   //39
#define    _F36         ((T)4.48)   //40
#define    _F37         ((T)4.592)   //41
#define    _F38         ((T)4.704)   //42
#define    _F39         ((T)4.816)   //43
#define    _F310         ((T)4.928)   //44
#define    _F40         ((T)5.04)   //45
#define    _F41         ((T)5.152)   //46
#define    _F42         ((T)5.264)   //47
#define    _F43         ((T)5.376)   //48
#define    _F44         ((T)5.488)   //49
#define    _F45         ((T)5.6)   //50
#define    _F46         ((T)5.712)   //51
#define    _F47         ((T)5.824)   //52
#define    _F48         ((T)5.936)   //53
#define    _F49         ((T)6.048)   //54
#define    _F410         ((T)6.16)   //55
#define    _F50         ((T)6.272)   //56
#define    _F51         ((T)6.384)   //57
#define    _F52         ((T)6.496)   //58
#define    _F53         ((T)6.608)   //59
#define    _F54         ((T)6.72)   //60
#define    _F55         ((T)6.832)   //61
#define    _F56         ((T)6.944)   //62
#define    _F57         ((T)7.056)   //63
#define    _F58         ((T)7.168)   //64
#define    _F59         ((T)7.28)   //65
#define    _F510         ((T)7.392)   //66
#define    _F60         ((T)7.504)   //67
#define    _F61         ((T)7.616)   //68
#define    _F62         ((T)7.728)   //69
#define    _F63         ((T)7.84)   //70
#define    _F64         ((T)7.952)   //71
#define    _F65         ((T)8.064)   //72
#define    _F66         ((T)8.176)   //73
#define    _F67         ((T)8.288)   //74
#define    _F68         ((T)8.4)   //75
#define    _F69         ((T)8.512)   //76
#define    _F610         ((T)8.624)   //77
#define    _F70         ((T)8.736)   //78
#define    _F71         ((T)8.848)   //79
#define    _F72         ((T)8.96)   //80
#define    _F73         ((T)9.072)   //81
#define    _F74         ((T)9.184)   //82
#define    _F75         ((T)9.296)   //83
#define    _F76         ((T)9.408)   //84
#define    _F77         ((T)9.52)   //85
#define    _F78         ((T)9.632)   //86
#define    _F79         ((T)9.744)   //87
#define    _F710         ((T)9.856)   //88
#define    _F80         ((T)9.968)   //89
#define    _F81         ((T)10.08)   //90
#define    _F82         ((T)10.192)   //91
#define    _F83         ((T)10.304)   //92
#define    _F84         ((T)10.416)   //93
#define    _F85         ((T)10.528)   //94
#define    _F86         ((T)10.64)   //95
#define    _F87         ((T)10.752)   //96
#define    _F88         ((T)10.864)   //97
#define    _F89         ((T)10.976)   //98
#define    _F810         ((T)11.088)   //99
#define    _F90         ((T)11.2)   //100
#define    _F91         ((T)11.312)   //101
#define    _F92         ((T)11.424)   //102
#define    _F93         ((T)11.536)   //103
#define    _F94         ((T)11.648)   //104
#define    _F95         ((T)11.76)   //105
#define    _F96         ((T)11.872)   //106
#define    _F97         ((T)11.984)   //107
#define    _F98         ((T)12.096)   //108
#define    _F99         ((T)12.208)   //109
#define    _F910         ((T)12.32)   //110
#define    _F100         ((T)12.432)   //111
#define    _F101         ((T)12.544)   //112
#define    _F102         ((T)12.656)   //113
#define    _F103         ((T)12.768)   //114
#define    _F104         ((T)12.88)   //115
#define    _F105         ((T)12.992)   //116
#define    _F106         ((T)13.104)   //117
#define    _F107         ((T)13.216)   //118
#define    _F108         ((T)13.328)   //119
#define    _F109         ((T)13.44)   //120
#define    _F1010         ((T)13.552)   //121

namespace stencil2d_121pt_v3 {
	static const int WARP_SIZE = 32;
	static const int FILTER_WIDTH = 11;
	static const int FILTER_HEIGHT = 11;

	template<typename T, int BLOCK_SIZE, int PROCESS_DATA_COUNT>
	__global__ void j2d121pt(const T* __restrict__ src, T* dst, int width, int height)
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
			
			if (laneId > FILTER_WIDTH - 2) {
				sum += data[i + 0] * _F010;
				sum += data[i + 1] * _F110;
				sum += data[i + 2] * _F210;
				sum += data[i + 3] * _F310;
				sum += data[i + 4] * _F410;
				sum += data[i + 5] * _F510;
				sum += data[i + 6] * _F610;
				sum += data[i + 7] * _F710;
				sum += data[i + 8] * _F810;
				sum += data[i + 9] * _F910;
				sum += data[i + 10] * _F1010;
			}

			sum = __my_shfl_down(sum, 1);
			if (laneId > FILTER_WIDTH - 2 - 1 && laneId < WARP_SIZE) {
				sum += data[i + 0] * _F09;
				sum += data[i + 1] * _F19;
				sum += data[i + 2] * _F29;
				sum += data[i + 3] * _F39;
				sum += data[i + 4] * _F49;
				sum += data[i + 5] * _F59;
				sum += data[i + 6] * _F69;
				sum += data[i + 7] * _F79;
				sum += data[i + 8] * _F89;
				sum += data[i + 9] * _F99;
				sum += data[i + 10] * _F109;
			}

			sum = __my_shfl_down(sum, 1);
			if (laneId > FILTER_WIDTH - 2 - 2 && laneId < WARP_SIZE - 1) {
				sum += data[i + 0] * _F08;
				sum += data[i + 1] * _F18;
				sum += data[i + 2] * _F28;
				sum += data[i + 3] * _F38;
				sum += data[i + 4] * _F48;
				sum += data[i + 5] * _F58;
				sum += data[i + 6] * _F68;
				sum += data[i + 7] * _F78;
				sum += data[i + 8] * _F88;
				sum += data[i + 9] * _F98;
				sum += data[i + 10] * _F108;
			}

			sum = __my_shfl_down(sum, 1);
			if (laneId > FILTER_WIDTH - 2 - 3 && laneId < WARP_SIZE - 2) {
				sum += data[i + 0] * _F07;
				sum += data[i + 1] * _F17;
				sum += data[i + 2] * _F27;
				sum += data[i + 3] * _F37;
				sum += data[i + 4] * _F47;
				sum += data[i + 5] * _F57;
				sum += data[i + 6] * _F67;
				sum += data[i + 7] * _F77;
				sum += data[i + 8] * _F87;
				sum += data[i + 9] * _F97;
				sum += data[i + 10] * _F107;
			}

			sum = __my_shfl_down(sum, 1);
			if (laneId > FILTER_WIDTH - 2 - 4 && laneId < WARP_SIZE - 3) {
				sum += data[i + 0] * _F06;
				sum += data[i + 1] * _F16;
				sum += data[i + 2] * _F26;
				sum += data[i + 3] * _F36;
				sum += data[i + 4] * _F46;
				sum += data[i + 5] * _F56;
				sum += data[i + 6] * _F66;
				sum += data[i + 7] * _F76;
				sum += data[i + 8] * _F86;
				sum += data[i + 9] * _F96;
				sum += data[i + 10] * _F106;
			}

			sum = __my_shfl_down(sum, 1);
			if (laneId > FILTER_WIDTH - 2 - 5 && laneId < WARP_SIZE - 4) {
				sum += data[i + 0] * _F05;
				sum += data[i + 1] * _F15;
				sum += data[i + 2] * _F25;
				sum += data[i + 3] * _F35;
				sum += data[i + 4] * _F45;
				sum += data[i + 5] * _F55;
				sum += data[i + 6] * _F65;
				sum += data[i + 7] * _F75;
				sum += data[i + 8] * _F85;
				sum += data[i + 9] * _F95;
				sum += data[i + 10] * _F105;
			}

			sum = __my_shfl_down(sum, 1);
			if (laneId > FILTER_WIDTH - 2 - 6 && laneId < WARP_SIZE - 5) {
				sum += data[i + 0] * _F04;
				sum += data[i + 1] * _F14;
				sum += data[i + 2] * _F24;
				sum += data[i + 3] * _F34;
				sum += data[i + 4] * _F44;
				sum += data[i + 5] * _F54;
				sum += data[i + 6] * _F64;
				sum += data[i + 7] * _F74;
				sum += data[i + 8] * _F84;
				sum += data[i + 9] * _F94;
				sum += data[i + 10] * _F104;
			}

			sum = __my_shfl_down(sum, 1);
			if (laneId > FILTER_WIDTH - 2 - 7 && laneId < WARP_SIZE - 6) {
				sum += data[i + 0] * _F03;
				sum += data[i + 1] * _F13;
				sum += data[i + 2] * _F23;
				sum += data[i + 3] * _F33;
				sum += data[i + 4] * _F43;
				sum += data[i + 5] * _F53;
				sum += data[i + 6] * _F63;
				sum += data[i + 7] * _F73;
				sum += data[i + 8] * _F83;
				sum += data[i + 9] * _F93;
				sum += data[i + 10] * _F103;
			}

			sum = __my_shfl_down(sum, 1);
			if (laneId > FILTER_WIDTH - 2 - 8 && laneId < WARP_SIZE - 7) {
				sum += data[i + 0] * _F02;
				sum += data[i + 1] * _F12;
				sum += data[i + 2] * _F22;
				sum += data[i + 3] * _F32;
				sum += data[i + 4] * _F42;
				sum += data[i + 5] * _F52;
				sum += data[i + 6] * _F62;
				sum += data[i + 7] * _F72;
				sum += data[i + 8] * _F82;
				sum += data[i + 9] * _F92;
				sum += data[i + 10] * _F102;
			}

			sum = __my_shfl_down(sum, 1);
			if (laneId > FILTER_WIDTH - 2 - 9 && laneId < WARP_SIZE - 8) {
				sum += data[i + 0] * _F01;
				sum += data[i + 1] * _F11;
				sum += data[i + 2] * _F21;
				sum += data[i + 3] * _F31;
				sum += data[i + 4] * _F41;
				sum += data[i + 5] * _F51;
				sum += data[i + 6] * _F61;
				sum += data[i + 7] * _F71;
				sum += data[i + 8] * _F81;
				sum += data[i + 9] * _F91;
				sum += data[i + 10] * _F101;
			}

			sum = __my_shfl_down(sum, 1);
			if (laneId < WARP_SIZE - 9) {
				sum += data[i + 0] * _F00;
				sum += data[i + 1] * _F10;
				sum += data[i + 2] * _F20;
				sum += data[i + 3] * _F30;
				sum += data[i + 4] * _F40;
				sum += data[i + 5] * _F50;
				sum += data[i + 6] * _F60;
				sum += data[i + 7] * _F70;
				sum += data[i + 8] * _F80;
				sum += data[i + 9] * _F90;
				sum += data[i + 10] * _F100;
			}

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
			{ 0.112, 0.224, 0.336, 0.448, 0.56, 0.672, 0.784, 0.896, 1.008, 1.12, 1.232, },
			{ 1.344, 1.456, 1.568, 1.68, 1.792, 1.904, 2.016, 2.128, 2.24, 2.352, 2.464, },
			{ 2.576, 2.688, 2.8, 2.912, 3.024, 3.136, 3.248, 3.36, 3.472, 3.584, 3.696, },
			{ 3.808, 3.92, 4.032, 4.144, 4.256, 4.368, 4.48, 4.592, 4.704, 4.816, 4.928, },
			{ 5.04, 5.152, 5.264, 5.376, 5.488, 5.6, 5.712, 5.824, 5.936, 6.048, 6.16, },
			{ 6.272, 6.384, 6.496, 6.608, 6.72, 6.832, 6.944, 7.056, 7.168, 7.28, 7.392, },
			{ 7.504, 7.616, 7.728, 7.84, 7.952, 8.064, 8.176, 8.288, 8.4, 8.512, 8.624, },
			{ 8.736, 8.848, 8.96, 9.072, 9.184, 9.296, 9.408, 9.52, 9.632, 9.744, 9.856, },
			{ 9.968, 10.08, 10.192, 10.304, 10.416, 10.528, 10.64, 10.752, 10.864, 10.976, 11.088, },
			{ 11.2, 11.312, 11.424, 11.536, 11.648, 11.76, 11.872, 11.984, 12.096, 12.208, 12.32, },
			{ 12.432, 12.544, 12.656, 12.768, 12.88, 12.992, 13.104, 13.216, 13.328, 13.44, 13.552, },
		};

		cudaEventRecord(start, 0);
		for (int s = 0; s < nRepeatCount; s++) {
			j2d121pt<DataType, BLOCK_SIZE, PROCESS_DATA_COUNT> << <grid_size, block_size >> >
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
			if (x > FILTER_WIDTH / 2 && x < width - FILTER_WIDTH / 2 && y > FILTER_HEIGHT / 2 && y < height - FILTER_HEIGHT / 2)
				dif += abs(imgVerify.data[i] - imgDst.data[i]);
		}
		printf("verify dif =%f, avg-dif=%e\n", dif, dif / img.width / img.height);
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

int stencil_121pt_v3(int argc, char** argv) {
	DISPLAY_FUNCTION("");
	printf("datatype=double\n");
	int size = 8192; if (argc > 1) size = atoi(argv[1]);
	const int P = 4;
	const int B = 128;
	stencil2d_121pt_v3::Test<double, P, B>(size, size);
	return 0;
}


