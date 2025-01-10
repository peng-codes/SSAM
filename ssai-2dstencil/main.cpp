#include <stdio.h>
#include <string.h>

extern int stencil_5pt_v1(int argc, char** argv);
extern int stencil_5pt_v2(int argc, char** argv);
extern int stencil_9pt(int argc, char** argv);
extern int stencil_13pt(int argc, char** argv);
extern int stencil_17pt(int argc, char** argv);
extern int stencil_21pt(int argc, char** argv);
extern int stencil_25pt(int argc, char** argv);
extern int stencil_64pt(int argc, char** argv);
extern int stencil_81pt(int argc, char** argv);
extern int stencil_121pt(int argc, char** argv);
extern int stencil_tmp_5pt(int argc, char** argv);
extern int stencil_tmp_9pt(int argc, char** argv);
extern int stencil_121pt_v2(int argc, char** argv);
extern int stencil_121pt_v3(int argc, char** argv);
extern int stencil_121pt_v4(int argc, char** argv);

int main(int argc, char** argv)
{
	for (int i = 0; i < argc; i++){
		printf("%s ", argv[i]);
		if (i == argc - 1) printf("\n");
	}
	const char* pts  = argv[2];
	const char* dtype = argv[3];
	
#define CALL_FUNC(num)\
	if (strcmp(pts, #num) == 0) {\
		if (strcmp(dtype, "double") == 0) {\
			extern int stencil_##num##pt_double(int argc, char** argv);\
			return stencil_##num##pt_double(argc, argv);\
		}\
		else if (strcmp(dtype, "float") == 0) {\
			extern int stencil_##num##pt_float(int argc, char** argv);\
			return stencil_##num##pt_float(argc, argv);\
		}else {\
			printf("error!\n");\
		}\
	}
	
	CALL_FUNC(5);
	CALL_FUNC(9);
	CALL_FUNC(13);
	CALL_FUNC(17);
	CALL_FUNC(21);
	CALL_FUNC(25);	
	CALL_FUNC(64);
	CALL_FUNC(81);
	CALL_FUNC(121);
	CALL_FUNC(tmp_5);
	CALL_FUNC(tmp_9);
	printf("error, do not call functions\n");
	return 0;
}

//{
	//	if (strcmp(pts, "5pt")) {
//		if (strcmp(dtype, "double") == 0) {
//			extern int stencil_5pt_double(int argc, char** argv);
//			return stencil_5pt_double(argc, argv);
//		}
//		else if (strcmp(dtype, "float") == 0) {
//			extern int stencil_5pt_float(int argc, char** argv);
//			return stencil_5pt_float(argc, argv);
//		}
//		else {
//			printf("error!\n");
//		}		
//	}
//}
