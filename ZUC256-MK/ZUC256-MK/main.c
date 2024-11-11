#include"ZUC256-MK.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#define single_thread 1
#define PARALLEL_NUM 16
#define	MAXBYTELEN 32768
#define TOTXTFILE single_thread
double totalspeed = 0;


void ZUC256_SpeedTest_AVX512()
{
	totalspeed = 0;
	FILE* fp = NULL, * fpscreen = stdout;
	double start, finish, totaltime, speed, evenspeed[3] = { 0 };
	int TIMES, i, j, mode, fpfile, NUMBERS;
	u32 msg[MAXBYTELEN * PARALLEL_NUM / 4];
	u8* k = (u8*)msg;
	u8* iv = (u8*)(msg + 8);
	u32 wordlen, NNN;

	NUMBERS = 10;
	for (j = 0; j < NUMBERS; j++)
	{
		memset(msg, 0xab, MAXBYTELEN * PARALLEL_NUM);
		for (mode = 2; mode < 3; mode++)
		{
			switch (mode)
			{
			case 0: NNN = 20;	wordlen = 256 / sizeof(u32);	break; // short message
			case 1: NNN = 19;	wordlen = 2048 / sizeof(u32);	break; // middle message
			case 2: NNN = 16;	wordlen = 9000 / sizeof(u32);	break; // long message
			}
			TIMES = 1 << NNN;
			start = clock();
			for (i = 0; i < TIMES; i++)
			{
				//ZUC256_CRYPT_AVX512(msg, k, iv, msg, wordlen);
				ZUC256_AVX512(msg, wordlen, k, iv);
			}
			finish = clock();
			totaltime = (finish - start) / CLOCKS_PER_SEC;
			speed = (((double)wordlen * 32 * PARALLEL_NUM) / 1000.0) * ((double)TIMES / 1000.0) / totaltime;
			evenspeed[mode] += speed;
			fpfile = 0;
#if(TOTXTFILE==1)
			for (; fpfile < 1; fpfile++)
#endif
			{
				switch (fpfile)
				{
				case 0: fp = fpscreen; break;
				}
				fprintf(fp, "\n\n ZUC256_AVX512_LINE_%d : one line generate %d bytes keystream", PARALLEL_NUM, wordlen * (int)sizeof(u32));
				fprintf(fp, "\n Test Function\t ZUC256_CRYPT_AVX512");
				fprintf(fp, "\n Test Times\t 1<<%d", NNN);
				fprintf(fp, "\n Test Time\t %.3f seconds", totaltime);
				fprintf(fp, "\n Test Speed\t %.2f Gbps", speed / 1000.0);
			}
		}
		fprintf(fpscreen, "\n\n\n");
	}
	printf("\n\n test times = %d, max even speed is %.2f Gbps\n\n", NUMBERS, evenspeed[2] / NUMBERS / 1000.0);
	totalspeed += evenspeed[2] / NUMBERS;
}



void ZUC256_MAC_SpeedTest_AVX512()
{
#define MAC_BITLENGTH 32
	totalspeed = 0;
	FILE* fp = NULL, * fpscreen = stdout;
	double start, finish, totaltime, speed, evenspeed[3] = { 0 };
	int TIMES, i, j, mode, fpfile, NUMBERS;
	u32 msg[MAXBYTELEN * PARALLEL_NUM / 4];
	u8* k = (u8*)msg;
	u8* iv = (u8*)(msg + 8);
	u32 wordlen, NNN, msg_len;

	NUMBERS = 10;
	for (j = 0; j < NUMBERS; j++)
	{
		memset(msg, 0xab, MAXBYTELEN * PARALLEL_NUM);
		for (mode = 2; mode < 3; mode++)
		{
			switch (mode)
			{
			case 0: NNN = 20;	wordlen = 256 / sizeof(u32);	break; // short message
			case 1: NNN = 19;	wordlen = 2048 / sizeof(u32);	break; // middle message
			case 2: NNN = 16;	wordlen = 9000 / sizeof(u32);	break; // long message
			}
			TIMES = 1 << NNN;
			msg_len = (wordlen - 2);
			start = clock();
			for (i = 0; i < TIMES; i++)
			{
				ZUC256_MAC_AVX512(msg, MAC_BITLENGTH, k, iv, msg, msg_len);
			}
			finish = clock();
			totaltime = (finish - start) / CLOCKS_PER_SEC;
			speed = (((double)wordlen * 32 * PARALLEL_NUM) / 1000.0) * ((double)TIMES / 1000.0) / totaltime;
			evenspeed[mode] += speed;
			fpfile = 0;
#if(TOTXTFILE==1)
			for (; fpfile < 1; fpfile++)
#endif
			{
				switch (fpfile)
				{
				case 0: fp = fpscreen; break;
				}
				fprintf(fp, "\n\n ZUC256_MAC%d_AVX512_LINE_%d : one line with %d bytes massage generate %d bits MAC",
					MAC_BITLENGTH, PARALLEL_NUM, msg_len * 4, MAC_BITLENGTH);
				fprintf(fp, "\n Test Function\t ZUC256_MAC%d_AVX512", MAC_BITLENGTH);
				fprintf(fp, "\n Test Times\t 1<<%d", NNN);
				fprintf(fp, "\n Test Time\t %.3f seconds", totaltime);
				fprintf(fp, "\n Test Speed\t %.2f Gbps", speed / 1000.0);
			}
		}
		fprintf(fpscreen, "\n\n\n");
	}
	printf("\n\n test times = %d, max even speed is %.2f Gbps\n\n", NUMBERS, evenspeed[2] / NUMBERS / 1000.0);
	totalspeed += evenspeed[2] / NUMBERS;
}

#if single_thread
void main()
{
	//ZUC256_SelfTest_AVX512();
	//ZUC256_SpeedTest_AVX512();
	//ZUC256_MAC_SelfTest_AVX512();
	ZUC256_MAC_SpeedTest_AVX512();
	system("pause");
}
#endif


#if !single_thread
#include <Windows.h>
#include <process.h>

void main()
{
	int i, NUM = 1;
	HANDLE thread[32];
	unsigned int ID[32];
	totalspeed = 0;

	while (1)
	{
		printf("\n input process number : ");
		scanf_s("%d", &NUM);
		if (NUM < 1 || NUM>12)
			printf("\nNUM is error\n");
		else
			break;
	}

	for (i = 0; i < NUM; i++)
	{
		//printf("\n press any key to continue \n");
		//getchar();
		thread[i] = (HANDLE)_beginthreadex(NULL, 0, ZUC256_SpeedTest_AVX512, (LPVOID)NULL, 0, ID + i);
		//thread[i] = (HANDLE)_beginthreadex(NULL, 0, ZUC256_MAC_SpeedTest_AVX512, (LPVOID)NULL, 0, ID + i);
	}
	for (i = 0; i < NUM; i++)
	{
		WaitForSingleObject(thread[i], INFINITE);
	}
	for (i = 0; i < NUM; i++)
	{
		CloseHandle(thread[i]);
	}

	printf("\n\n total speed is %.2f Gbps\n\n", totalspeed / 1000.0);
	printf(" press any key to end \n");
	getchar(); getchar();
}
#endif
