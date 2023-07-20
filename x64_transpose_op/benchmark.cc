#include <iostream>
#include <vector>
#include <mm_malloc.h>
#include <ctime>
#include <assert.h>
#include <iomanip>

// #include "mlas.h"       // TODO
#include "mytranspose.h"

const std::vector<int> N({1023});    // 自定义测试序列
// const std::vector<int> N({1023});
const int WARMUP = 50;  // warmup round
const int TROUND = 50;  // test round

void TestFloat(int mat_size, float* x, float* y, void (*func) (const float* input, float* output, size_t M, size_t N))
{
    cout << "test_size: " << mat_size << "x" << mat_size << endl;
    for(int i = 0; i < WARMUP; ++i)
        func(x, y, mat_size, mat_size);
    clock_t s = clock();
    for(int i = 0; i < TROUND; ++i)
        func(x, y, mat_size, mat_size);
    cout << float(clock() - s) / CLOCKS_PER_SEC * 1000 << " ms" << endl;
}

int Check(int size, float* x, float* y)
{
    size_t sum = 0;
    size *= size;
    for(int i = 0; i < size; ++i)
        sum += (x[i] - y[i]);

    if(sum == 0)
        return 0;   // OK
    else
        return -1;
}

void PrintArr(int size, float* x)
{
    int t = size * size;
    for(int i = 0; i < t; ++i)
    {
        if(i % size == 0)
            cout<< endl;
        cout << std::setw(5) << x[i] << " ";
    }
    cout<< endl;
}

int main()
{
    for(int i = 0; i < N.size(); ++i)
    {
        int n = N[i];
        float* x = (float*)_mm_malloc(sizeof(float) * n * n, 32);
        float* y = (float*)_mm_malloc(sizeof(float) * n * n, 32);
        float* check = (float*)_mm_malloc(sizeof(float) * n * n, 32);
        for(int i=0; i<n*n; ++i)    x[i]=i % 1234567;
        for(int i=0; i<n*n; ++i)    y[i]=i % 1234567;

        cout << "-------------------test_base-------------------" << endl;
        TestFloat(n, x, check, Transpose_v1_base);     // test_v1
        // PrintArr(n, check);

        cout << "-------------------test_cache-------------------" << endl;
        TestFloat(n, x, y, Transpose_v2_cache);
        // PrintArr(n, y);
        assert(Check(n, y, check) == 0);

        cout << "-------------------test_simd-------------------" << endl;
        TestFloat(n, x, y, Transpose_v3_simd);
        // PrintArr(n, y);
        assert(Check(n, y, check) == 0);

        // cout << "-------------------test-------------------" << endl;
        // PrintArr(n, y);
        // TestFloat(n, x, y, Try_Float);
        // PrintArr(n, y);
        // // assert(Check(n, y, check) == 0);

        _mm_free(x);
        _mm_free(y);
        _mm_free(check);
    }

    return 0;
}
