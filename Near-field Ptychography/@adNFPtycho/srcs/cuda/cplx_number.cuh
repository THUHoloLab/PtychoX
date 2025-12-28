#pragma once
#include <cuda_runtime.h>

#define PI 3.141592654f
#define TwoPI 6.28318531f

typedef struct complex32 {
    float re;
    float im;

    __device__ __forceinline__ complex32 operator+(const complex32 &b) const {return {re + b.re, im + b.im};}   
    __device__ __forceinline__ complex32 operator-(const complex32 &b) const {return {re - b.re, im - b.im};}
    __device__ __forceinline__ complex32 operator*(const complex32 &b) const {return {re * b.re - im * b.im, re * b.im + im * b.re};}
    __device__ __forceinline__ complex32 operator*(const float &b) const {return {re * b, im * b};}
    __device__ __forceinline__ complex32 operator*(const float2 &b) const {return {re * b.x - im * b.y, re * b.y + im * b.x};}
    __device__ __forceinline__ complex32 operator/(const float &b) const {
        float a = 1.0f / b;
        return {re * a, im * a};
    }
    __device__ __forceinline__ complex32 operator/(const complex32 &b) const {
        float denominator = 1.0f / (b.re * b.re + b.im * b.im);
        // if (denominator == 0){
        //     throw runtime_error("Error: Division by zero!");
        // }
        return{
            (re * b.re + im * b.im) * denominator,
            (im * b.re - re * b.im) * denominator
        };
    }
    __device__ __forceinline__ complex32 conj(void) const{return {re, -im};}
    __device__ __forceinline__ complex32 sign(void) const{float ang = atan2f(im, re); return {__cosf(ang), __sinf(ang)};}
    __device__ __forceinline__ complex32 inv(void) const{float temp = re * re + im * im; return {re / temp, -im / temp};}

    __device__ __forceinline__ float abs(void) const {return sqrtf(re * re + im * im);}
    __device__ __forceinline__ float angle(void) const {return atan2f(im, re);};

} cplx32_t;

__host__ __device__ cplx32_t make_complex(float a, float b){
    return {a, b};
}