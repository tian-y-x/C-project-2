#pragma once
typedef struct conv_param
{
    int pad;
    int stride;
    int kernel_size;
    int in_channels;
    int out_channels;
    float* p_weight;
    float* p_bias;
} conv_param;

typedef struct fc_param {
    int in_features;
    int out_features;
    float* p_weight;
    float* p_bias;
} fc_param;

extern float firstPool[];
extern float secondPool[];

extern const int FIR_O_SIZE;
extern float first_out[];

extern const int SEC_O_SIZE;
extern float second_out[];

extern const int THID_O_SIZE;
extern float third_out[];


void First_conv(const int size);
void Second_conv(const int size);
void Third_conv(const int size);

void ReLU(float* M, const int size);
void MaxPool(float* M_in, float* M_out, const int size, const int num);
void AvgPool(float* M_in, float* M_out, const int size, const int num);

void Gemm();
void softmax();
