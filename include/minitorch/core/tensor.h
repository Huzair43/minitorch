#ifndef MINITORCH_TENSOR_H
#define MINITORCH_TENSOR_H

typedef struct {
    float* data;     
    int* shape;      
    int* strides;    
    int ndim;        
    int size;        
} Tensor;

/* Core tensor creation & management */
Tensor* tensor_create(const int* shape, int ndim);
Tensor* tensor_zeros(const int* shape, int ndim);
Tensor* tensor_ones(const int* shape, int ndim);
Tensor* tensor_clone(const Tensor* t);
void tensor_free(Tensor* t);

/* Indexing & access */
int tensor_get_index(const Tensor* t, const int* indices);
float tensor_get(const Tensor* t, const int* indices);
void tensor_set(Tensor* t, const int* indices, float value);

/* Reshaping */
Tensor* tensor_reshape(const Tensor* t, const int* new_shape, int new_ndim);

/* Debug */
void tensor_print(const Tensor* t);

#endif
