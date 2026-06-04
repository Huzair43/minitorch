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
int tensor_numel(const int* shape, int ndim);
int tensor_same_shape(const Tensor* a, const Tensor* b);
Tensor* tensor_create(const int* shape, int ndim);
Tensor* tensor_zeros(const int* shape, int ndim);
Tensor* tensor_ones(const int* shape, int ndim);
Tensor* tensor_clone(const Tensor* t);
Tensor* tensor_broadcast_to(const Tensor* t, const int* target_shape, int target_ndim);
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
