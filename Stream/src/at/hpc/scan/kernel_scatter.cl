 __kernel void scatter(__global float* input,
        __global int* newIndices,
        int length,
        __global float *output) {
    int thid = get_global_id(0);
    int newIndex = newIndices[thid];
    if(newIndex < length){
        output[newIndex] = input[thid]; // write output
    }
 }
