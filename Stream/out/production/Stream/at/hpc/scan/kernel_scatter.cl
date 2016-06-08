 __kernel void scatter(__global float* input,
        __global int* newIndices,
        int length,
        __global float *output) {
    int thid = get_global_id(0);
    if(thid != 0){
        int newIndex = newIndices[thid];
            printf("thid: %d, New: %d, %f\n", thid, newIndex, input[thid - 1]);
        if(newIndex != newIndices[thid - 1]){
            output[newIndex - 1] = input[thid - 1]; // write output
        }
    }
 }
