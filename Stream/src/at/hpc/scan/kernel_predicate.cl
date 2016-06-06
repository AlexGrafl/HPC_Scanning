__kernel void apply(__global float *input,
        __global int *output){
            int thid = get_global_id(0);
            output[thid] = (fmod(input[thid], 2)) == 0 ? 1 : 0;
        }
