__kernel void add(__global float *input,
        __global float *lastOuts,
        __global float *output){
            int gid = get_group_id(0);
            int thid = get_global_id(0);
            output[thid] = input[thid] + lastOuts[gid];
        }
