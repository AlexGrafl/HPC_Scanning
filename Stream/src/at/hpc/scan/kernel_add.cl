__kernel void add(__global int *input,
        __global int *lastOuts,
        __global int *output){
            int gid = get_group_id(0);
            int thid = get_global_id(0);
            output[thid] = input[thid] + lastOuts[gid];
        }
