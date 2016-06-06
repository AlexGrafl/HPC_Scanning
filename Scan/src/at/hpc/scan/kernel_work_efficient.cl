__kernel void scan(__global float *g_idata,
    __global float *g_odata,
    __local float *temp,
    int lastOutsEnabled,
    __global float *lastOuts) {
	int global_thid = get_global_id(0);
	int thid = get_local_id(0);
	int offset = 1;
	temp[2 * thid] = g_idata[2 * global_thid]; // load input into shared memory
	temp[2 * thid + 1] = g_idata[2 * global_thid + 1];
	int x = 2 * get_local_size(0);
	for (int d = x >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		if (thid < d) {
			int ai = offset * (2 * thid + 1) - 1;
			int bi = offset * (2 * thid + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
    }
    if (thid == 0) {
        temp[x - 1] = 0;
    } // clear the last element
    for (int d = 1; d < x; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (thid < d) {

            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    g_odata[2 * global_thid] = temp[2 * thid]; // write results to device memory
    g_odata[2 * global_thid + 1] = temp[2 * thid + 1];
    if(lastOutsEnabled == 1){
        if(thid == get_local_size(0) - 1){
            int lastSumIndex = 2 * thid + 1;
            float lastSum = temp[lastSumIndex];
            int nextElementIndex = 2 * (get_group_id(0) + 1) * get_local_size(0) - 1;
            float nextElement = g_idata[nextElementIndex];
            float sum = lastSum + nextElement;
            int groupId = get_group_id(0);
//            printf("%d, %d %f, %d %f\n", groupId, lastSumIndex, lastSum, nextElementIndex, nextElement);
            lastOuts[groupId] = sum;
        }
    }
}