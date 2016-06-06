 __kernel void scan(__global float *g_idata,
        __global float *g_odata,
        int n,
        __local float *temp) {
    int thid = get_global_id(0);
    int pout = 0, pin = 1;
    // Load input into shared memory.
    // This is exclusive scan, so shift right by one
    // and set first element to 0
    temp[pout * n + thid] = (thid > 0) ? g_idata[thid - 1];
    for (int offset = 1; offset < n; offset *= 2) {
        pout = 1 - pout; // swap double buffer indices
        pin = 1 - pin;
        if (thid >= offset) {
            temp[pout * n + thid] = temp[pin * n + thid] + temp[pin * n + thid - offset];
        }
        else {
            temp[pout * n + thid] = temp[pin * n + thid];
        }
    }
    g_odata[thid] = temp[pout * n + thid]; // write output
 }
