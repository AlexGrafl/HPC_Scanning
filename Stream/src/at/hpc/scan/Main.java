package at.hpc.scan;

import org.jocl.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.sql.Time;
import java.util.concurrent.TimeUnit;
import java.util.Arrays;

import static org.jocl.CL.*;

public class Main
{

    private static final int WORK_GROUP_COUNT = 128;
    private static final int DATA_SIZE = 2048;

    public static void main(String args[]){

        String data = readFile("data.txt");
        String[] split = data.split(",");
        float[] inputArray = new float[split.length / DATA_SIZE];
        for (int i = 0; i < split.length / DATA_SIZE; i++) {
            inputArray[i] = Float.valueOf(split[i]);
        }
        System.out.println("Array lenght power of two: \t\t" + ((inputArray.length & (inputArray.length - 1)) == 0 ? true : false));
        if (((inputArray.length & (inputArray.length - 1)) == 0) == false){
            return;
        }

        System.out.println("WORK_GROUP_COUNT power of two: \t\t" + ((WORK_GROUP_COUNT & (WORK_GROUP_COUNT - 1)) == 0 ? true : false));
        if (((WORK_GROUP_COUNT & (WORK_GROUP_COUNT - 1)) == 0) == false){
            return;
        }
//        int[] doubleArray = new Random().ints((long) Math.pow(2, 16), 0, 1000).toArray();
//        float[] inputArray = new float[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
//        System.out.print("{");
//        for (int i = 0 ; i < doubleArray.length; i++) {
//            inputArray[i] = (float) doubleArray[i];
//            System.out.print(inputArray[i] + ",");
//        }
//        System.out.print("}");
        int[] predicatedArray = new int[inputArray.length];
        int[] scannedPredicates = new int[inputArray.length];
        int[] lastOutsArray = new int[WORK_GROUP_COUNT];
        int[] scannedLastOutsArray = new int[WORK_GROUP_COUNT];

//        String programSource = readFile("src/at/hpc/scan/kernel_scatter.cl");
        String programSource = readFile("src/at/hpc/scan/kernel_work_efficient.cl");
        String predicateProgramSource = readFile("src/at/hpc/scan/kernel_predicate.cl");
        String addProgramSource = readFile("src/at/hpc/scan/kernel_add.cl");
        String scatterProgramSource = readFile("src/at/hpc/scan/kernel_scatter.cl");

        // The platform, device type and device number
        // that will be used
        final int platformIndex = 0;
        final long deviceType = CL_DEVICE_TYPE_ALL;
        final int deviceIndex = 0;

        // Enable exceptions and subsequently omit error checks in this sample
        setExceptionsEnabled(true);

        // Obtain the number of platforms
        int numPlatformsArray[] = new int[1];
        clGetPlatformIDs(0, null, numPlatformsArray);
        int numPlatforms = numPlatformsArray[0];

        // Obtain a platform ID
        cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
        clGetPlatformIDs(platforms.length, platforms, null);
        cl_platform_id platform = platforms[platformIndex];

        // Initialize the context properties
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);

        // Obtain the number of devices for the platform
        int numDevicesArray[] = new int[1];
        clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
        int numDevices = numDevicesArray[0];

        // Obtain a device ID 
        cl_device_id devices[] = new cl_device_id[numDevices];
        clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
        cl_device_id device = devices[deviceIndex];

        // Create a context for the selected device
        cl_context context = clCreateContext(
                contextProperties, 1, new cl_device_id[]{device},
                null, null, null);

        cl_command_queue commandQueue =
                clCreateCommandQueue(context, device, 0, null);

        System.out.println("Input: \t" + Arrays.toString(inputArray));

        cl_program predicateProgram = clCreateProgramWithSource(context, 1, new String[]{predicateProgramSource}, null, null);
        clBuildProgram(predicateProgram, 0, null, null, null, null);
        cl_kernel predicateKernel = clCreateKernel(predicateProgram, "apply", null);
        applyPredicate(inputArray, predicatedArray, context, commandQueue, predicateKernel);
        clReleaseKernel(predicateKernel);
        clReleaseProgram(predicateProgram);

        System.out.println("Predicated: \t\t" + Arrays.toString(predicatedArray));

        cl_program program = clCreateProgramWithSource(context,
                1, new String[]{ programSource }, null, null);
        clBuildProgram(program, 0, null, null, null, null);
        cl_kernel kernel = clCreateKernel(program, "scan", null);
        long time1 = scanInput(predicatedArray, scannedPredicates, lastOutsArray, context, commandQueue, kernel);
        long time2 = scanInput(lastOutsArray, scannedLastOutsArray, null, context, commandQueue, kernel);
        clReleaseKernel(kernel);
        clReleaseProgram(program);

        cl_program addProgram = clCreateProgramWithSource(context, 1, new String[]{addProgramSource}, null, null);
        clBuildProgram(addProgram, 0, null, null, null, null);
        cl_kernel addKernel = clCreateKernel(addProgram, "add", null);
        long time3 = addLastOutsToOut(scannedLastOutsArray, scannedPredicates, context, commandQueue, addKernel);
//        System.out.println("Output2: \t" + Arrays.toString(outputArray));
        clReleaseKernel(addKernel);
        clReleaseProgram(addProgram);

        System.out.println("Scanned: \t\t" + Arrays.toString(scannedPredicates));

        float[] scatteredArray = new float[scannedPredicates[scannedPredicates.length - 1]];
        cl_program scatterProgram = clCreateProgramWithSource(context, 1, new String[]{scatterProgramSource}, null, null);
        clBuildProgram(scatterProgram, 0, null, null, null, null);
        cl_kernel scatterKernel = clCreateKernel(scatterProgram, "scatter", null);

        long time4 = applyScatter(inputArray, scannedLastOutsArray, scatteredArray, context, commandQueue, scatterKernel);

        System.out.println("Scattered: \t\t" + Arrays.toString(scatteredArray));
        System.out.println("GPU: " + TimeUnit.NANOSECONDS.toMillis(time1 + time2 + time3 + time4));

        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);
    }

    private static long addLastOutsToOut(int[] lastOutsScannedArray, int[] outputArray, cl_context context, cl_command_queue commandQueue, cl_kernel kernel) {
        Pointer lastOutsPointer = Pointer.to(lastOutsScannedArray);
        Pointer inPointer = Pointer.to(outputArray);
        int[] tempOutArray = new int[outputArray.length];
        Pointer outPointer = Pointer.to(tempOutArray);
        cl_mem inBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, outputArray.length * Sizeof.cl_int, inPointer, null);
        cl_mem lastOutsBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, lastOutsScannedArray.length * Sizeof.cl_int, lastOutsPointer, null);
        cl_mem outBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, tempOutArray.length * Sizeof.cl_int, null, null);
        clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(inBuffer));
        clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(lastOutsBuffer));
        clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(outBuffer));
        long global_work_size[] = new long[]{outputArray.length};
        long local_work_size[] = new long[]{ outputArray.length / lastOutsScannedArray.length };
        long start = System.nanoTime();
        clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, global_work_size, local_work_size, 0, null, null);
        clEnqueueReadBuffer(commandQueue, outBuffer, true, 0, outputArray.length * Sizeof.cl_int,
                outPointer, 0, null, null);
        clReleaseMemObject(inBuffer);
        clReleaseMemObject(outBuffer);
        clReleaseMemObject(lastOutsBuffer);
        System.arraycopy(tempOutArray, 0, outputArray, 0, tempOutArray.length);
        long end = System.nanoTime();

        return end - start;
    }

    private static void applyPredicate(float[] intput, int[] predicatedArray, cl_context context, cl_command_queue commandQueue, cl_kernel kernel) {
        Pointer inPointer = Pointer.to(intput);
        int[] tempOutArray = new int[predicatedArray.length];
        Pointer outPointer = Pointer.to(tempOutArray);
        cl_mem inBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, intput.length * Sizeof.cl_float, inPointer, null);
        cl_mem outBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, predicatedArray.length * Sizeof.cl_int, null, null);
        clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(inBuffer));
        clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(outBuffer));
        long global_work_size[] = new long[]{intput.length};
        long local_work_size[] = new long[]{ intput.length / WORK_GROUP_COUNT };
        clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, global_work_size, local_work_size, 0, null, null);
        clEnqueueReadBuffer(commandQueue, outBuffer, true, 0, predicatedArray.length * Sizeof.cl_int,
                outPointer, 0, null, null);
        clReleaseMemObject(inBuffer);
        clReleaseMemObject(outBuffer);
        System.arraycopy(tempOutArray, 0, predicatedArray, 0, tempOutArray.length);
    }

    private static long scanInput(int[] inArray, int[] outArray, int[] lastOutsArray, cl_context context, cl_command_queue commandQueue, cl_kernel kernel) {
        Pointer inPointer = Pointer.to(inArray);
        int[] tempOutArray = new int[outArray.length];
        int[] tempLastOutsArray = null;
        Pointer outPointer = Pointer.to(tempOutArray);
        Pointer lastOutsPointer = null;
        boolean lastOutsEnabled = lastOutsArray != null;
        if(lastOutsEnabled) {
            tempLastOutsArray = new int[lastOutsArray.length];
            lastOutsPointer = Pointer.to(tempLastOutsArray);
        }
        long localWorkSize = inArray.length / (WORK_GROUP_COUNT * 2);
        if(localWorkSize == 0) localWorkSize = inArray.length;

        cl_mem inBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inArray.length * Sizeof.cl_int, inPointer, null);
        cl_mem outBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, outArray.length * Sizeof.cl_int, null, null);
        cl_mem lastOutsBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, lastOutsEnabled ? lastOutsArray.length * Sizeof.cl_int : 1, null, null);
        clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(inBuffer));
        clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(outBuffer));
        clSetKernelArg(kernel, 2, Sizeof.cl_int * 2 * localWorkSize, null);
        clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{lastOutsEnabled ? 1 : 0}));
        clSetKernelArg(kernel, 4, Sizeof.cl_mem, Pointer.to(lastOutsBuffer));

        long global_work_size[] = new long[]{localWorkSize == inArray.length ? inArray.length : inArray.length / 2};
        long local_work_size[] = new long[]{ localWorkSize };

        long start = System.nanoTime();
        clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, global_work_size, local_work_size, 0, null, null);
        clEnqueueReadBuffer(commandQueue, outBuffer, true, 0, outArray.length * Sizeof.cl_int,
                outPointer, 0, null, null);
        if(lastOutsEnabled) {
            clEnqueueReadBuffer(commandQueue, lastOutsBuffer, true, 0, lastOutsArray.length * Sizeof.cl_int,
                    lastOutsPointer, 0, null, null);
        }
        clReleaseMemObject(inBuffer);
        clReleaseMemObject(outBuffer);
        System.arraycopy(tempOutArray, 0, outArray, 0, tempOutArray.length);
        if(lastOutsEnabled) {
            clReleaseMemObject(lastOutsBuffer);
            System.arraycopy(tempLastOutsArray, 0, lastOutsArray, 0, tempLastOutsArray.length);
        }
        long end = System.nanoTime();
        return end-start;
    }

    private static long applyScatter(float[] input, int[] scannedPredicates, float[] output, cl_context context, cl_command_queue commandQueue, cl_kernel kernel){
        Pointer inPointer = Pointer.to(input);
        Pointer predicatesPointer = Pointer.to(scannedPredicates);
        float[] tempOutArray = new float[output.length];
        Pointer outPointer = Pointer.to(tempOutArray);
        long localWorkSize = input.length / (WORK_GROUP_COUNT * 2);
        if(localWorkSize == 0) localWorkSize = input.length;

        cl_mem inBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input.length * Sizeof.cl_float, inPointer, null);
        cl_mem scannedPredicatesBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, scannedPredicates.length * Sizeof.cl_int, predicatesPointer, null);
        cl_mem outBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output.length * Sizeof.cl_float, null, null);

        clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(inBuffer));
        clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(scannedPredicatesBuffer));
        clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[]{output.length}));
        clSetKernelArg(kernel, 3, Sizeof.cl_mem, Pointer.to(outBuffer));
        long global_work_size[] = new long[]{localWorkSize == input.length ? input.length : input.length / 2};
        long local_work_size[] = new long[]{ localWorkSize };

        long start = System.nanoTime();
        clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, global_work_size, local_work_size, 0, null, null);
        clEnqueueReadBuffer(commandQueue, outBuffer, true, 0, output.length * Sizeof.cl_float,
                outPointer, 0, null, null);

        clReleaseMemObject(inBuffer);
        clReleaseMemObject(scannedPredicatesBuffer);
        clReleaseMemObject(outBuffer);
        System.arraycopy(tempOutArray, 0, output, 0, tempOutArray.length);
        long end = System.nanoTime();
        return end-start;
    }

    private static String readFile(String fileName)
    {
        try
        {
            BufferedReader br = new BufferedReader(new FileReader(fileName));
            StringBuilder sb = new StringBuilder();
            String line = null;
            while (true)
            {
                line = br.readLine();
                if (line == null)
                {
                    break;
                }
                sb.append(line+"\n");
            }
            return sb.toString();
        }
        catch (IOException e)
        {
            e.printStackTrace();
            return "";
        }
    }

    private static float[] sequentialScan(float[] input){
        float[] output = new float[input.length];
        output[0] = 0;
        for (int i = 1; i < input.length; i++) {
            output[i] = output[i - 1] + input[i - 1];
        }
        return output;
    }
}