package at.hpc.scan;

import org.jocl.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import static org.jocl.CL.*;

public class Main
{

    private static final int WORK_GROUP_COUNT = 1024;

    public static void main(String args[])
    {

//        float[] inputArray = new float[]{3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1,3,1,7,2,4,1,6,3,3,1,7,2,12,32,14,1};

//        float[] inputArray = new float[]{5,1,1,1,1,1,1,1,1,1,1,1,1,1,2,6,7,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,5,1,1,1,1,1,1,1,1,1,1,1,1,1,2,6,7,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8};
        int[] doubleArray = new Random().ints((long) Math.pow(2, 16), 0, 1000).toArray();
        float[] inputArray = new float[doubleArray.length];
        for (int i = 0 ; i < doubleArray.length; i++) {
            inputArray[i] = (float) doubleArray[i];
        }
        float[] outputArray = new float[inputArray.length];
        float[] lastOutsArray = new float[WORK_GROUP_COUNT];
        float[] lastOutsScannedArray = new float[WORK_GROUP_COUNT];

//        String programSource = readFile("src/at/hpc/scan/kernel.cl");
        String programSource = readFile("src/at/hpc/scan/kernel_work_efficient.cl");
        String addProgramSource = readFile("src/at/hpc/scan/kernel_add.cl");

        // The platform, device type and device number
        // that will be used
        final int platformIndex = 2;
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
        cl_program program = clCreateProgramWithSource(context,
                1, new String[]{ programSource }, null, null);
        clBuildProgram(program, 0, null, null, null, null);
        cl_kernel kernel = clCreateKernel(program, "scan", null);
        System.out.println("Num: \t\t" + inputArray.length);
//        System.out.println("Input: \t\t" + Arrays.toString(inputArray));
        float[] seqOut = sequentialScan(inputArray);
        scanInput(inputArray, outputArray, lastOutsArray, context, commandQueue, kernel);
//        System.out.println("Output1: \t" + Arrays.toString(outputArray));
        scanInput(lastOutsArray, lastOutsScannedArray, null, context, commandQueue, kernel);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        cl_program add_program = clCreateProgramWithSource(context, 1, new String[]{addProgramSource}, null, null);
        clBuildProgram(add_program, 0, null, null, null, null);
        cl_kernel add_kernel = clCreateKernel(add_program, "add", null);
        addLastOutsToOut(lastOutsScannedArray, outputArray, context, commandQueue, add_kernel);
//        System.out.println("Output2: \t" + Arrays.toString(outputArray));
        clReleaseKernel(add_kernel);
        clReleaseProgram(add_program);
        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);
        for(int i = 0; i < seqOut.length; i ++){
            if(seqOut[i] != outputArray[i]){
                System.out.println("NOPE! " + i + " " + seqOut[i] + " != " + outputArray[i] + " diff by " + (seqOut[i] - outputArray[i]));
                break;
            }
        }
//        System.out.println("SeqOutput: \t" + Arrays.toString(seqOut));
//        System.out.println("LastOuts: \t" + Arrays.toString(lastOutsArray));
//        System.out.println("Scanned: \t" + Arrays.toString(lastOutsScannedArray));
        System.out.println("Parallel successful ? " + (Arrays.equals(outputArray, seqOut) ? "true" : "false"));
    }

    private static void addLastOutsToOut(float[] lastOutsScannedArray, float[] outputArray, cl_context context, cl_command_queue commandQueue, cl_kernel kernel) {
        Pointer lastOutsPointer = Pointer.to(lastOutsScannedArray);
        Pointer inPointer = Pointer.to(outputArray);
        float[] tempOutArray = new float[outputArray.length];
        Pointer outPointer = Pointer.to(tempOutArray);
        cl_mem inBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, outputArray.length * Sizeof.cl_float, inPointer, null);
        cl_mem lastOutsBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, lastOutsScannedArray.length * Sizeof.cl_float, lastOutsPointer, null);
        cl_mem outBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, tempOutArray.length * Sizeof.cl_float, null, null);
        clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(inBuffer));
        clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(lastOutsBuffer));
        clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(outBuffer));
        long global_work_size[] = new long[]{outputArray.length};
        long local_work_size[] = new long[]{ outputArray.length / lastOutsScannedArray.length };
        clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, global_work_size, local_work_size, 0, null, null);
        clEnqueueReadBuffer(commandQueue, outBuffer, true, 0, outputArray.length * Sizeof.cl_float,
                outPointer, 0, null, null);
        clReleaseMemObject(inBuffer);
        clReleaseMemObject(outBuffer);
        clReleaseMemObject(lastOutsBuffer);
        System.arraycopy(tempOutArray, 0, outputArray, 0, tempOutArray.length);
    }

    private static void scanInput(float[] inArray, float[] outArray, float[] lastOutsArray, cl_context context, cl_command_queue commandQueue, cl_kernel kernel) {
        Pointer inPointer = Pointer.to(inArray);
        float[] tempOutArray = new float[outArray.length];
        float[] tempLastOutsArray = null;
        Pointer outPointer = Pointer.to(tempOutArray);
        Pointer lastOutsPointer = null;
        boolean lastOutsEnabled = lastOutsArray != null;
        if(lastOutsEnabled) {
            tempLastOutsArray = new float[lastOutsArray.length];
            lastOutsPointer = Pointer.to(tempLastOutsArray);
        }
        long localWorkSize = inArray.length / (WORK_GROUP_COUNT * 2);
        if(localWorkSize == 0) localWorkSize = inArray.length;

        cl_mem inBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inArray.length * Sizeof.cl_float, inPointer, null);
        cl_mem outBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, outArray.length * Sizeof.cl_float, null, null);
        cl_mem lastOutsBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, lastOutsEnabled ? lastOutsArray.length * Sizeof.cl_float : 1, null, null);
        clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(inBuffer));
        clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(outBuffer));
        clSetKernelArg(kernel, 2, Sizeof.cl_float * 2 * localWorkSize, null);
        clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{lastOutsEnabled ? 1 : 0}));
        clSetKernelArg(kernel, 4, Sizeof.cl_mem, Pointer.to(lastOutsBuffer));

        long global_work_size[] = new long[]{localWorkSize == inArray.length ? inArray.length : inArray.length / 2};
        long local_work_size[] = new long[]{ localWorkSize };
        System.out.println("global work size " + global_work_size[0]);
        System.out.println("local work size " + local_work_size[0]);
        clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, global_work_size, local_work_size, 0, null, null);
        clEnqueueReadBuffer(commandQueue, outBuffer, true, 0, outArray.length * Sizeof.cl_float,
                outPointer, 0, null, null);
        if(lastOutsEnabled) {
            clEnqueueReadBuffer(commandQueue, lastOutsBuffer, true, 0, lastOutsArray.length * Sizeof.cl_float,
                    lastOutsPointer, 0, null, null);
        }
        clReleaseMemObject(inBuffer);
        clReleaseMemObject(outBuffer);
        System.arraycopy(tempOutArray, 0, outArray, 0, tempOutArray.length);
        if(lastOutsEnabled) {
            clReleaseMemObject(lastOutsBuffer);
            System.arraycopy(tempLastOutsArray, 0, lastOutsArray, 0, tempLastOutsArray.length);
        }
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