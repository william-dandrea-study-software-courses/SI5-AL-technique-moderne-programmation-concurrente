// Which GPU do I have?
#include <iostream>

int main() {
    int noOfDevices;
    /* get no. of device */
    cudaGetDeviceCount(&noOfDevices);
    cudaDeviceProp prop;
    for (int i = 0; i < noOfDevices; i++) {
        /*get device properties */
        cudaGetDeviceProperties(&prop, i);

        std::cout << "Device Name:\t\n" << prop.name << std::endl;
        std::cout << "Total global memory:\t" << prop.totalGlobalMem << std::endl;
        std::cout << "Shared memory / SM:\t" << prop.sharedMemPerBlock << std::endl;
        std::cout << "Registers / SM:\t" << prop.regsPerBlock << std::endl;
        std::cout << "max threads/blocks:\t" << prop.maxThreadsPerBlock << std::endl;
        std::cout << "max threads in each dimension:\t" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1]
                  << ", " << prop.maxThreadsDim[1] << std::endl;
        std::cout << "max blocks in each dimension:\t" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", "
                  << prop.maxGridSize[1] << std::endl;
        std::cout << "Nb of multiproc on device:\t" << prop.multiProcessorCount << prop.maxGridSize[1] << std::endl;
    }

}
