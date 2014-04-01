import pycuda.autoinit
import pycuda.driver as drv
import numpy
import array
import time
import math

from pycuda.compiler import SourceModule

class CudaSpatialGrep(object):

    def __init__(self,  block = (256, 1, 1), grid = (15, 1)):

        self.mod = SourceModule("""
typedef unsigned long long int uint64_t;
__global__ void spatial_grep(double *lonandlat, double *coordinates, uint64_t *hit, uint64_t *length)
{
    const uint64_t i = gridDim.x * blockDim.x * blockIdx.y + blockIdx.x * blockDim.x + threadIdx.x;

    int batch_size = 50;
    
    const uint64_t i_for_batch = i * batch_size;

    for (int coordinate_index = 0; (coordinate_index < batch_size) && (i_for_batch + coordinate_index < length[0]); coordinate_index++) {

        if ((coordinates[(i_for_batch + coordinate_index) * 2] > lonandlat[0] && coordinates[(i_for_batch + coordinate_index) * 2 + 1] > lonandlat[1]) && (coordinates[(i_for_batch + coordinate_index) * 2] < lonandlat[2] && coordinates[(i_for_batch + coordinate_index) * 2 + 1] < lonandlat[3])) {
            hit[i_for_batch + coordinate_index] = 1;         
        } else {
            hit[i_for_batch + coordinate_index] = 0;
        }

    }

}
""")

        self.spatial_grep = self.mod.get_function("spatial_grep")

        self.block = block
        self.grid = grid             

    def benchmark_begin(self, title):
        print "start to " + title
        self.start = time.clock()

    def benchmark_end(self, title):
        print "end of " + title
        elapsed = (time.clock() - self.start)
        print "time: " + str(elapsed)

        return elapsed

    def cuda_spatial_grep(self, lonandlat, coordinates):

        lonandlat = numpy.array(lonandlat).astype(numpy.float64)
        coordinates = numpy.array(coordinates).astype(numpy.float64)
        length = numpy.array([coordinates.shape[0] / 2]).astype(numpy.uint64)
        hits = numpy.zeros(length[0]).astype(numpy.uint64)

        print(length)
 
        custom_grid = (int(math.ceil(float(length[0]) / (50 * 256))), 1)
        print "custom grid: ", custom_grid
 
        self.spatial_grep(
                drv.In(lonandlat), drv.In(coordinates), drv.Out(hits), drv.In(length),
                block = self.block, grid = custom_grid)
        
        print hits
        print hits.shape
        print numpy.sum(hits)
        
        return hits

 
