module use /opt/insy/modulefiles
module load cuda
module load devtoolset/8

mkdir debug
cd debug
cmake3 -DCMAKE_CUDA_FLAGS="-arch=compute_50" ..
make -j4
./imgproc-benchmark -a ../images/42.png
./imgproc-benchmark -a ../images/calm.png
./imgproc-benchmark -a ../images/nudibranch.png
./imgproc-benchmark -a ../images/squares.png
./imgproc-benchmark -a ../images/test.png
./imgproc-benchmark -a ../images/test2.png
