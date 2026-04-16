# test overhead

## default


$ sudo /home/yunwei37/workspace/gpu/co-processor-demo/gpu_ext_policy/microbench/memory/uvmbench --kernel=gemm --mode=uvm --size_factor=1.1 --iterations=10
UVM Microbenchmark - Tier 0 Synthetic Kernels
==============================================
GPU Memory: 32109 MB
Size Factor: 1.1 (oversubscription)
Total Working Set: 35320 MB
Stride Bytes: 4096 (page-level)
Kernel: gemm
Mode: uvm
Iterations: 10

Initializing weights (34400 MB)...
  100% complete
GEMM config: dim=4096, hidden=11008, layers=200, tokens=10
  Layer size: 172.0 MB, Total weights: 34400.0 MB

Results:
  Kernel: gemm
  Mode: uvm
  Working Set: 35320 MB
  Bytes Accessed: 344000 MB
  Median time: 37047.1 ms
  Min time: 36998.4 ms
  Max time: 37123.5 ms
  Bandwidth: 9.73654 GB/s
  Results written to: results.csv
yunwei37@lab:~/workspace/gpu$ 

$  sudo /home/yunwei37/workspace/gpu/co-processor-demo/gpu_ext_policy/microbench/memory/uvmbench --kernel=hotspot --mode=uvm --size_factor=1.1 --iterations=10
UVM Microbenchmark - Tier 0 Synthetic Kernels
==============================================
GPU Memory: 32109 MB
Size Factor: 1.1 (oversubscription)
Total Working Set: 35320 MB
Stride Bytes: 4096 (page-level)
Kernel: hotspot
Mode: uvm
Iterations: 10

Hotspot config: grid=55552x55552, iterations=10
  Allocated: 35316.8 MB, Total access: 353167.5 MB

Results:
  Kernel: hotspot
  Mode: uvm
  Working Set: 35320 MB
  Bytes Accessed: 353167 MB
  Median time: 10400 ms
  Min time: 10282.6 ms
  Max time: 10472.1 ms
  Bandwidth: 35.608 GB/s
  Results written to: results.csv


## gbpf 

 sudo /home/yunwei37/workspace/gpu/co-processor-demo/gpu_ext_policy/microbench/memory/uvmbench --kernel=gemm --mode=uvm --size_factor=1.1 --iterations=10
UVM Microbenchmark - Tier 0 Synthetic Kernels
==============================================
GPU Memory: 32109 MB
Size Factor: 1.1 (oversubscription)
Total Working Set: 35320 MB
Stride Bytes: 4096 (page-level)
Kernel: gemm
Mode: uvm
Iterations: 10

Initializing weights (34400 MB)...
  100% complete
GEMM config: dim=4096, hidden=11008, layers=200, tokens=10
  Layer size: 172.0 MB, Total weights: 34400.0 MB

Results:
  Kernel: gemm
  Mode: uvm
  Working Set: 35320 MB
  Bytes Accessed: 344000 MB
  Median time: 37036.2 ms
  Min time: 37025.8 ms
  Max time: 37069.3 ms
  Bandwidth: 9.73939 GB/s
  Results written to: results.csv

$ sudo /home/yunwei37/workspace/gpu/co-processor-demo/gpu_ext_policy/microbench/memory/uvmbench --kernel=hotspot --mode=uvm --size_factor=1.1 --iterations=10
UVM Microbenchmark - Tier 0 Synthetic Kernels
==============================================
GPU Memory: 32109 MB
Size Factor: 1.1 (oversubscription)
Total Working Set: 35320 MB
Stride Bytes: 4096 (page-level)
Kernel: hotspot
Mode: uvm
Iterations: 10

Hotspot config: grid=55552x55552, iterations=10
  Allocated: 35316.8 MB, Total access: 353167.5 MB

Results:
  Kernel: hotspot
  Mode: uvm
  Working Set: 35320 MB
  Bytes Accessed: 353167 MB
  Median time: 10417.3 ms
  Min time: 10262.9 ms
  Max time: 10511.6 ms
  Bandwidth: 35.5487 GB/s
  Results written to: results.csv

## test with sequencial prefetch

$ sudo /home/yunwei37/workspace/gpu/co-processor-demo/gpu_ext_policy/src/prefetch_adaptive_sequential -b 4096

$ sudo /home/yunwei37/workspace/gpu/co-processor-demo/gpu_ext_policy/microbench/memory/uvmbench --kernel=gemm --mode=uvm --size_factor=1.1 --iterations=10
UVM Microbenchmark - Tier 0 Synthetic Kernels
==============================================
GPU Memory: 32109 MB
Size Factor: 1.1 (oversubscription)
Total Working Set: 35320 MB
Stride Bytes: 4096 (page-level)
Kernel: gemm
Mode: uvm
Iterations: 10

Initializing weights (34400 MB)...
  100% complete
GEMM config: dim=4096, hidden=11008, layers=200, tokens=10
  Layer size: 172.0 MB, Total weights: 34400.0 MB

Results:
  Kernel: gemm
  Mode: uvm
  Working Set: 35320 MB
  Bytes Accessed: 344000 MB
  Median time: 22475.4 ms
  Min time: 22452.3 ms
  Max time: 23000 ms
  Bandwidth: 16.0491 GB/s
  Results written to: results.csv
yunwei37@lab:~/workspace/gpu$ /hom


##




