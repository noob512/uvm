baseline :median ≈ 1922.22 ms bw ≈ 21.89 GB/s
device-only baseline：seq_device_prefetch + uvm median ≈ 1430.89 ms bw ≈ 29.41 GB/s
好的协同：prefetch_always_max + seq_device_prefetch + uvm median ≈ 1088.34 ms bw ≈ 38.67 GB/s
坏的协同：prefetch_adaptive_sequential + seq_stream + uvm： median ≈ 2089.77 ms bw = 20.14 GB/s
param:
--mode=uvm
--working_set_bytes= 40 
--stride_bytes=4096（页粒度访问）
--iterations=5
kernel：
seq_stream：普通顺序流式 kernel，不做 device prefetch。
seq_device_prefetch：带 PTX prefetch.global.L2 的顺序流式 kernel（device prefetch）。
最后一个好像写错了
prefetch_adaptive_sequential + seq_device_prefetch + uvm： median ≈ 2089.77 ms bw = 20.14 GB/s
