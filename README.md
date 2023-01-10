# yolov8-seg
1. 修改几个库目录，PROTOBUF_DIR、CUDA_TOOLKIT_ROOT_DIR、CUDNN_DIR、TENSORRT_DIR、OPENBLAS_DIR  
2. 运行
```
mkdir build
cd build
make -j8
cd ../workspace
./pro
```
3. 注意事项
- 需要利用PROTOBUF重新去执行onnx下面的make_pb.sh生成代码替换src下的onnx相关代码，protobuf可以自己编译
- 利用到了openblas，可以直接下载.a文件或者自己编译，编译时需注意CPU型号
- 导出的onnx的输出必须是1-8400-116和1-32-160-160才能正常解析和后处理

4. TODO
- 后处理加速
