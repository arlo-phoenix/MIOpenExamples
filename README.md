# MIOpenExamples
MIOpen examples


```bash
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" cmake -B build
cmake --build build
./build/MiOpenExample
```