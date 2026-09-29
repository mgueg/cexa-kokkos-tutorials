
```shell
cmake -B build_openmp -DKokkos_ROOT=$HOME/install_openmp
cmake --build build_openmp
./build_openmp/exercise/exe02
```

```shell
   79  module add arch/a100
   80  module add cuda/
   81  cmake -B build_cuda -DKokkos_ROOT=$HOME/install_cuda
   82  cmake --build build_cuda
   87  srun  -A for@a100 -C a100 --ntasks=1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=8 --hint=nomultithread   ./build_cuda/exercise/exe02 50 20 10
```

