```
srun  -A for@a100 -C a100 --ntasks=1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=8 --hint=nomultithread  --time=01:00:00 --pty bash
bash-5.1$ module li
Currently Loaded Modulefiles:
 1) arch/a100   2) cuda/13.2.1   3) cmake/3.31.4
cmake -B build_cuda -DKokkos_ROOT=$HOME/install_cuda
cmake -B build_openmp -DKokkos_ROOT=$HOME/install_openmp
cmake --build build_openmp
cmake --build build_cuda
./build_cuda/exercise/exe04  100000
./build_openmp/exercise/exe04  100000

```
