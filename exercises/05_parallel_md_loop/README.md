# Exercise 5: MD parallel loop

## Objective

The goal of this exercise is to use multidimensional parallel loops with `Kokkos::MDRangePolicy`.

## Step 1: Create two Views

In the file `main.cpp`, create two 2D Views `T` and `T_new` of size `Nx` x `Ny` and of type `double`.

## Step 2: Initialize `T` and `T_new`

Use the scalar fill syntax (`Kokkos::deep_copy`) to initialize `T_new` to `-1.0`.

Replace the `Kokkos::RangePolicy` with a 2D `Kokkos::MDRangePolicy` to initialize `T` with the formula:

```cpp
T(i, j) = (i % 2 == 0) ? 2.0 : 1.0;
```

## Step 3: Update the Jacobi iteration

Replace the `Kokkos::RangePolicy` with a 2D `Kokkos::MDRangePolicy` to iterate over the interior points of `T` (excluding the boundaries) and compute `T_new` with the Jacobi formula:

```cpp
T_new(i, j) = 0.25 * (T(i-1, j) + T(i+1, j) + T(i, j-1) + T(i, j+1));
```

## Step 4: Check the result

Create a mirror View called `T_host_mirror` and copy the data from `T_new` to it.
Compute the sum of the absolute errors, the expected value is `1.5` on all interior points.

## Step 5: Timers

Use `Kokkos::Timer` to measure the time spent in the initialization and in the Jacobi update.
Call `Kokkos::fence()` before stopping each timer, since kernels are executed asynchronously.

## Step 6: Compile and run the program

Prepare the OpenMP environment:

```sh
export OMP_NUM_THREADS=<number of cores to use>
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
```

Compile the program with the OpenMP backend and run it.
Use a large View size for more meaningful results:

```sh
build_openmp/exe05 10240 10240
```

If you have access to a GPU, compile and run the program with the corresponding backend.
Compare the times between the two backends.

## Bonus step: Iteration order

By default, `Kokkos::MDRangePolicy` chooses an iteration order suited to the default layout of the execution space.

Modify the Jacobi update to test combinations of `Iterate::Left` and `Iterate::Right`, and compare the timings with a Host backend and a Device backend.

- Which combination is the fastest on each backend?
