#include <iostream>
#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]) {

    Kokkos::initialize(argc, argv);
    {

        // _____________________________________________________
        // Parameters

        int N = 100;

        Kokkos::Timer timer;
        double timer_start = 0;
        double timer_stop  = 0;

        // _____________________________________________________
        // Read Nx from the command line

        if (argc > 1) {
            N = std::atoi(argv[1]);
        }

        // _____________________________________________________
        // Create a 1D view of size Nx

        std::cout << "Creation the 1D view `vector` of size " << N << std::endl;
	Kokkos::View<double*>
		matrix("1D vector size Nx",N);
	timer_start = timer.seconds();
        // _____________________________________________________
        // Initialize the vector using a parallel loop
	// computation will be done depending of backend
	// cpu if cpu backend and gpu if GPU backend
	// you can control with RangePolicy<ExecutionSpace>
	// and set RangePolicy<DefaultHostExecutionSpace>
	Kokkos::parallel_for(
			"1st loop",
			N,
			KOKKOS_LAMBDA(int i) 
			{
			  matrix(i) = i+1.41;
			}
			);

        // ... Fence ...
        Kokkos::fence("1st loop fence");
        // _____________________________________________________
        // Create a mirror view of the vector

	auto mirror_view = Kokkos::create_mirror_view_and_copy(matrix); 
        // _____________________________________________________
        // Deep copy the vector to the mirror view

        // ... Deep copy the vector to the mirror view ...
	std::cout << " after create_mirror_view_and_copy() " << std::endl;

        // _____________________________________________________
        // Check the result
	// _____________________________________________________
	///!!! beware this below check does not work
	///!!! if used with GPU backend, becuz of inaccessible data (matrix variable)
	///!!!  throw Kokkos::View ERROR: attempt to access inaccessible memory space (label="1D vector size Nx")
	//
        double error = 0.0;
        for (int i = 0; i < N; i++) {
           error  +=  std::abs(matrix(i)-mirror_view(i));
        }
	std::cout << " Error between mirrors ?? "  << error << std::endl;
	timer_stop = timer.seconds();
	std::cout << "Time to compute parallel loop: " << timer_stop - timer_start << std::endl;
    }
    Kokkos::finalize();

    return 0;
}
