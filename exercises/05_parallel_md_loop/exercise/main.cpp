#include <iostream>

#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]) {

    Kokkos::initialize(argc, argv);
    {

        // Parameters

        int Nx = 128;
        int Ny = 128;
	using Def = Kokkos::DefaultExecutionSpace;

        // _____________________________________________________
        // Read Nx, Ny from the command line

        if (argc >= 3) {
            Nx = std::atoi(argv[1]);
            Ny = std::atoi(argv[2]);
        }

	Kokkos::Timer timer;
        double timer_start, timer_stop;

        // _____________________________________________________
        // Create two 2D View of size Nx x Ny

        std::cout << "Creation of two 2D Views of size " << Nx << " x " << Ny << std::endl;
	Kokkos::View<float**>
		T_new("T_new",Nx,Ny);
	Kokkos::View<float**>
		T("T",Nx,Ny);


        // _____________________________________________________
        // Initialize "T_new" field with -1.0
	// ANOTHER WAY IS TO deep_copy with scalar
#if 0
	Kokkos::parallel_for(
		"Init T_new",
		Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace,
					Kokkos::Rank<2> >
					({0,0},{Nx,Ny}),
		KOKKOS_LAMBDA (int i, int j) {
			T_new(i,j) = -1.0;
		}
		);
        // ...
#endif
	timer_start = timer.seconds();
	Kokkos::deep_copy(T_new, -1.0);
        Kokkos::fence();
        timer_stop = timer.seconds();

        std::cout << "Time of deep_copy : " << timer_stop - timer_start << std::endl;

        // _____________________________________________________
        // Initialize "T" field

	timer_start = timer.seconds();
        Kokkos::parallel_for("Initialization for T", 
			Kokkos::MDRangePolicy<Def,
						Kokkos::Rank<2>>
			({0,0},{Nx,Ny}),
		KOKKOS_LAMBDA(const int i,const int j) {
		    T(i, j) = (i % 2 == 0) ? 2.0 : 1.0;
        });

        Kokkos::fence();
        timer_stop = timer.seconds();

        std::cout << "Time of Init T : " << timer_stop - timer_start << std::endl;
        // ...

        // _____________________________________________________
        // Jacobi Update
        // Beware of indexing : if no debug option, produce segfault
	// if you access to undefined range
	timer_start = timer.seconds();
        Kokkos::parallel_for("Jacobi Update for T", 
			Kokkos::MDRangePolicy<Def,
						Kokkos::Rank<2>>
			({1,1},{Nx-1,Ny-1}),
		KOKKOS_LAMBDA(const int i,const int j) {
		    T_new(i, j) = 0.25 * (T(i-1, j) + T(i+1, j) + T(i, j-1) + T(i, j+1));
        });
        Kokkos::fence();
        timer_stop = timer.seconds();

        std::cout << "Time of Update T : " << timer_stop - timer_start << std::endl;

        // ...

        // _____________________________________________________
        // Copy the result to a Host accessible View
	
	timer_start = timer.seconds();
        auto T_host_mirror = Kokkos::create_mirror_view_and_copy(T_new); 

        Kokkos::fence();
        timer_stop = timer.seconds();

        std::cout << "Time of copy back to the Host : " << timer_stop - timer_start << std::endl;
        // _____________________________________________________
        // Check the result

        double error = 0.0;
        for (int i = 1; i < Nx - 1; i++) {
            for (int j = 1; j < Ny - 1; j++) {
                error += std::abs(1.5 - T_host_mirror(i,j));
            }
        }

        std::cout << "Error: " << error << std::endl;

    }
    Kokkos::finalize();

    return 0;
}
