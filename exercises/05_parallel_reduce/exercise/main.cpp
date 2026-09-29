#include <iostream>
#include <Kokkos_Core.hpp>

// test to use minloc with 2d view
// see kokkos doc
struct Idx2d_T {
   int value[2];
   int& operator[](int i) { return value[i]; }
   const int& operator[](int i) const { return value[i]; }
};

// specialization of reduction with my own type
template<>
struct Kokkos::reduction_identity<Idx2d_T>{
   static constexpr Idx2d_T min() { return {0,0};}
};


int main(int argc, char* argv[]) {

    Kokkos::initialize(argc, argv);
    {

        // _____________________________________________________
        // Parameters

        int Nx = 100;
        int Ny = 100;

        // _____________________________________________________
        // Read Nx, Ny from the command line

        if (argc >= 3) {
            Nx = std::atoi(argv[1]);
            Ny = std::atoi(argv[2]);
        }

        // _____________________________________________________
        // Create a 2D view of size Nx x Ny

        std::cout << "Creation of a 2D view `matrix` of size " << Nx << " x " << Ny << std::endl;

        // ... Create a 2D view of size Nx x Ny ...
	Kokkos::View<double**>
		matrix("Mat",Nx,Ny);
        // _____________________________________________________
        // Initialize the vector using a parallel loop

        // ... Initialize the Matrix using Kokkos::parallel_for ...
        Kokkos::parallel_for(
                "Init matrix",
                Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace,
                                        Kokkos::Rank<2> >
                                        ({0,0},{Nx,Ny}),
                KOKKOS_LAMBDA (int i, int j) {
                        matrix(i,j) = i+j*std::sin(j);
                }
                );

        Kokkos::fence();

        // _____________________________________________________
        // Compute the sum of the matrix
	double sum_result ;
	Kokkos::parallel_reduce(
		"Sum",
		Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace,
                                        Kokkos::Rank<2> >
                                        ({0,0},{Nx,Ny}),
                KOKKOS_LAMBDA (int i, int j, double& loc_result) {
                        loc_result += matrix(i,j);
                },
		sum_result // possible in the case of Sum
                );

        Kokkos::fence();

        // ... Print the result ...
	std::cout << " Sum Result : " << sum_result << std::endl;
        // _____________________________________________________
        // Compute the maximum of the matrix

        // ... Compute the maximum of the matrix ...
        double max_result ;
        Kokkos::parallel_reduce(
                "MinLoc",
                Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace,
                                        Kokkos::Rank<2> >
                                        ({0,0},{Nx,Ny}),
                KOKKOS_LAMBDA (int i, int j, double& loc_result) {
                        loc_result = Kokkos::max(matrix(i,j), loc_result);
	                },
		Kokkos::Max<double>(max_result) 
                );

        Kokkos::fence();

        // ... Print the result ...
	std::cout << " Max Result : " << max_result << std::endl;
#if 0
	// test with Minloc for 2d arrays
	// FOR THE MOMENT DONT WORK...
	using MinLocType = Kokkos::MinLoc<double, Idx2d_T>;
	using MinLocVal = typename MinLocType::value_type;
	MinLocVal minloc;
        Kokkos::parallel_reduce(
                "MinLoc",
                Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace,
                                        Kokkos::Rank<2> >
                                        ({0,0},{Nx,Ny}),
                KOKKOS_LAMBDA (int i, int j, MinLocVal& loc_result) {
			if (matrix(i,j) < loc_result.val)
			{
		          loc_result.val = matrix(i,j);
			  loc_result.loc[0] = i;
			  loc_result.loc[1] = j;
			}
                },
		MinLocType(minloc) 
                );
#endif
        Kokkos::fence();
    }
    Kokkos::finalize();

    return 0;
}
