#include <iostream>
#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]) {

    Kokkos::initialize(argc, argv);
    {

        int Nx = 10;
        int Ny = 15;
        int Nz = 20;

        // Read Nx, Ny, Nz from the command line
        if (argc == 4) {
            Nx = std::atoi(argv[1]);
            Ny = std::atoi(argv[2]);
            Nz = std::atoi(argv[3]);
        }

        // Create a 3D view of size Nx x Ny x Nz and type `double`
	Kokkos::View<double***>
		mat("My1st 3D MAT", Nx,Ny,Nz);

        // Get the rank
        const int mat_rank = mat.rank();
	std::cout << " rank from Kokkos " << mat_rank << std::endl;
        // Get the extent of the view
        auto views = { mat.extent(0), mat.extent(1), mat.extent(2)}; 
        auto strides = { mat.stride(0), mat.stride(1), mat.stride(2)}; 
        auto layout = mat.layout() ;
	//std::cout << " layout from Kokkos "  << layout << std::endl;
	std::cout << " view from Kokkos "  << std::endl;
	std::copy(views.begin(), views.end(), 
		std::ostream_iterator<int>(std::cout," - "));
	std::cout << std::endl;
        // Get the stride of the view
        // ...
	std::cout << " strides from Kokkos "  << std::endl;
	std::copy(strides.begin(), strides.end(), 
		std::ostream_iterator<int>(std::cout," - "));
	std::cout << std::endl;
    }
    Kokkos::finalize();

    return 0;
}
