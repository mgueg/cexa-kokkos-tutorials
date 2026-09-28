#include <iostream>
#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
    //Kokkos::initialize(argc,argv);

    // ... Init Kokkos here ...
    Kokkos::ScopeGuard kokkos(argc,argv);
    // ... call configuration function here ...
    //{
	std::cout << "Hello from Kokkos" << std::endl;
	Kokkos::print_configuration(std::cout);
    //}
    // ... Finalize Kokkos here ...
    //Kokkos::finalize();
    return 0;
}
