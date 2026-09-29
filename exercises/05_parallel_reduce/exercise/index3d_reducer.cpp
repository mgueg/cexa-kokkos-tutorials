///FIXME: https://kokkos.org/kokkos-core-wiki/API/core/builtinreducers/MinLoc.html
///FIXME: for the moment this code dont compile with kokkos 5.2.2 

#include <Kokkos_Core.hpp>
struct Idx3D_t {
  int value[3];
  int& operator[](int i) { return value[i]; }
  const int& operator[](int i) const { return value[i]; }
};
template <>
struct Kokkos::reduction_identity<Idx3D_t> {
  static constexpr Idx3D_t min() { return {0, 0, 0}; }
};

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<double***> a("A", 5, 5, 5);
    Kokkos::deep_copy(a, 10);
    a(2, 3, 1)        = 5;
    using MinLoc_t    = Kokkos::MinLoc<double, Idx3D_t>;
    using MinLocVal_t = typename MinLoc_t::value_type;
    MinLocVal_t result;
    Kokkos::parallel_reduce(
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {5, 5, 5}),
        KOKKOS_LAMBDA(int i, int j, int k, MinLocVal_t& val) {
          if (a(i, j, k) < val.val) {
            val.val    = a(i, j, k);
            val.loc[0] = i;
            val.loc[1] = j;
            val.loc[2] = k;
          }
        },
        MinLoc_t(result));
    printf("%lf %i %i %i\n", result.val, result.loc[0], result.loc[1],
           result.loc[2]);
  }
  Kokkos::finalize();
}
