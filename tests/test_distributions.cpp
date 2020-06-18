//
// Created by LevZ on 6/18/2020.
//

#include "../distributions/distributions.h"

int main() {
    Normal dist(23, 15);
    auto samples = dist.random_sequence(100000);
    cout << "samples.mean(), samples.std() = " << samples.mean() << ", " << samples.std() << endl;
    float arr[] = {1., 2., 3.};
    Vector v1(3, arr, false, true);
    auto v2 = 1 / v1;
    MultivariateGaussian gaussian(v1, v2);
    auto vector_samples = gaussian.random_sequence(10000);

    cout << "vector_samples.mean(axis=0), vector_samples.std(axis=0) = "
         << vector_samples.mean(0) << ", "
         << vector_samples.std(0) << endl;

    cout << "Vector::arange(0, 5) = " << Vector::arange(0, 5) << endl;
    cout << "Vector::linspace(0, 5, 10) = " << Vector::linspace(0, 5, 10) << endl;
    cout << "Vector::concat({v1, v2}) = " << Vector::concat({v1, v2}) << endl;
    cout << "sin(Vector::arange(-3, 3) * PI) = " << sin(Vector::arange(-3, 3) * PI) << endl;

    EncapsulateUnivariate encapsulated_normal(dist);
    MultivariateDistribution& encap = encapsulated_normal;
    cout << "encap.random_sequence(3) = " << encap.random_sequence(3) << endl;
    cout << "encap.random_sequence(1000).mean() = " << encap.random_sequence(1000).mean() << endl;
    UnivariateProjection project_gaussian(gaussian, 1);
    UnivariateDistribution& proj = project_gaussian;
    cout << "proj.random_sequence(7) = " << proj.random_sequence(7) << endl;
    cout << "proj.random_sequence(1000) = " << proj.random_sequence(1000).mean() << endl;
    cout << "proj = " << proj << endl;
    cout << "proj = "; proj.print(cout, "    ", true) << endl;
    v1 = Vector::arange(0, 20);
    v2 = Vector::ones_like(v1);
    gaussian = MultivariateGaussian(v1, v2);
    project_gaussian = UnivariateProjection(gaussian, 6);
    cout << "proj.random_sequence(7) = " << proj.random_sequence(7) << endl;
    cout << "proj.random_sequence(1000) = " << proj.random_sequence(1000).mean() << endl;
    cout << "proj = " << proj << endl;
    cout << "proj = "; proj.print(cout, "    ", true) << endl;
}