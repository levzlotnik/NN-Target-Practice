//
// Created by LevZ on 6/18/2020.
//

#include "../distributions/distributions.h"

int main() {
    Normal dist(23, 15);
    auto samples = dist.sample_sequence(1000);
    cout << "samples.mean(), samples.std() = " << samples.mean() << ", " << samples.std() << endl;
    float arr[] = {1., 2., 3.};
    Vector v1(3, arr, false, true);
    auto v2 = 1 / v1;
    MultivariateGaussian gaussian(v1, v2);
    auto vector_samples = gaussian.sample_sequence(10000);

    cout << "vector_samples.mean(axis=0), vector_samples.std(axis=0) = "
         << vector_samples.mean(0) << ", "
         << vector_samples.std(0) << endl;


    EncapsulateUnivariate encapsulated_normal(dist);
    MultivariateDistribution& encap = encapsulated_normal;
    cout << "encap.sample_sequence(3) = " << encap.sample_sequence(3) << endl;
    cout << "encap.sample_sequence(1000).mean() = " << encap.sample_sequence(1000).mean() << endl;


}