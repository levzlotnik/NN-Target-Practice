//
// Created by LevZ on 6/15/2020.
//

#ifndef TARGETPRACTICE_DISTRIBUTIONBASE_H
#define TARGETPRACTICE_DISTRIBUTIONBASE_H
#include "../common.h"
#include <type_traits>
#include "../BLAS/blas.h"
#include <memory>

#define INDENT_INC "    "

template<typename T>
struct sequence_type;

template<> struct sequence_type<float> {using T = Vector;};
template<> struct sequence_type<Vector> {using T = Matrix;};

using namespace std;

template<class T>
class DistributionBase {
public:
    using sample_type = T;
    using sequence_type = typename sequence_type<T>::T;

    virtual ~DistributionBase() {};

    virtual float logp(sample_type samples) = 0;
    virtual sequence_type logp_batch(sequence_type samples) = 0;

    virtual sample_type sample() = 0;
    virtual sequence_type sample_sequence(size_t n) = 0;

    virtual sample_type rsample(const vector<Vector>& inputs) const = 0;
    virtual sequence_type jac_rsample(int i, const vector<Vector>& inputs, sample_type output) const = 0;

    virtual ostream &print(ostream &os, string indent, bool deep = false) const = 0;

    friend ostream& operator <<(ostream& os, const DistributionBase& dist) {
        return dist.print(os, "", false);
    }

    [[nodiscard]] virtual DistributionBase* clone() const = 0;
};


class UnivariateDistribution : public DistributionBase<float> {
public:
    sequence_type logp_batch(sequence_type samples) override {
        return samples.apply([this](float& x){ return this->logp(x);});
    }

    sequence_type sample_sequence(size_t n) override {
        Vector res(n);
        for (float& x: res)
            x = this->sample();
        return res;
    }

    [[nodiscard]] UnivariateDistribution *clone() const override = 0;

};

class MultivariateDistribution : public DistributionBase<Vector> {
public:
    int sample_shape;
    explicit MultivariateDistribution(int k): sample_shape(k) {}

    sequence_type logp_batch(sequence_type samples) override {
        Matrix ret(samples.n, 1);
        for (int i=0; i < ret.n; ++i)
            ret(i, 0) = this->logp(samples.get_row(i));
        return ret;
    }

    Matrix sample_sequence(size_t n) override {
        Matrix res(n, sample_shape, false);
        for (int i=0; i < n; ++i)
            res.set_row(i, this->sample());
        return res;
    }

    [[nodiscard]] MultivariateDistribution *clone() const override = 0;
};


class EncapsulateUnivariate : public MultivariateDistribution {
private:
    std::shared_ptr<UnivariateDistribution> dist;

public:
    explicit EncapsulateUnivariate(const UnivariateDistribution& dist) : dist(dist.clone()),
        MultivariateDistribution(1) {

    }

    float logp(Vector val) override {
        return dist->logp(val[0]);
    }

    Vector sample() override {
        return Vector(1, dist->sample());
    }

    sample_type rsample(const vector<Vector> &inputs) const override {
        return Vector(1, dist->rsample(inputs));
    }

    sequence_type jac_rsample(int i, const vector<Vector> &inputs, sample_type output) const override {
        if (output.n != 1)
            throw runtime_error("Wtf dude output was supposed to have n = 1, "
                                "got n = " + to_string(output.n) + ".");
        Matrix res_vec({ dist->jac_rsample(i, inputs, output[0]) });
        return res_vec;
    }

    ostream &print(ostream &os, string indent, bool deep) const override {
        os << "EncapsulateUnivariate(";
        if (deep){
            os << endl << indent << "dist=";
            dist->print(os, indent + INDENT_INC, true);
            os << indent << endl << ")";
        }
        else {
            os << "dist=";
            dist->print(os,  indent + INDENT_INC, false);
            os << ")";
        }
        return os;
    }

    [[nodiscard]] EncapsulateUnivariate *clone() const override {
        return new EncapsulateUnivariate(*this);
    }
};

// Model / Graph default distribution type is MultivariateDistribution
// Because it can represent UnivariateDistributions as well by using size()==1.

using Distribution = MultivariateDistribution;


#endif //TARGETPRACTICE_DISTRIBUTIONBASE_H
