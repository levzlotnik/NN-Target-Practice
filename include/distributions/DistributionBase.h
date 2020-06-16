//
// Created by LevZ on 6/15/2020.
//

#ifndef BLAS_DISTRIBUTIONBASE_H
#define BLAS_DISTRIBUTIONBASE_H
#include "../../common.h"
#include <type_traits>
#include "../BLAS.h"
#include <memory>

#define INDENT_INC "    "

using namespace std;

template<class ResultType>
class DistributionBase {
public:
    using T = ResultType;

//    virtual ~DistributionBase() {};

    virtual float logp(T val) = 0;

    virtual T random() = 0;

    virtual ostream &print(ostream &os, string indent, bool deep = false) const = 0;

    friend ostream& operator <<(ostream& os, const DistributionBase& dist) {
        return dist.print(os, "", false);
    }

    virtual DistributionBase* clone() const = 0;
};


class UnivariateDistribution : public DistributionBase<float> {
public:
    Vector random_sequence(size_t n) {
        Vector res(n);
        for (float& x: res)
            x = this->random();
        return res;
    }

    virtual UnivariateDistribution *clone() const override = 0;

};
class MultivariateDistribution : public DistributionBase<Vector> {
public:
    int k;
    explicit MultivariateDistribution(int k): k(k) {}

    Matrix random_sequence(size_t n) {
        Matrix res(n, k);
        for (int i=0; i < n; ++i)
            res.set_row(i, this->random());
        return res;
    }

    virtual MultivariateDistribution *clone() const override = 0;
};


class EncapsulateUnivariate : public MultivariateDistribution {
private:
    std::shared_ptr<UnivariateDistribution> dist;

public:
    EncapsulateUnivariate(const UnivariateDistribution& dist) : dist(dist.clone()),
        MultivariateDistribution(1) {

    }

    float logp(Vector val) override {
        return dist->logp(val[0]);
    }

    Vector random() override {
        return Vector(1, dist->random());
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

    EncapsulateUnivariate *clone() const override {
        return new EncapsulateUnivariate(*this);
    }
};

class UnivariateProjection : public UnivariateDistribution {
private:
    int i; // index of projection
    std::shared_ptr<MultivariateDistribution> dist;

public:
    explicit UnivariateProjection(const MultivariateDistribution& dist, int i = 0) : dist(dist.clone()), i(i) {}

    float logp(T val) override {
        warning::warn("UserWarning: cannot determine the marginal probability "
                    "for projection without additional information. This is usually caused by the"
                    " intractability of the integral of all dependencies + joint variables.");
        return 0;
    }

    T random() override {
        return dist->random()[i];
    }

    ostream &print(ostream &os, string indent, bool deep) const override {
        os << "UnivariateProjection(i=" << i << ", ";
        if (deep){
            os << endl << indent << "dist=";
            dist->print(os, indent + INDENT_INC, true);
            os << indent << endl << ")";
        }
        else {
            os << "dist=";
            dist->print(os, indent + INDENT_INC, false);
            os << ")";
        }
        return os;
    }

    UnivariateProjection *clone() const override {
        return new UnivariateProjection(*this);
    }
};

// Model / Graph default distribution type is MultivariateDistribution
// Because it can represent UnivariateDistributions as well

using Distribution = MultivariateDistribution;


#endif //BLAS_DISTRIBUTIONBASE_H
