//
// Created by LevZ on 6/16/2020.
//

#ifndef TARGETPRACTICE_VARIABLEBASE_H
#define TARGETPRACTICE_VARIABLEBASE_H

#include <memory>
#include <utility>
#include "utils/GraphvizPrinter.h"
#include "blas/blas.h"
#include "../common.h"

using namespace std;
using blas::Tensor;

namespace autograd {
    template<typename T>
    class Variable;

    template<typename T>
    class AutogradVariable;

    template<typename T>
    class VariableBase {
    protected:
        vector<Variable<T>> dependencies;
        vector<VariableBase *> dependees;
        Tensor<T> _data;
        Tensor<T> _grad;
        string name;

        void check_graph_integrity(unordered_set<VariableBase *> &visited);

        // Backpropagates the gradient to the current dependencies.
        // If recursive=true - backpropagates for all dependencies as well.
        bool grad_accumulation_complete() const;

        virtual void backward(VariableBase *dependee, bool recursive) = 0;

        unordered_map<VariableBase *, int> unvisited_dependees;

        friend class AutogradVariable<T>;

        VariableBase(string name, const Tensor<T> &data, const Tensor<T> &grad_data, bool requires_grad = true) :
                name(std::move(name)), _data(data),
                _grad(grad_data), requires_grad(requires_grad) {}

        VariableBase(const string &name, const Tensor<T> &data, bool requires_grad = true)
                : VariableBase(name, data, blas::zeros_like(data), requires_grad) {}

    public:
        bool requires_grad;

        virtual ~VariableBase();

        Tensor<T> &data();

        Tensor<T> &grad();

        inline shape_t shape() const { return _data.shape; }

        virtual void add_dependency(const Variable<T> &dep);

        void remove_dependency(const Variable<T> &dep);

        void accumulate_grad(const Tensor<T> &grad);

        virtual Tensor<T> forward() = 0;

        // Prepares the graph for backward operation. Should be called before
        // calling backward each time.
        void prepare_backward();

        // Backpropagates through the entire graph and assigns gradients for every variable
        // according to the current gradient.
        void backward();

        void zero_grad(bool recursive);

        // Returns true if this is a leaf of the graph - idx_proj.e. no dependencies.
        bool is_leaf() const;

        // Returns true if this is a root of the graph - idx_proj.e. no dependees.
        virtual bool is_root() const;

        virtual bool is_param() const { return false; }

        virtual bool is_input_buffer() const { return false; }

        void check_graph_integrity();

        inline VariableBase &rename(const string &new_name) {
            this->name = new_name;
            return (*this);
        }

        Tensor<T> forward_recursive();

        ostream &print_graphviz(ostream &os);

        GraphvizPrinter &gather_connection_graphviz(GraphvizPrinter &gvzp);

    private:
        virtual string node_style_graphviz();
    };

    template<typename T>
    class Variable {
    private:
        using ptr_type = shared_ptr<VariableBase<T>>;
        ptr_type ptr;
    public:
        inline explicit Variable(ptr_type p) : ptr(p) {}
        inline explicit Variable(VariableBase<T> *p) : ptr(p) {}
        inline Variable &rename(string name) {
            ptr->rename(name);
            return (*this);
        }
        inline VariableBase<T> &operator*() const noexcept { return *ptr; }
        inline VariableBase<T> *get() const noexcept { return ptr.get(); }
        inline VariableBase<T>* operator->() const noexcept { return ptr.operator->(); }
        inline bool equals(const Variable &other) const noexcept { return ptr == other.ptr; }
        inline shape_t shape() const { return ptr->shape(); }
        inline Tensor<T>& data() const { return ptr->data(); }
        inline Tensor<T>& grad() const { return ptr->grad(); }

        // Math Operations:
#define DECL_VARIABLE_MATH_METHOD(func) \
        Variable<T> func() const;

        MACRO_MATH_FUNCTIONS(DECL_VARIABLE_MATH_METHOD)


    };
}

#endif //TARGETPRACTICE_VARIABLEBASE_H
