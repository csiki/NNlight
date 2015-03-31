/**
 * Project NNlight
 */

#ifndef _NEURON_H
#define _NEURON_H

#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <random>
#include <numeric>

using std::unordered_set;
using std::unordered_map;
using std::shared_ptr;

namespace NNlight {

class Neuron;
typedef shared_ptr<Neuron> NeuronPtr;

class Neuron {
public:
	friend class OutputNeuron;
    
	/**
	 * Connects two given neurons: from --> to.
	 * @param from
	 * @param to
	 */
	template <typename FromPtr, typename ToPtr>
	static void connect(FromPtr from, ToPtr to)
	{
		from->connect_output(std::dynamic_pointer_cast<Neuron>(to));
		to->connect_input(std::dynamic_pointer_cast<Neuron>(from));
	}

    /**
     * Creates a neuron instance with the given learning rate.
     * @param learning_rate_
     */
	Neuron(double learning_rate_ = def_learning_rate, double regularization_ = def_regularization);
	virtual ~Neuron();
    
    /**
     * Induces input, activating the neuron.
     * @param from
     * @param act
     */
    virtual void propagate(NeuronPtr from, double act);
    
    /**
     * Forces the neuron to alter its weight (that is connects this to the 'from' neuron) if necessary.
     * @param from
     * @param err
     */
    virtual void backpropagate(NeuronPtr from, double err);

	/**
     * Default learning rate.
     */
    static double def_learning_rate;

	/**
     * Default regularization term. Force weights to stay lower, serve generalization.
     */
	static double def_regularization;

protected:
    /**
     * Weight of the bias input neuron.
     */
    double biasweight;
    /**
     * Weights mapped to each input neuron.
     */
    unordered_map<NeuronPtr, double> input_weights;
    /**
     * Propagated input values mapped to each input neuron.
     */
    unordered_map<NeuronPtr, double> inputs;
    /**
     * Set of output neurons.
     */
    unordered_set<NeuronPtr> outputs;
    /**
     * Error (or delta) value is mapped to each output neuron.
     */
    unordered_map<NeuronPtr, double> errors;
    /**
     * Activation of the neuron caused by the latest forward propagation.
     */
    double activation;
    /**
     * Learning rate. The higher, the faster the neuron learns - if too high, optimum may not found.
     */
    double learning_rate;
	/**
     * Regularization term. Force weights to stay lower, serve generalization.
     */
    double regularization;

private:
	/**
     * Connects the given 'neuro' neuron as an input of this neuron.
     * @param neuro
     */
    virtual void connect_input(NeuronPtr& neuro);

	/**
     * Connects the given 'neuro' neuron as an output of this neuron.
     * @param neuro
     */
	void Neuron::connect_output(NeuronPtr& neuro);
};

/**
 * Creates a neuron and a (shared) pointer to it, so it can be connected to other neurons.
 */
template <typename NeuronType>
std::shared_ptr<NeuronType> make_neuron(double learning_rate_ = Neuron::def_learning_rate, double regularization_ = Neuron::def_regularization)
{
	return std::make_shared<NeuronType>(learning_rate_, regularization_);
}

/**
 * Creates a (shared) pointer to neuron, so it can be connected to other neurons.
 * Beware, could be dangerous if neuron is deallocated before all pointers to it are destroyed! Do not mix with other make_neuron; only use this if necessary!
 */
/*template <typename NeuronType>
std::shared_ptr<NeuronType> make_neuron(NeuronType& neuron)
{
	static unordered_map<NeuronType*, shared_ptr<NeuronType>> bare_ptr_shared_ptr_map;
	auto ins = bare_ptr_shared_ptr_map.insert(std::make_pair(&neuron, shared_ptr<NeuronType>(&neuron)));

	return (ins.first)->second;
}*/

}

#endif //_NEURON_H