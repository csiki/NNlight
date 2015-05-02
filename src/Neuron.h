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
#include <array>

using std::unordered_set;
using std::unordered_map;
using std::shared_ptr;

namespace NNlight {

class Neuron;
typedef shared_ptr<Neuron> NeuronPtr;

class Neuron : public std::enable_shared_from_this<Neuron> {
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

	template <size_t N, size_t M>
	static void connect_layers(std::array<NeuronPtr, N>& layer_from, std::array<NeuronPtr, M>& layer_to)
	{
		for (auto& from : layer_from)
			for (auto& to : layer_to)
				connect(from, to);
	}

	/**
	 * Sets the lower and upper bounds of randomized initial weight values.
	 * @param lower_bound
	 * @param upper_bound
	 */
	static void set_initial_weight_bounds(double lower_bound, double upper_bound);

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
	 * Randomize a new value for all weights (including the bias) in the range of [lower_bound, upper_bound).
	 * @param lower_bound
	 * @param upper_bound
	 */
	void reset_weights(double lower_bound = def_weight_lower_bound, double upper_bound = def_weight_upper_bound);

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
     * Default value of weights initial lower bound.
     */
	static double def_weight_lower_bound;
	/**
     * Default value of weights initial upper bound.
     */
	static double def_weight_upper_bound;

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

}

#endif //_NEURON_H