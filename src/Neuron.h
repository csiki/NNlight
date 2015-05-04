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
#include <cmath>
#include <functional>
#include "ActivationOutOfBoundsException.h"

using std::unordered_set;
using std::unordered_map;
using std::shared_ptr;

namespace NNlight {

class Neuron;
typedef shared_ptr<Neuron> NeuronPtr;
typedef std::function<double(double)> ActivationFun;

class Neuron : public std::enable_shared_from_this<Neuron>
{
public:
	// from http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.27.7876&rep=rep1&type=pdf
	// TODO doc
	class Rprop
	{
	public:
		const static double def_delta0, def_deltamax, def_incr_factor, def_decr_factor;

		Rprop() {}
		Rprop(const unordered_map<NeuronPtr, double>& inputs, double delta0_ = def_delta0, double deltamax_ = def_deltamax, double incr_factor_ = def_incr_factor, double decr_factor_ = def_decr_factor);
		double operator()(const NeuronPtr& weight_of_input, double grad);
		double operator()(double bias_grad);
		void reset();

	private:
		double delta0, deltamax;
		double incr_factor, decr_factor;
		double bias_prev_grad, bias_delta;
		unordered_map<NeuronPtr, double> deltas;
		unordered_map<NeuronPtr, double> prev_grads;
	};

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
	 * Connects two given layers of neurons. Each neuron from layer_from is connected to each neuron of layer_to.
	 * @param layer_from
	 * @param layer_to
	 */
	template <typename NeuronPtrTypeFrom, typename NeuronPtrTypeTo, size_t N, size_t M>
	static void connect_layers(std::array<NeuronPtrTypeFrom, N>& layer_from, std::array<NeuronPtrTypeTo, M>& layer_to)
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
	Neuron(double learning_rate_ = def_learning_rate, double regularization_ = def_regularization); // TODO std::pair<ActivationFun, ActivationFun> act_fun_ = def_activation_fun);
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
	 * Randomize a new value for all weights (including the bias) in the range of [lower_bound, upper_bound). Also clears inputs, and errors.
	 * @param lower_bound
	 * @param upper_bound
	 */
	void reset(double lower_bound = def_weight_lower_bound, double upper_bound = def_weight_upper_bound);

	/**
	 * Activates the use of the default gradient-descent weight update method. Set by default.
	 * @param learning_rate_
	 * @param regularization_
	 */
	void use_default_backpropation(double learning_rate_, double regularization_);

	/**
	 * Activates the use of the advanced gradient-descent weight update method. Stick to the default parameters.
	 * Call only after the input neurons are added.
	 * @param delta0
	 * @param deltamax
	 * @param incr_factor
	 * @param decr_factor
	 */
	void use_resilient_backpropagation(double delta0 = Rprop::def_delta0, double deltamax = Rprop::def_deltamax,
		double incr_factor = Rprop::def_incr_factor, double decr_factor = Rprop::def_decr_factor);

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

	/**
     * Predicate for the calculation of Rprop weight updates.
     */
	Rprop rprop;

	/*
	 * Indicates if the Rprop advanced adaptive weight update is in use.
	 */
	bool use_rprop;

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
     * Default activation function - sigmoid logistic function.
     */
	static std::function<double(double)> def_activation_fun;

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
 * @param learning_rate_
 * @param regularization_ regularization parameter, propotional to the generalization of the network
 */
template <typename NeuronType>
std::shared_ptr<NeuronType> make_neuron(double learning_rate_ = Neuron::def_learning_rate, double regularization_ = Neuron::def_regularization)
{
	return std::make_shared<NeuronType>(learning_rate_, regularization_);
}

/**
 * Creates a neuron layer and a (shared) pointer to each neuron in it, so they can be connected to other neurons.
 * @param learning_rate_
 * @param regularization_ regularization parameter, propotional to the generalization of the network
 */
template <typename NeuronType, size_t N>
std::array<std::shared_ptr<NeuronType>, N> make_layer(double learning_rate_ = Neuron::def_learning_rate, double regularization_ = Neuron::def_regularization)
{
	std::array<std::shared_ptr<NeuronType>, N> layer;
	for (auto& neur : layer)
		neur = make_neuron<NeuronType>(learning_rate_, regularization_);
	return layer;
}

}

#endif //_NEURON_H