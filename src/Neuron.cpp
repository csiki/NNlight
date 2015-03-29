/**
 * Project NNlight
 */

#include "Neuron.h"

/**
 * Neuron implementation
 * 
 * Represents a (hidden) neuron, that has both inputs and outputs as neurons. Implements a learning method that alters the input weights as the function of propagated input and backpropagated error. Use sigmoid as nonlinear function.
 */

namespace NNlight {

double Neuron::def_learning_rate = 0.2;
double Neuron::def_regularization = 0.2;

/**
 * Creates a neuron instance with the given learning rate.
 * @param learning_rate_
 */
Neuron::Neuron(double learning_rate_)
	: learning_rate(learning_rate_), activation(0.0)
{
	// rand biasweight
	std::uniform_real_distribution<double> distr(0.0, 2.0);
	std::random_device rand_dev;
	biasweight = distr(rand_dev);
	std::cout << biasweight;
}

/**
 * Induces input, activating the neuron.
 * @param from
 * @param act
 */
void Neuron::propagate(NeuronPtr from, double act)
{
	if (input_weights.find(from) == input_weights.end())
		throw std::exception("Neuron that propagated potential is not connected as input!");
	if (inputs.find(from) != inputs.end())
		throw std::exception("Activation is already propagated from given neuron!");

	inputs.insert(std::make_pair(from, act));
	if (inputs.size() == input_weights.size()) // all input neuron propagated their activation
	{
		// calculate activation
		auto& tmp_weights = input_weights;
		activation = std::accumulate(inputs.begin(), inputs.end(), 0.0,
			[&tmp_weights] (double acc, const std::pair<NeuronPtr, double>& in) {
				return acc + tmp_weights[in.first] * in.second;
		});
		activation = 1.0 / (1 + std::exp(-activation)); // sigmoid nonlinear function

		// remove all inputs
		inputs.clear();

		// propage activation further
		for (auto& out : outputs)
			out->propagate(NeuronPtr(this), activation);
	}
}

/**
 * Forces the neuron to alter its weight (that is connects this to the 'from' neuron) if necessary.
 * @param from
 * @param err
 */
void Neuron::backpropagate(NeuronPtr from, double err)
{
	if (outputs.find(from) == outputs.end())
		throw std::exception("Neuron that backpropagated potential is not connected as output!");
	if (errors.find(from) != errors.end())
		throw std::exception("Activation is already backpropagated from given neuron!");

	errors.insert(std::make_pair(from, err));
	if (errors.size() == outputs.size()) // all output neuron backpropagated their activation
	{
		// calculate delta value
		double delta = std::accumulate(errors.begin(), errors.end(), 0.0,
			[] (double acc, const std::pair<NeuronPtr, double>& out) {
				return acc + out.second; // already multiplied by output weight
		});
		delta *= activation * (1.0 - activation); // multiplied by sigmoid derivate of activation

		// adjust input weights
		for (auto& in : input_weights)
		{
			auto grad = in.first->activation * delta + regularization * in.second;
			in.second -= learning_rate * grad;
		}

		// remove all errors
		errors.clear();

		// bacpropage error further
		for (auto& in : input_weights)
			in.first->backpropagate(NeuronPtr(this), delta * in.second); // multiplied by input weight
	}
}

/**
 * Connects the given 'neuro' neuron as an input of this neuron. Also this neuron is added as an output neuron of the given neuron.
 * @param neuro
 */
void Neuron::connect_input(Neuron& neuro)
{

}

}