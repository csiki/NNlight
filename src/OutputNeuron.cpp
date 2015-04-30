/**
 * Project NNlight
 */


#include "OutputNeuron.h"
#include <iostream> // TODO rm

/**
 * OutputNeuron implementation
 * 
 * Represents a neuron that output is not connected to any other neuron, but it's activation (output) is readable.
 */

namespace NNlight {

/**
 * Creates a neuron instance with the given learning rate.
 * @param learning_rate_
 */
OutputNeuron::OutputNeuron(double learning_rate_, double regularization_)
	: Neuron(learning_rate_, regularization_)
{}

/**
 * Returns the output activation of the neuron.
 * @return double
 */
double OutputNeuron::get_activation()
{
    return activation;
}

/**
 * Forces the neuron to alter its weight if necessary. More simple calculation of delta value compared to overriden function.
 * @param from disregarded
 * @param err
 */
void OutputNeuron::backpropagate(NeuronPtr from, double delta)
{
	// adjust bias & input weights
	biasweight += learning_rate * delta;
	for (auto& in : input_weights)
	{
		//auto grad = in.first->activation * delta + regularization * in.second;
		auto grad = in.first->activation * delta * activation * (1 - activation) + regularization * in.second;
		in.second += learning_rate * grad;
	}

	// bacpropage error further
	for (auto& in : input_weights)
		in.first->backpropagate(shared_from_this(), delta * in.second); // multiplied by input weight
}

}