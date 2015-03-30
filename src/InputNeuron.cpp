/**
 * Project NNlight
 */


#include "InputNeuron.h"

/**
 * InputNeuron implementation
 * 
 * Represents a neuron that has no input connected to any other neuron, but an input value can be fed directly to it.
 */

namespace NNlight {

/**
 * Creates a input neuron instance with the given learning rate.
 * @param learning_rate_
 */
InputNeuron::InputNeuron(double learning_rate_, double regularization_)
	: Neuron(learning_rate_, regularization_)
{}

/**
 * Activates the neuron with the given input. Forces the neuron to forward propagate this value to its outputs.
 * @param input
 */
void InputNeuron::feed(double input) 
{
	activation = input;
	for (auto& out : outputs)
		out->propagate(NeuronPtr(this), activation);
}

/**
 * Induces input, activating the neuron. Calls function feed.
 * @param from disregarded
 * @param act
 */
void InputNeuron::propagate(NeuronPtr from, double act)
{
	feed(act);
}

/**
 * As there is no input weights, nothing happens.
 * @param from
 * @param err
 */
void InputNeuron::backpropagate(NeuronPtr from, double err) {}

}