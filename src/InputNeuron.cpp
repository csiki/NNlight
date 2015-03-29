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
InputNeuron::InputNeuron(double learning_rate_) : Neuron(learning_rate_)
{

}

/**
 * Activates the neuron with the given input. Forces the neuron to forward propagate this value to its outputs.
 * @param input
 */
void InputNeuron::feed(double input) 
{

}

}