/**
 * Project NNlight
 */


#include "OutputNeuron.h"

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
OutputNeuron::OutputNeuron(double learning_rate_) : Neuron(learning_rate_)
{

}

/**
 * Returns the output activation of the neuron.
 * @return double
 */
double OutputNeuron::get_activation()
{
    return 0.0;
}

}