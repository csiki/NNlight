/**
 * Project Untitled
 */


#include "Neuron.h"

/**
 * Neuron implementation
 * 
 * Represents a (hidden) neuron, that has both inputs and outputs as neurons. Implements a learning method that alters the input weights as the function of propagated input and backpropagated error. Use sigmoid as nonlinear function.
 */


/**
 * Creates a neuron instance with the given learning rate.
 * @param learning_rate_
 */
void Neuron::Neuron(double learning_rate_) {

}

/**
 * Induces input, activating the neuron.
 * @param from
 * @param act
 */
virtual void Neuron::propagate(NeuronPtr from, double act) {

}

/**
 * Forces the neuron to alter its weight (that is connects this to the 'from' neuron) if necessary.
 * @param from
 * @param err
 */
virtual void Neuron::backpropagate(NeuronPtr from, double err) {

}

/**
 * Connects the given 'neuro' neuron as an input of this neuron. Also this neuron is added as an output neuron of the given neuron.
 * @param neuro
 */
virtual void Neuron::connect_input(Neuron& neuro) {

}