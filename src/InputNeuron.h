/**
 * Project NNlight
 */

#ifndef _INPUTNEURON_H
#define _INPUTNEURON_H

#include "Neuron.h"

namespace NNlight {

class InputNeuron: public Neuron
{
public: 
    
    /**
     * Creates a input neuron instance with the given learning rate.
     * @param learning_rate_
     */
    InputNeuron(double learning_rate_ = def_learning_rate, double regularization_ = def_regularization);
    
    /**
     * Activates the neuron with the given input. Forces the neuron to forward propagate this value to its outputs.
     * @param input
     */
    void feed(double input);

	/**
     * Induces input, activating the neuron. Calls function feed.
     * @param from disregarded
     * @param act
     */
    void propagate(NeuronPtr from, double act);
    
    /**
     * As there is no input weights, nothing happens.
	 * @param from
     * @param err
     */
    void backpropagate(NeuronPtr from, double err);
};

typedef shared_ptr<InputNeuron> InputNeuronPtr;

}

#endif //_INPUTNEURON_H