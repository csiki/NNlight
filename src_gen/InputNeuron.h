/**
 * Project Untitled
 */


#ifndef _INPUTNEURON_H
#define _INPUTNEURON_H

#include "Neuron.h"


class InputNeuron: public Neuron {
public: 
    
    /**
     * Creates a input neuron instance with the given learning rate.
     * @param learning_rate_
     */
    void InputNeuron(double learning_rate_);
    
    /**
     * Activates the neuron with the given input. Forces the neuron to forward propagate this value to its outputs.
     * @param input
     */
    void feed(double input);
};

#endif //_INPUTNEURON_H