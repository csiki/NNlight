/**
 * Project NNlight
 */

#ifndef _OUTPUTNEURON_H
#define _OUTPUTNEURON_H

#include "Neuron.h"

namespace NNlight {

class OutputNeuron: public Neuron {
public: 
    
    /**
     * Creates a neuron instance with the given learning rate.
     * @param learning_rate_
     */
	OutputNeuron(double learning_rate_ = def_learning_rate, double regularization_ = def_regularization);
    
    /**
     * Returns the output activation of the neuron.
     */
    double get_activation();
    
    /**
     * Forces the neuron to alter its weight if necessary. More simple calculation of delta value compared to overriden function.
     * @param from disregarded
     * @param delta
     */
    void backpropagate(NeuronPtr from, double delta);
};

typedef shared_ptr<OutputNeuron> OutputNeuronPtr;

}

#endif //_OUTPUTNEURON_H