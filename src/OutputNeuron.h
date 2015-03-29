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
	OutputNeuron(double learning_rate_ = def_learning_rate);
    
    /**
     * Returns the output activation of the neuron.
     */
    double get_activation();
};

typedef shared_ptr<OutputNeuron> OutputNeuronPtr;

}

#endif //_OUTPUTNEURON_H