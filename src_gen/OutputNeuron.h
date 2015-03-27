/**
 * Project Untitled
 */


#ifndef _OUTPUTNEURON_H
#define _OUTPUTNEURON_H

#include "Neuron.h"


class OutputNeuron: public Neuron {
public: 
    
    /**
     * Creates a neuron instance with the given learning rate.
     * @param learning_rate_
     */
    void OutputNeuron(double learning_rate_);
    
    /**
     * Returns the output activation of the neuron.
     */
    double get_activation();
};

#endif //_OUTPUTNEURON_H