/**
 * Project Untitled
 */


#ifndef _NEURON_H
#define _NEURON_H

class Neuron {
public: 
    
    /**
     * Creates a neuron instance with the given learning rate.
     * @param learning_rate_
     */
    void Neuron(double learning_rate_);
    
    /**
     * Induces input, activating the neuron.
     * @param from
     * @param act
     */
    virtual void propagate(NeuronPtr from, double act) = 0;
    
    /**
     * Forces the neuron to alter its weight (that is connects this to the 'from' neuron) if necessary.
     * @param from
     * @param err
     */
    virtual void backpropagate(NeuronPtr from, double err) = 0;
    
    /**
     * Connects the given 'neuro' neuron as an input of this neuron. Also this neuron is added as an output neuron of the given neuron.
     * @param neuro
     */
    virtual void connect_input(Neuron& neuro) = 0;
private: 
    /**
     * Default learning rate.
     */
    static double def_learning_rate;
    /**
     * Weight of the bias input neuron.
     */
    double biasweight;
    /**
     * Weights mapped to each input neuron.
     */
    map<NeuronPtr, double> input_weights;
    /**
     * Propagated input values mapped to each input neuron.
     */
    map<NeuronPtr, double> inputs;
    /**
     * Set of output neurons.
     */
    set<NeuronPtr> outputs;
    /**
     * Error (or delta) value is mapped to each output neuron.
     */
    map<NeuronPtr, double> errors;
    /**
     * Activation of the neuron caused by the latest forward propagation.
     */
    double activation;
    /**
     * Learning rate. The higher, the faster the neuron learns - if too high, optimum may not found.
     */
    double learning_rate;
};

#endif //_NEURON_H