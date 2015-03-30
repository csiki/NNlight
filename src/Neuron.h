/**
 * Project NNlight
 */

#ifndef _NEURON_H
#define _NEURON_H

#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <random>
#include <iostream>
#include <numeric>

using std::unordered_set;
using std::unordered_map;
using std::shared_ptr;

namespace NNlight {

class Neuron;
typedef shared_ptr<Neuron> NeuronPtr;

class Neuron {
public:
	friend class OutputNeuron;
    
    /**
     * Creates a neuron instance with the given learning rate.
     * @param learning_rate_
     */
	Neuron(double learning_rate_ = def_learning_rate, double regularization_ = def_regularization);
    
    /**
     * Induces input, activating the neuron.
     * @param from
     * @param act
     */
    virtual void propagate(NeuronPtr from, double act);
    
    /**
     * Forces the neuron to alter its weight (that is connects this to the 'from' neuron) if necessary.
     * @param from
     * @param err
     */
    virtual void backpropagate(NeuronPtr from, double err);
    
    /**
     * Connects the given 'neuro' neuron as an input of this neuron. Also this neuron is added as an output neuron of the given neuron.
     * @param neuro
     */
    virtual void connect_input(Neuron& neuro);

	/**
     * Default learning rate.
     */
    static double def_learning_rate;

	/**
     * Default regularization term. Force weights to stay lower, serve generalization.
     */
	static double def_regularization;

protected:
    /**
     * Weight of the bias input neuron.
     */
    double biasweight;
    /**
     * Weights mapped to each input neuron.
     */
    unordered_map<NeuronPtr, double> input_weights;
    /**
     * Propagated input values mapped to each input neuron.
     */
    unordered_map<NeuronPtr, double> inputs;
    /**
     * Set of output neurons.
     */
    unordered_set<NeuronPtr> outputs;
    /**
     * Error (or delta) value is mapped to each output neuron.
     */
    unordered_map<NeuronPtr, double> errors;
    /**
     * Activation of the neuron caused by the latest forward propagation.
     */
    double activation;
    /**
     * Learning rate. The higher, the faster the neuron learns - if too high, optimum may not found.
     */
    double learning_rate;
	/**
     * Regularization term. Force weights to stay lower, serve generalization.
     */
    double regularization;
};

}

#endif //_NEURON_H