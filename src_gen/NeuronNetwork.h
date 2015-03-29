/**
 * Project Untitled
 */


#ifndef _NEURONNETWORK_H
#define _NEURONNETWORK_H

#include "Neuron.h"
#include "OutputNeuron.h"
#include "InputNeuron.h"


class NeuronNetwork {
public: 
    
    /**
     * Creates a network of neurons, initially without the neurons. Neurons can be added after creation.
     */
    void NeuronNetwork();
    
    /**
     * Adds an input neuron to the network. The more input neurons are added, the more input values are needed for training and testing. The number of input neurons determines the dimension of the input data.
     * @param neuro
     */
    void add_input_neuron(InputNeuron& neuro);
    
    /**
     * Adds an output neuron to the network. Number of added output neurons equals to the number of output values the network provides.
     * @param neuro
     */
    void add_output_neuron(OutputNeuron& neuro);
    
    /**
     * Adds a hidden neuron to the network. All neurons added need to be reached from each other via weighted edges. Input or output neurons may not added here.
     * @param neuro
     */
    void add_hidden_neuron(Neuron& neuro);
    
    /**
     * Adds the shared pointer of an input neuron to the network. The more input neurons are added, the more input values are needed for training and testing. The number of input neurons determines the dimension of the input data.
     * @param neuroptr
     */
    void add_input_neuron(InputNeuronPtr neuroptr);
    
    /**
     * Adds the shared pointer of an output neuron to the network. Number of added output neurons equals to the number of output values the network provides.
     * @param neuroptr
     */
    void add_output_neuron(OutputNeuronPtr neuroptr);
    
    /**
     * Adds the shared pointer of a hidden neuron to the network. All neurons added need to be reached from each other via weighted edges. Input or output neurons may not added here.
     * @param neuroptr
     */
    void add_hidden_neuron(NeuronPtr neuroptr);
    
    /**
     * Force the interconnected neurons to learn in a supervised way by the given input and desired output. Use this overload if the input is already separated from the output.
     * @param input
     * @param desired_output
     * @param log_stream
     * @param train_ratio
     */
    void train(const vector<vector<double>>& input, const vector<double>& desired_output, ostream& log_stream, double train_ratio);
    
    /**
     * Force the interconnected neurons to learn in a supervised way by the given input and desired output. Use this overload if both the input and desired output values are contained in a stream (file, console, etc.). 
     * @param train_stream
     * @param log_stream
     * @param train_ratio
     */
    void train(istream& train_stream, ostream& log_stream, double train_ratio);
    
    /**
     * Activates the input neurons of the network with the given input and reads the output of the output neurons. Use this overload if the input is already put in a vector and the output is expected in a vector. The output vector is filled with as many elements as the number of output neurons in the network.
     * @param input
     * @param output
     */
    void test(const vector<double>& input, vector<double>& output);
    
    /**
     * Activates the input neurons of the network with the given input and reads the output of the output neurons. Use this overload if the input is already put in a vector and the output is expected to be written on a stream.
     * @param input
     * @param output_stream
     */
    void test(const vector<double>& input, ostream& output_stream);
    
    /**
     * Activates the input neurons of the network with the given input and reads the output of the output neurons. Use this overload if the input is on a stream and the output is expected in a vector. The output vector is filled with as many elements as the number of output neurons in the network.
     * @param input_stream
     * @param output
     */
    void test(istream& input_stream, vector<double>& output);
    
    /**
     * Activates the input neurons of the network with the given input and reads the output of the output neurons. Use this overload if the input is on a stream and the output is expected to be written on a stream as well.
     * @param input_stream
     * @param output_stream
     */
    void test(istream& input_stream, ostream& output_stream);
private: 
    /**
     * Default value of the ratio of training samples to all the samples. 1-<this> means the test ratio.
     */
    static double def_train_test_ratio;
    /**
     * All neurons in network.
     */
    Vector<Neuron> neurons;
    /**
     * Output neurons in network.
     */
    Vector<OutputNeuron> outputs;
    /**
     * Input neurons in network.
     */
    Vector<InputNeuron> inputs;
};

#endif //_NEURONNETWORK_H