/**
 * Project Untitled
 */


#include "NeuronNetwork.h"

/**
 * NeuronNetwork implementation
 * 
 * Represents a network of neurons. Implements a training and testing methods to force the neurons to learn and to evaluate the output of the network (for a given input), respectively.
 */


/**
 * Creates a network of neurons, initially without the neurons. Neurons can be added after creation.
 */
void NeuronNetwork::NeuronNetwork() {

}

/**
 * Adds an input neuron to the network. The more input neurons are added, the more input values are needed for training and testing. The number of input neurons determines the dimension of the input data.
 * @param neuro
 */
void NeuronNetwork::add_input_neuron(InputNeuron& neuro) {

}

/**
 * Adds an output neuron to the network. Number of added output neurons equals to the number of output values the network provides.
 * @param neuro
 */
void NeuronNetwork::add_output_neuron(OutputNeuron& neuro) {

}

/**
 * Adds a hidden neuron to the network. All neurons added need to be reached from each other via weighted edges. Input or output neurons may not added here.
 * @param neuro
 */
void NeuronNetwork::add_hidden_neuron(Neuron& neuro) {

}

/**
 * Adds the shared pointer of an input neuron to the network. The more input neurons are added, the more input values are needed for training and testing. The number of input neurons determines the dimension of the input data.
 * @param neuroptr
 */
void NeuronNetwork::add_input_neuron(InputNeuronPtr neuroptr) {

}

/**
 * Adds the shared pointer of an output neuron to the network. Number of added output neurons equals to the number of output values the network provides.
 * @param neuroptr
 */
void NeuronNetwork::add_output_neuron(OutputNeuronPtr neuroptr) {

}

/**
 * Adds the shared pointer of a hidden neuron to the network. All neurons added need to be reached from each other via weighted edges. Input or output neurons may not added here.
 * @param neuroptr
 */
void NeuronNetwork::add_hidden_neuron(NeuronPtr neuroptr) {

}

/**
 * Force the interconnected neurons to learn in a supervised way by the given input and desired output. Use this overload if the input is already separated from the output.
 * @param input
 * @param desired_output
 * @param train_ratio
 */
void NeuronNetwork::train(const vector<vector<double>>& input, const vector<double>& desired_output, double train_ratio) {

}

/**
 * Force the interconnected neurons to learn in a supervised way by the given input and desired output. Use this overload if both the input and desired output values are contained in a stream (file, console, etc.). 
 * @param train_stream
 * @param train_ratio
 */
void NeuronNetwork::train(istream& train_stream, double train_ratio) {

}

/**
 * Activates the input neurons of the network with the given input and reads the output of the output neurons. Use this overload if the input is already put in a vector and the output is expected in a vector. The output vector is filled with as many elements as the number of output neurons in the network.
 * @param input
 */
void NeuronNetwork::test(const vector<double>& input) {

}

/**
 * Activates the input neurons of the network with the given input and reads the output of the output neurons. Use this overload if the input is already put in a vector and the output is expected to be written on a stream.
 * @param input
 */
void NeuronNetwork::test(const vector<double>& input) {

}

/**
 * Activates the input neurons of the network with the given input and reads the output of the output neurons. Use this overload if the input is on a stream and the output is expected in a vector. The output vector is filled with as many elements as the number of output neurons in the network.
 * @param input_stream
 */
void NeuronNetwork::test(istream& input_stream) {

}

/**
 * Activates the input neurons of the network with the given input and reads the output of the output neurons. Use this overload if the input is on a stream and the output is expected to be written on a stream as well.
 * @param input_stream
 */
void NeuronNetwork::test(istream& input_stream) {

}