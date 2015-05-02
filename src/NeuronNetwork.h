/**
 * Project NNlight
 */

#ifndef _NEURONNETWORK_H
#define _NEURONNETWORK_H

#include <vector>
#include <iostream>
#include <set>
#include <limits>
#include <algorithm>
#include <iterator>
#include <deque>
#include <string>
#include "Neuron.h"
#include "OutputNeuron.h"
#include "InputNeuron.h"
#include "ActivationOutOfBoundsException.h"

using std::vector;
using std::istream;
using std::ostream;
using std::set;
using std::deque;
using std::string;

namespace NNlight {

class NeuronNetwork {

	class NNSettings // TODO doc
	{
		friend NeuronNetwork;
	public:
		NNSettings();
		void restart_training_if_stuck(bool do_restart, double restart_threshold_ = 0.1, size_t max_nrestart = 10);
		void set_max_num_of_epochs(size_t max_nepoch_);

	private:
		NNSettings& operator=(const NNSettings& _) {}
		bool restart_if_high_error;
		double restart_threshold;
		size_t max_nrestart;
		size_t max_nepoch;
		// TODO
	};

public: 

	NNSettings settings;

    /**
     * Creates a network of neurons, initially without the neurons. Neurons can be added after creation.
     */
    NeuronNetwork();
    
    /**
     * Adds an input neuron to the network. The more input neurons are added, the more input values are needed for training and testing. The number of input neurons determines the dimension of the input data.
     * @param neuro
     */
    void add_neuron(InputNeuron& neuro);
    
    /**
     * Adds an output neuron to the network. Number of added output neurons equals to the number of output values the network provides.
     * @param neuro
     */
    void add_neuron(OutputNeuron& neuro);
    
    /**
     * Adds a hidden neuron to the network. All neurons added need to be reached from each other via weighted edges. Input or output neurons may not added here.
     * @param neuro
     */
    void add_neuron(Neuron& neuro);
    
    /**
     * Adds the shared pointer of an input neuron to the network. The more input neurons are added, the more input values are needed for training and testing. The number of input neurons determines the dimension of the input data.
     * @param neuroptr
     */
    void add_neuron(InputNeuronPtr neuroptr);
    
    /**
     * Adds the shared pointer of an output neuron to the network. Number of added output neurons equals to the number of output values the network provides.
     * @param neuroptr
     */
    void add_neuron(OutputNeuronPtr neuroptr);
    
    /**
     * Adds the shared pointer of a hidden neuron to the network. All neurons added need to be reached from each other via weighted edges. Input or output neurons may not added here.
     * @param neuroptr
     */
    void add_neuron(NeuronPtr neuroptr);
    
    /**
     * Force the interconnected neurons to learn in a supervised way by the given input and desired output. Use this overload if the input is already separated from the output.
     * @param input
     * @param desired_output
     * @param log_stream
     * @param train_ratio
     */
    void train(vector<vector<double>> input, vector<vector<double>> desired_output, ostream& log_stream, double train_ratio, bool batch_mode = false);
    
    /**
     * Force the interconnected neurons to learn in a supervised way by the given input and desired output. Use this overload if both the input and desired output values are contained in a stream (file, console, etc.). 
     * @param train_stream
     * @param log_stream
     * @param train_ratio
     */
    void train(istream& train_stream, ostream& log_stream, double train_ratio, bool batch_mode = false);
    
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
    void test(const vector<double>& input, ostream& output_stream, string delimiter = " ");
    
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
    void test(istream& input_stream, ostream& output_stream, string delimiter = " ");

	/**
	 * Randomize a new value for all weights (including the bias) in the range of [Neuron::def_lower_bound, Neuron::def_upper_bound) for the whole network. Also clears Neuron::inputs and Neuron::errors.
	 */
	void reset();

private: 
    /**
     * Default value of the ratio of training samples to all the samples. 1-<this> means the test ratio.
     */
    static double def_train_test_ratio;
	/**
     * Default value of the maximum number of epochs permited for a training.
     */
    static size_t def_max_epoch;
	/**
	 * Epsilon value: if the error change is between this value and 0, than no more learning iteration is conducted.
	 */
	static double err_eps;
	/**
	 * After test_err_increase_threshold number of test error increase iteration by iteration, the network is restored to the state when it started increasing.
	 */
	static size_t test_err_increase_threshold;
    /**
     * All neurons in network.
     */
    set<NeuronPtr> neurons;
    /**
     * Output neurons in network.
     */
    set<OutputNeuronPtr> outputs;
    /**
     * Input neurons in network.
     */
    set<InputNeuronPtr> inputs;
};

template <typename order_iterator, typename value_iterator>
void reorder(order_iterator order_begin, order_iterator order_end, value_iterator v);

}

#endif //_NEURONNETWORK_H