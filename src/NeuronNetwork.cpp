/**
 * Project NNlight
 */

#include "NeuronNetwork.h"

/**
 * NeuronNetwork implementation
 * 
 * Represents a network of neurons. Implements a training and testing methods to force the neurons to learn and to evaluate the output of the network (for a given input), respectively.
 */

namespace NNlight {

double NeuronNetwork::def_train_test_ratio = 0.8;
double NeuronNetwork::acc_eps = 1e-4;

/**
 * Creates a network of neurons, initially without the neurons. Neurons can be added after creation.
 */
NeuronNetwork::NeuronNetwork() {}

/**
 * Adds an input neuron to the network. The more input neurons are added, the more input values are needed for training and testing. The number of input neurons determines the dimension of the input data.
 * @param neuro
 */
void NeuronNetwork::add_input_neuron(InputNeuron& neuro)
{
	auto ins = neurons.insert(std::make_shared<InputNeuron>(neuro));
	if (!ins.second)
		throw std::exception("Input neuron is already added!");

	inputs.insert(std::make_shared<InputNeuron>(neuro));
}

/**
 * Adds an output neuron to the network. Number of added output neurons equals to the number of output values the network provides.
 * @param neuro
 */
void NeuronNetwork::add_output_neuron(OutputNeuron& neuro)
{
	auto ins = neurons.insert(std::make_shared<OutputNeuron>(neuro));
	if (!ins.second)
		throw std::exception("Output neuron is already added!");

	outputs.insert(std::make_shared<OutputNeuron>(neuro));
}

/**
 * Adds a hidden neuron to the network. All neurons added need to be reached from each other via weighted edges. Input or output neurons may not added here.
 * @param neuro
 */
void NeuronNetwork::add_hidden_neuron(Neuron& neuro)
{
	auto ins = neurons.insert(std::make_shared<Neuron>(neuro));
	if (!ins.second)
		throw std::exception("Hidden neuron is already added!");
}

/**
 * Adds the shared pointer of an input neuron to the network. The more input neurons are added, the more input values are needed for training and testing. The number of input neurons determines the dimension of the input data.
 * @param neuroptr
 */
void NeuronNetwork::add_input_neuron(InputNeuronPtr neuroptr)
{
	auto ins = neurons.insert(neuroptr);
	if (!ins.second)
		throw std::exception("Input neuron is already added!");

	inputs.insert(neuroptr);
}

/**
 * Adds the shared pointer of an output neuron to the network. Number of added output neurons equals to the number of output values the network provides.
 * @param neuroptr
 */
void NeuronNetwork::add_output_neuron(OutputNeuronPtr neuroptr)
{
	auto ins = neurons.insert(neuroptr);
	if (!ins.second)
		throw std::exception("Output neuron is already added!");
	
	outputs.insert(neuroptr);
}

/**
 * Adds the shared pointer of a hidden neuron to the network. All neurons added need to be reached from each other via weighted edges. Input or output neurons may not added here.
 * @param neuroptr
 */
void NeuronNetwork::add_hidden_neuron(NeuronPtr neuroptr)
{
	auto ins = neurons.insert(neuroptr);
	if (!ins.second)
		throw std::exception("Hidden neuron is already added!");
}

/**
 * Force the interconnected neurons to learn in a supervised way by the given input and desired output. Use this overload if the input is already separated from the output.
 * @param input
 * @param desired_output
 * @param log_stream
 * @param train_ratio
 */
void NeuronNetwork::train(const vector<vector<double>>& input, const vector<vector<double>>& desired_output, ostream& log_stream, double train_ratio)
{
	double delta_acc = std::numeric_limits<double>::max(); // accuracy change iteration by iteration
	// create and fill shuffle indices, so input and desired output can be shuffled at the same time
	vector<size_t> shuffle_indices(input.size());
	std::iota(shuffle_indices.begin(), shuffle_indices.end(), 0);
	// separate train and test, input and desired output vectors
	auto train_in_beg = input.begin();
	auto train_in_end = input.begin() + static_cast<size_t>(input.size() * train_ratio);
	auto train_dout_beg = desired_output.begin();
	auto train_dout_end = desired_output.begin() + static_cast<size_t>(desired_output.size() * train_ratio);
	auto test_in_beg = input.begin() + static_cast<size_t>(input.size() * train_ratio);
	auto test_in_end = input.end();
	auto test_dout_beg = desired_output.begin() + static_cast<size_t>(desired_output.size() * train_ratio);
	auto test_dout_end = desired_output.end();
	// create performance vectors to be able to average over errors of a training set
	vector<double> train_perf(input.size() * train_ratio);
	vector<double> test_perf(input.size() - input.size() * train_ratio);

	while (!(std::abs(delta_acc) < acc_eps)) // while the change is significant
	{
		// shuffle train inputs & desired outputs
		// TODO

		// iterate over all samples
		for (size_t sample_index = 0; sample_index < input.size() * train_ratio; ++sample_index) // TODO iterátorokra cseréld
		{
			// forward propagation
			size_t neur_index = 0;
			for (auto& in : inputs) in->feed(input[sample_index][neur_index++]);
			
			// backward propagation
			vector<double> err;
			err.reserve(outputs.size());
			std::transform(outputs.begin(), outputs.end(), desired_output[sample_index].begin(), std::back_inserter(err),
				[] (const OutputNeuronPtr& neur, const double& d_out) {
					return std::pow(neur->get_activation() - d_out, 2.0); // MSE
			});
			neur_index = 0;
			for (auto& out : outputs) out->backpropagate(nullptr, err[neur_index++]);

			// update training performance by averaging over errors
			train_perf[sample_index] = std::accumulate(err.begin(), err.end(), 0.0);
			train_perf[sample_index] /= err.size();

		}
		
		// check test performance
		for (size_t sample_index = input.size() * train_ratio; sample_index < input.size(); ++sample_index) // TODO iterátorokra cseréld
		{
			// TODO
		}
	}
}

/**
 * Force the interconnected neurons to learn in a supervised way by the given input and desired output. Use this overload if both the input and desired output values are contained in a stream (file, console, etc.). 
 * @param train_stream
 * @param log_stream
 * @param train_ratio
 */
void NeuronNetwork::train(istream& train_stream, ostream& log_stream, double train_ratio)
{

}

/**
 * Activates the input neurons of the network with the given input and reads the output of the output neurons. Use this overload if the input is already put in a vector and the output is expected in a vector. The output vector is filled with as many elements as the number of output neurons in the network.
 * @param input
 * @param output
 */
void NeuronNetwork::test(const vector<double>& input, vector<double>& output)
{

}

/**
 * Activates the input neurons of the network with the given input and reads the output of the output neurons. Use this overload if the input is already put in a vector and the output is expected to be written on a stream.
 * @param input
 * @param output_stream
 */
void NeuronNetwork::test(const vector<double>& input, ostream& output_stream)
{

}

/**
 * Activates the input neurons of the network with the given input and reads the output of the output neurons. Use this overload if the input is on a stream and the output is expected in a vector. The output vector is filled with as many elements as the number of output neurons in the network.
 * @param input_stream
 * @param output
 */
void NeuronNetwork::test(istream& input_stream, vector<double>& output)
{

}

/**
 * Activates the input neurons of the network with the given input and reads the output of the output neurons. Use this overload if the input is on a stream and the output is expected to be written on a stream as well.
 * @param input_stream
 * @param output_stream
 */
void NeuronNetwork::test(istream& input_stream, ostream& output_stream)
{

}

}