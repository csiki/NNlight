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
double NeuronNetwork::err_eps = 1e-4;
size_t NeuronNetwork::test_err_increase_threshold = 10;

/**
 * Creates a network of neurons, initially without the neurons. Neurons can be added after creation.
 */
NeuronNetwork::NeuronNetwork() {}

/**
 * Adds an input neuron to the network. The more input neurons are added, the more input values are needed for training and testing. The number of input neurons determines the dimension of the input data.
 * @param neuro
 */
void NeuronNetwork::add_neuron(InputNeuron& neuro)
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
void NeuronNetwork::add_neuron(OutputNeuron& neuro)
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
void NeuronNetwork::add_neuron(Neuron& neuro)
{
	auto ins = neurons.insert(std::make_shared<Neuron>(neuro));
	if (!ins.second)
		throw std::exception("Hidden neuron is already added!");
}

/**
 * Adds the shared pointer of an input neuron to the network. The more input neurons are added, the more input values are needed for training and testing. The number of input neurons determines the dimension of the input data.
 * @param neuroptr
 */
void NeuronNetwork::add_neuron(InputNeuronPtr neuroptr)
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
void NeuronNetwork::add_neuron(OutputNeuronPtr neuroptr)
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
void NeuronNetwork::add_neuron(NeuronPtr neuroptr)
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
void NeuronNetwork::train(vector<vector<double>> input, vector<vector<double>> desired_output, ostream& log_stream, double train_ratio)
{
	double prev_train_err = std::numeric_limits<double>::max(); // averaged training error
	double prev_test_err = std::numeric_limits<double>::max(); // averaged test error
	double delta_train_err = std::numeric_limits<double>::max(); // accuracy change iteration by iteration
	double delta_test_err = std::numeric_limits<double>::max();
	deque<bool> test_err_is_increasing(test_err_increase_threshold, false);
	// create and fill shuffle indices, so input and desired output can be shuffled at the same time
	vector<size_t> shuffle_indices(static_cast<size_t>(input.size() * train_ratio));
	std::iota(shuffle_indices.begin(), shuffle_indices.end(), 0);
	// separate train and test, input and desired output vectors so shuffling them possible without extra vectors
	auto train_in_beg = input.begin();
	auto train_in_end = input.begin() + static_cast<size_t>(input.size() * train_ratio);
	auto train_dout_beg = desired_output.begin();
	auto train_dout_end = desired_output.begin() + static_cast<size_t>(desired_output.size() * train_ratio);
	auto test_in_beg = train_in_end;
	auto test_in_end = input.end();
	auto test_dout_beg = train_dout_end;
	auto test_dout_end = desired_output.end();
	// create performance vectors to be able to average over errors of a training set
	vector<double> err(outputs.size());
	vector<double> train_perf(static_cast<size_t>(input.size() * train_ratio));
	vector<double> test_perf(static_cast<size_t>(input.size() - input.size() * train_ratio));
	// random stuff for shuffling
	std::random_device rand_dev;
	std::mt19937 gen(rand_dev());

	log_stream << "Training initiated ..." << std::endl;
	while (std::abs(delta_train_err) > err_eps // while the change is significant
		&& !std::all_of(test_err_is_increasing.begin(), test_err_is_increasing.end(), [] (bool inc) { return inc; } )) // not increasing previously
	{
		// shuffle train inputs & desired outputs
		std::shuffle(shuffle_indices.begin(), shuffle_indices.end(), gen);
		reorder(shuffle_indices.begin(), shuffle_indices.end(), input.begin());
		reorder(shuffle_indices.begin(), shuffle_indices.end(), desired_output.begin());

		size_t sample_index = 0;
		// check performance on test data
		if (std::distance(test_in_beg, test_in_end))
		{
			for (auto iit = test_in_beg, oit = test_dout_beg;
				iit != test_in_end && oit != test_dout_end;
				++iit, ++oit)
			{
				// forward propagation
				size_t neur_index = 0;
				for (auto& in : inputs) in->feed((*iit)[neur_index++]);
			
				// backward propagation
				std::transform(outputs.begin(), outputs.end(), oit->begin(), err.begin(),
					[] (const OutputNeuronPtr& neur, const double& d_out) {
						return std::pow(neur->get_activation() - d_out, 2.0); // MSE
				});

				// update test performance by averaging over errors
				test_perf[sample_index] = std::accumulate(err.begin(), err.end(), 0.0);
				test_perf[sample_index] /= err.size();
				++sample_index;
			}
			double avg_test_err = std::accumulate(test_perf.begin(), test_perf.end(), 0.0);
			avg_test_err /= test_perf.size();
			test_err_is_increasing.push_back( avg_test_err > prev_test_err );
			if (test_err_is_increasing.size() > test_err_increase_threshold)
				test_err_is_increasing.pop_front();
			delta_test_err = prev_test_err - avg_test_err;
			prev_test_err = avg_test_err;
		}
		
		// iterate over all train samples, forward- & backpropagation
		sample_index = 0;
		for (auto iit = train_in_beg, oit = train_dout_beg;
			iit != train_in_end && oit != train_dout_end;
			++iit, ++oit)
		{
			// forward propagation
			size_t neur_index = 0;
			for (auto in : inputs) in->feed((*iit)[neur_index++]); // TODO para
			
			// backward propagation
			std::transform(outputs.begin(), outputs.end(), oit->begin(), err.begin(),
				[] (const OutputNeuronPtr& neur, const double& d_out) {
					return std::pow(neur->get_activation() - d_out, 2.0); // MSE
			});
			neur_index = 0;
			for (auto& out : outputs) out->backpropagate(nullptr, err[neur_index++]);

			// update training performance by averaging over errors
			train_perf[sample_index] = std::accumulate(err.begin(), err.end(), 0.0);
			train_perf[sample_index] /= err.size();
			++sample_index;
		}
		double avg_train_err = std::accumulate(train_perf.begin(), train_perf.end(), 0.0);
		avg_train_err /= train_perf.size();
		delta_train_err = prev_train_err - avg_train_err;
		prev_train_err = avg_train_err;

		// log
		log_stream << "Train error: " << prev_train_err << std::endl;
		log_stream << "Train error delta: " << delta_train_err << std::endl;
		log_stream << "Test error: " << prev_test_err << std::endl;
		log_stream << "Test error delta: " << delta_test_err << std::endl;
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
	vector<vector<double>> input;
	vector<vector<double>> desired_output;
	log_stream << "Reading inputs and desired outputs from stream ..." << std::endl;
	while (!train_stream.eof())
	{
		input.push_back(vector<double>(inputs.size()));
		desired_output.push_back(vector<double>(outputs.size()));
		for (auto& in : input.back())
			train_stream >> in;
		for (auto& out : desired_output.back())
			train_stream >> out;
	}
	train(input, desired_output, log_stream, train_ratio);
}

/**
 * Activates the input neurons of the network with the given input and reads the output of the output neurons. Use this overload if the input is already put in a vector and the output is expected in a vector. The output vector is filled with as many elements as the number of output neurons in the network.
 * @param input
 * @param output
 */
void NeuronNetwork::test(const vector<double>& input, vector<double>& output)
{
	// forward propagation
	size_t neur_index = 0;
	for (auto& in : inputs) in->feed(input[neur_index++]);

	// write output
	output.resize(outputs.size());
	neur_index = 0;
	for (auto& out : outputs) output[neur_index++] = out->get_activation();
}

/**
 * Activates the input neurons of the network with the given input and reads the output of the output neurons. Use this overload if the input is already put in a vector and the output is expected to be written on a stream.
 * @param input
 * @param output_stream
 */
void NeuronNetwork::test(const vector<double>& input, ostream& output_stream, string delimiter)
{
	// forward propagation
	size_t neur_index = 0;
	for (auto& in : inputs) in->feed(input[neur_index++]);

	// write output
	for (auto& out : outputs) output_stream << out->get_activation() << delimiter;
}

/**
 * Activates the input neurons of the network with the given input and reads the output of the output neurons. Use this overload if the input is on a stream and the output is expected in a vector. The output vector is filled with as many elements as the number of output neurons in the network.
 * @param input_stream
 * @param output
 */
void NeuronNetwork::test(istream& input_stream, vector<double>& output)
{
	// forward propagation
	double in_val;
	for (auto& in : inputs)
	{
		input_stream >> in_val;
		in->feed(in_val);
	}

	// write output
	output.resize(outputs.size());
	size_t neur_index = 0;
	for (auto& out : outputs) output[neur_index++] = out->get_activation();
}

/**
 * Activates the input neurons of the network with the given input and reads the output of the output neurons. Use this overload if the input is on a stream and the output is expected to be written on a stream as well.
 * @param input_stream
 * @param output_stream
 */
void NeuronNetwork::test(istream& input_stream, ostream& output_stream, string delimiter)
{
	// forward propagation
	double in_val;
	for (auto& in : inputs)
	{
		input_stream >> in_val;
		in->feed(in_val);
	}

	// write output
	for (auto& out : outputs) output_stream << out->get_activation() << delimiter;
}

template <typename order_iterator, typename value_iterator>
void reorder(order_iterator order_begin, order_iterator order_end, value_iterator v)
{
    typedef typename std::iterator_traits<value_iterator>::value_type value_t;
    typedef typename std::iterator_traits<order_iterator>::value_type index_t;
    typedef typename std::iterator_traits<order_iterator>::difference_type diff_t;

    diff_t remaining = order_end - 1 - order_begin;
    for (index_t s = index_t(), d; remaining > 0; ++s)
	{
    	for (d = order_begin[s]; d > s; d = order_begin[d]);
    	if (d == s)
		{
    		--remaining;
    		value_t temp = v[s];
    		while (d = order_begin[d], d != s)
			{
    			swap(temp, v[d]);
    			--remaining;
    		}
    		v[s] = temp;
    	}
    }
}

}