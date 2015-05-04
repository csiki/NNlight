# NNlight - a lightweight neural network implementation#

An easily integrable multilayer perceptron library. Mainly for educational proposes. Has no external library dependencies.

Uses gradient-descent backpropagation learning method with various settings.

# XOR data file to learn from

First the 2 inputs than the desired output in each line. Samples do not need to be separated by lines.

	0 0 0
	0 1 1
	1 0 1
	1 1 0

# Simple XOR example

	using namespace std;
	using namespace NNlight;
	
	// create layers of neurons - it's also possible to work with individual neurons instead of layers
	auto input_layer = make_layer<InputNeuron, 2>(); // 2 inputs (2 bits) - XOR has 2 inputs
	auto hidden_layer1 = make_layer<Neuron, 6>(); // 6 and
	auto hidden_layer2 = make_layer<Neuron, 6>(); // 6 neurons in the hidden layer
	auto output_layer = make_layer<OutputNeuron, 1>(); // 1 output (1 bit) - XOR has 1 output
	
	Neuron::connect_layers(input_layer, hidden_layer1); // connects each neuron of input_layer to each neuron of hidden_layer1
	Neuron::connect_layers(hidden_layer1, hidden_layer2);
	Neuron::connect_layers(hidden_layer2, output_layer); // [INPUT] -> input_layer -> hidden_layer1 -> hidden_layer2 -> output_layer -> [OUTPUT]
	
	// create network & add neuron layers to it
	NeuronNetwork network;
	network.add_layer(input_layer);
	network.add_layer(hidden_layer1);
	network.add_layer(hidden_layer2);
	network.add_layer(output_layer);
	
	// load training data file & train the network
	ifstream data_file("xor.dat"); // should contain samples like 2 inputs followed by 1 desired output for every sample (separated by white-space)
	network.settings.restart_training_if_stuck(true, 0.1, 5); // restart training 5 times if error of 0.1 or lower is not achieved
	network.train(data_file, cout, 1); // logs to standard output and use every sample for training, leaving none for testing
	
	// test network by hand
	// write 2 bits to standard input then press enter (e.g. "0 1")
	// and wait for the answer of the network (e.g. "1")
	while (!cin.eof())
	{
		network.test(cin, cout);
		cout << endl;
	}

# Simple XOR example with individual neurons

	using namespace std;
	using namespace NNlight;
	
	// initiate neurons
	auto in1 = make_neuron<InputNeuron>();
	auto in2 = make_neuron<InputNeuron>();
	auto hidden1 = make_neuron<Neuron>();
	auto hidden2 = make_neuron<Neuron>();
	auto hidden3 = make_neuron<Neuron>();
	auto out = make_neuron<OutputNeuron>();

	// make connections neuron-by-neuron
	// it's also possible to put individual neurons in std::array structures and handle them as layers
	Neuron::connect(in1, hidden1);
	Neuron::connect(in1, hidden2);
	Neuron::connect(in2, hidden1);
	Neuron::connect(in2, hidden2);
	Neuron::connect(hidden1, out);
	Neuron::connect(hidden2, out);
	
	// initiate network & add neurons
	NeuronNetwork network;
	network.add_neuron(in1);
	network.add_neuron(in2);
	network.add_neuron(hidden1);
	network.add_neuron(hidden2);
	network.add_neuron(hidden3);
	network.add_neuron(out);

	// train network
	ifstream xor_file("xor.dat");
	network.settings.restart_training_if_stuck(true, 0.1, 50);
	network.train(xor_file, cout, 1);

	// test network
	while (!cin.eof())
	{
		network.test(cin, cout);
		cout << endl;
	}

# How to use resilient backpropagation (rprop)

Just before training the network, call the `use_resilient_backpropagation()` function. Always train the network in batch mode ("learn by epoch") when applying rprop.

	using namespace std;
	using namespace NNlight;
	...
	// train network
	ifstream data_file("xor.dat");
	network.settings.restart_training_if_stuck(true, 0.1, 100); // using rprop training is faster, though more restart may be needed
	network.use_resilient_backpropagation(); // activates rprop mode for all neurons in the network; always call after all neurons are connected and included in the network
	network.train(data_file, cout, 1, true); // set batch_mode to true
	...

Reference: [M. Riedmiller, “Advanced supervised learning in multi-layer perceptrons — From backpropagation to adaptive learning algorithms,” Computer Standards & Interfaces, vol. 16, no. 3, pp. 265–278, Jul. 1994.](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.27.7876&rep=rep1&type=pdf)