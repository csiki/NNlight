# NNlight - a lightweight neural network implementation#

An easily integrable multilayer perceptron library. Mainly for educational proposes. Has no external library dependencies.

Uses gradient-descent backpropagation learning method with various settings.

# Simple XOR example #

	using namespace std;
	using namespace NNlight;
	
	// create layers of neurons - it's also possible to work with individual neurons instead of layers
	auto input_layer = make_layer<InputNeuron, 2>(); // 2 inputs (2 bits) - XOR has 2 inputs
	auto hidden_layer1 = make_layer<Neuron, 6>(); // 6 and
	auto hidden_layer2 = make_layer<Neuron, 6>(); // 6 neurons in the hidden layer
	auto output_layer = make_layer<OutputNeuron, 1>(); // 1 output (1 bit) - XOR has 1 output
	
	Neuron::connect_layers(input_layer, hidden_layer1);
	Neuron::connect_layers(hidden_layer1, hidden_layer2);
	Neuron::connect_layers(hidden_layer2, output_layer);
	
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