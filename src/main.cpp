
#include <iostream>
#include <fstream>
#include <array>
#include "NeuronNetwork.h"
#include "Neuron.h"

using namespace std;
using namespace NNlight;

// TODO add parameter documentation
// TODO add momentum
// source: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.27.7876&rep=rep1&type=pdf
// TODO add rprop
// TODO add adjustable activation function and derivative
// TODO threadpool multithreading

int main(int argc, char* argv[])
{
	// initiate network
	auto input_layer = make_layer<InputNeuron, 9>();
	auto hidden_layer1 = make_layer<Neuron, 30>();
	auto hidden_layer2 = make_layer<Neuron, 30>();
	auto output_layer = make_layer<OutputNeuron, 1>();

	Neuron::connect_layers(input_layer, hidden_layer1);
	Neuron::connect_layers(hidden_layer1, hidden_layer2);
	Neuron::connect_layers(hidden_layer2, output_layer);

	NeuronNetwork network;
	network.add_layer(input_layer);
	network.add_layer(hidden_layer1);
	network.add_layer(hidden_layer2);
	network.add_layer(output_layer);

	// train network
	ifstream data_file("tic-tac-toe_proc.dat");
	network.settings.restart_training_if_stuck(true, 0.2, 10);
	network.train(data_file, cout, 0.9);

	// test network
	while (!cin.eof())
	{
		network.test(cin, cout);
		cout << endl;
	}

	return 0;
}