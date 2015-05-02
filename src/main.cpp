
#include <iostream>
#include <fstream>
#include <array>
#include "NeuronNetwork.h"
#include "Neuron.h"

using namespace std;
using namespace NNlight;

// TODO add parameter documentation
// TODO add batch learning to NeuronNetwork::train (as bool parameter)
// TODO add momentum
// source: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.27.7876&rep=rep1&type=pdf
// TODO add rprop

int main(int argc, char* argv[])
{
	// initiate neurons
	auto in1 = make_neuron<InputNeuron>();
	auto in2 = make_neuron<InputNeuron>();
	auto hidden1 = make_neuron<Neuron>();
	auto hidden2 = make_neuron<Neuron>();
	auto out = make_neuron<OutputNeuron>();

	/*// make connections the hard way
	Neuron::connect(in1, hidden1);
	Neuron::connect(in1, hidden2);
	Neuron::connect(in2, hidden1);
	Neuron::connect(in2, hidden2);
	Neuron::connect(hidden1, out);
	Neuron::connect(hidden2, out);*/

	// make connections the easy way by connecting layers instead of individual neurons
	array<NeuronPtr, 2> input_layer = {in1, in2}; // use std::array to build layers
	array<NeuronPtr, 2> hidden_layer = {hidden1, hidden2};
	array<NeuronPtr, 1> output_layer = {out};
	Neuron::connect_layers(input_layer, hidden_layer);
	Neuron::connect_layers(hidden_layer, output_layer);
	
	// initiate network & add neurons
	NeuronNetwork network;
	network.add_neuron(in1);
	network.add_neuron(in2);
	network.add_neuron(hidden1);
	network.add_neuron(hidden2);
	network.add_neuron(out);

	// train network
	ifstream xor_file("xor.dat");
	network.settings.restart_training_if_stuck(true);
	network.train(xor_file, cout, 1);

	while (!cin.eof())
	{
		network.test(cin, cout);
		cout << endl;
	}

	return 0;
}