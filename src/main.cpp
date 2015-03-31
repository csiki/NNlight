
#include <iostream>
#include <fstream>
#include "NeuronNetwork.h"
#include "Neuron.h"

using namespace std;
using namespace NNlight;

// TODO add parameter documentation

int main(int argc, char* argv[])
{
	// initiate neurons
	auto in1 = make_neuron<InputNeuron>();
	auto in2 = make_neuron<InputNeuron>();
	auto hidden1 = make_neuron<Neuron>();
	auto hidden2 = make_neuron<Neuron>();
	auto out = make_neuron<OutputNeuron>();

	// make connections
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
	network.add_neuron(out);

	// train network
	ifstream xor_file("xor.dat");
	network.train(xor_file, cout, 1);

	return 0;
}