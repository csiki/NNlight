/**
 * Project NNlight
 */

#include "Neuron.h"
#include <iostream> // TODO rm

/**
 * Neuron implementation
 * 
 * Represents a (hidden) neuron, that has both inputs and outputs as neurons. Implements a learning method that alters the input weights as the function of propagated input and backpropagated error. Use sigmoid as nonlinear function.
 */

namespace NNlight {

double Neuron::def_learning_rate = 0.01;
double Neuron::def_regularization = 0.05;
double Neuron::def_weight_lower_bound = -1.0;
double Neuron::def_weight_upper_bound = 1.0;

const double Neuron::Rprop::def_delta0 = 0.1;
const double Neuron::Rprop::def_deltamax = 50.0;
const double Neuron::Rprop::def_incr_factor = 1.2;
const double Neuron::Rprop::def_decr_factor = 0.5;

/**
 * Creates a neuron instance with the given learning rate.
 * @param learning_rate_
 */

Neuron::Neuron(double learning_rate_, double regularization_)
	: learning_rate(learning_rate_), regularization(regularization_), activation(0.0), use_rprop(false)
{
	// rand biasweight
	std::uniform_real_distribution<double> distr(def_weight_lower_bound, def_weight_upper_bound);
	std::random_device rand_dev;
	biasweight = distr(rand_dev);
}
Neuron::~Neuron() {}

/**
 * Induces input, activating the neuron.
 * @param from
 * @param act
 */
void Neuron::propagate(NeuronPtr from, double act)
{
	if (input_weights.find(from) == input_weights.end())
		throw std::exception("Neuron that propagated potential is not connected as input!");
	if (inputs.find(from) != inputs.end())
		throw std::exception("Activation is already propagated from given neuron!");

	inputs.insert(std::make_pair(from, act));
	if (inputs.size() == input_weights.size()) // all input neuron propagated their activation
	{
		// calculate activation
		auto& tmp_weights = input_weights;
		activation = std::accumulate(inputs.begin(), inputs.end(), biasweight,
			[&tmp_weights] (double acc, const std::pair<NeuronPtr, double>& in) {
				return acc + tmp_weights[in.first] * in.second;
		});
		activation = 1.0 / (1.0 + std::exp(-activation)); // sigmoid nonlinear function

		// remove all inputs
		inputs.clear();

		if (_isnan(activation))
			throw ActivationOutOfBoundsException();

		// propage activation further
		for (auto& out : outputs)
			out->propagate(shared_from_this(), activation);
	}
}

/**
 * Forces the neuron to alter its weight (that is connects this to the 'from' neuron) if necessary.
 * @param from
 * @param err
 */
void Neuron::backpropagate(NeuronPtr from, double err)
{
	if (outputs.find(from) == outputs.end())
		throw std::exception("Neuron that backpropagated potential is not connected as output!");
	if (errors.find(from) != errors.end())
		throw std::exception("Activation is already backpropagated from given neuron!");

	errors.insert(std::make_pair(from, err));
	if (errors.size() == outputs.size()) // all output neuron backpropagated their activation
	{
		// calculate delta value
		double delta = std::accumulate(errors.begin(), errors.end(), 0.0,
			[] (double acc, const std::pair<NeuronPtr, double>& err) {
				return acc + err.second; // already multiplied by output weight
		});

		// adjust bias & input weights
		if (use_rprop) biasweight += rprop(delta);
		else biasweight -= learning_rate * delta;
		for (auto& in : input_weights)
		{
			auto grad = in.first->activation * delta * activation * (1.0 - activation) - regularization * in.second;
			if (use_rprop) in.second += rprop(in.first, grad);
			else in.second -= learning_rate * grad;
		}

		// remove all errors
		errors.clear();

		// bacpropage error further
		for (auto& in : input_weights)
			in.first->backpropagate(shared_from_this(), delta * in.second); // multiplied by input weight
	}
}

/**
 * Randomize a new value for all weights (including the bias) in the range of [lower_bound, upper_bound). Also clears inputs, and errors.
 * @param lower_bound
 * @param upper_bound
 */
void Neuron::reset(double lower_bound, double upper_bound)
{
	if (upper_bound <= lower_bound)
		throw std::exception("Upper bound must be greater than lower bound!");
	
	std::uniform_real_distribution<double> distr(lower_bound, upper_bound);
	std::random_device rand_dev;
	biasweight = distr(rand_dev);
	for (auto& in : input_weights)
		in.second = distr(rand_dev);
	inputs.clear();
	errors.clear();
	activation = 0;
	rprop.reset();
}

/**
 * Activates the use of the default gradient-descent weight update method. Set by default.
 * @param learning_rate_
 * @param regularization_
 */
void Neuron::use_default_backpropation(double learning_rate_, double regularization_)
{
	learning_rate = learning_rate_;
	regularization = regularization_;
	use_rprop = false;
}

/**
 * Activates the use of the advanced gradient-descent weight update method. Stick to the default parameters.
 * Call only after the input neurons are added.
 * @param delta0
 * @param deltamax
 * @param incr_factor
 * @param decr_factor
 */
void Neuron::use_resilient_backpropagation(double delta0, double deltamax, double incr_factor, double decr_factor)
{
	rprop = Rprop(input_weights, delta0, deltamax, incr_factor, decr_factor);
	use_rprop = true;
}

/**
 * Connects the given 'neuro' neuron as an input of this neuron.
 * @param neuro
 */
void Neuron::connect_input(NeuronPtr& neuro)
{
	if (input_weights.find(neuro) != input_weights.end())
		throw std::exception("Neuron is already connected as input!");

	std::uniform_real_distribution<double> distr(def_weight_lower_bound, def_weight_upper_bound);
	std::random_device rand_dev;
	input_weights.insert(std::make_pair(neuro, distr(rand_dev)));
}

/**
 * Connects the given 'neuro' neuron as an output of this neuron.
 * @param neuro
 */
void Neuron::connect_output(NeuronPtr& neuro)
{
	if (outputs.find(neuro) != outputs.end())
		throw std::exception("Neuron is already connected as output!");

	outputs.insert(neuro);
}

/**
 * Sets the lower and upper bounds of randomized initial weight values.
 * @param lower_bound
 * @param upper_bound
 */
void Neuron::set_initial_weight_bounds(double lower_bound, double upper_bound)
{
	if (upper_bound <= lower_bound)
		throw std::exception("Upper bound must be greater than lower bound!");

	def_weight_lower_bound = lower_bound;
	def_weight_upper_bound = upper_bound;
}

Neuron::Rprop::Rprop(const unordered_map<NeuronPtr, double>& inputs, double delta0_, double deltamax_, double incr_factor_, double decr_factor_)
	: delta0(delta0_), deltamax(deltamax_), incr_factor(incr_factor_), decr_factor(decr_factor_)
{
	prev_grads.reserve(inputs.size());
	deltas.reserve(inputs.size());
	for (const auto& in : inputs)
	{
		prev_grads[in.first] = 0;
		deltas[in.first] = delta0;
	}
	bias_prev_grad = 0;
	bias_delta = delta0;
}

double Neuron::Rprop::operator()(const NeuronPtr& weight_of_input, double grad)
{
	// calculate new delta value from previous
	if (prev_grads[weight_of_input] * grad > 0)
		deltas[weight_of_input] *= incr_factor;
	else if (prev_grads[weight_of_input] * grad < 0)
		deltas[weight_of_input] *= decr_factor;
	prev_grads[weight_of_input] = grad;

	if (deltas[weight_of_input] > deltamax)
		deltas[weight_of_input] = deltamax;

	// calculate weight update
	if (grad > 0)
		return -deltas[weight_of_input];
	else if (grad < 0)
		return deltas[weight_of_input];
	return 0; // reached a platoo
}

double Neuron::Rprop::operator()(double bias_grad)
{
	// calculate new delta value from previous
	if (bias_prev_grad * bias_grad > 0)
		bias_delta *= incr_factor;
	else if (bias_prev_grad * bias_grad < 0)
		bias_delta *= decr_factor;
	bias_prev_grad = bias_grad;

	if (bias_delta > deltamax)
		bias_delta = deltamax;

	// calculate weight update
	if (bias_grad > 0)
		return -bias_delta;
	else if (bias_grad < 0)
		return bias_delta;
	return 0; // reached a platoo
}

void Neuron::Rprop::reset()
{
	bias_prev_grad = 0;
	bias_delta = delta0;
	for (auto& prev_grad : prev_grads)
		prev_grad.second = 0;
	for (auto& delta : deltas)
		delta.second = delta0;
}

}