/**
 * Project NNlight
 */

#ifndef _ACTIVATION_OUT_OF_BOUNDS_EXCEPTION_H
#define _ACTIVATION_OUT_OF_BOUNDS_EXCEPTION_H

#include <stdexcept>

class ActivationOutOfBoundsException : public std::runtime_error
{
public:
	ActivationOutOfBoundsException() : runtime_error("Activation has gone to - or + infinity! Not enough neurons maybe?") {}
};

#endif