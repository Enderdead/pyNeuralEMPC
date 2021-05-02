# pyNeuralEMPC

This library under development aims to provide a user-friendly interface for the use of the eMPC controller. This implementation will support all types of deep neural networks as models for a given problem. To do so, this library will not rely on an analytical solver like [do-mpc](https://github.com/do-mpc/do-mpc) with [casadi](https://web.casadi.org/). This library will automatically generate the nonlinear problem with its Jacobian and Hessian matrices to use a traditional nonlinear solver.

I plan to use [ipopt](https://github.com/coin-or/Ipopt) as a nonlinear solver first but other solvers will be supported in the future (like [RestartSQP](https://github.com/lanl-ansi/RestartSQP)).

The library interface will allow you to use tensorflow and pytorch models to describe the system dynamics.
Constraints on the state space will be possible as well as the definition of a custom cost function (in the same way as the model).

Only the feed forward neural network will be supported at the beginning of the development but the recurrent neural network will be added later.

## Work in progress

This library is not yet in stable release. Here is the planned roadmap:

TODO list :

- [ ] Architecture and design of class paterns.
- [ ] First working version with some features including three integration methods and a solver.
- [ ] Rédaction de la première version de documentation et de test.
- [ ] Adding a complete support for tensorflow and pytorch models.
- [ ] Add the simulating approch design.
- [ ] Add other solvers.

## Documentation

Since this library isn't finished yet, no documentation is provided !

## Example

Since this library isn't fi... blablabla, no example... =)
