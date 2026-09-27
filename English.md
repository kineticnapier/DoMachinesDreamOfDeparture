# Do Machines Dream of Departure?

This is an experimental project that trains an agent to play A Dance of Fire and Ice (ADOFAI) under human-like perceptual and physical constraints, with the goal of investigating whether it can discover effective fingering strategies different from those used by humans.

> The goal is not to build a machine that is simply faster than a human.  
> The goal is to see what a machine learns when it is given comparable constraints.

## Research Question

**Can a learning agent, given human-like perceptual and physical constraints, acquire effective ADOFAI fingering strategies that differ from those discovered by humans?**

## Principles

The agent does not directly emit key presses. Its actions are motor commands to simulated fingers; a key press occurs only after passing through muscle response, finger motion, and keyboard-switch actuation.

In the final experiments, the agent will not receive exact target timings from the level file. Like a human player, it must predict future actions from the perceptual information available to it.

## Roadmap

v0.1 does not attempt to reproduce the entire human body. It begins with a simplified two-finger model and tests whether the agent independently discovers alternating or other useful fingering patterns when a sequence is physically impossible to clear with a single finger.

See [`docs/specification.md`](docs/specification.md) for the current specification.

Experiments are recorded in [`docs/experiments.md`](docs/experiments.md).

## Final Chapter

> **Machines See Stars at the End of the Dream**

This title is reserved for the final experiment: the point at which a human-constrained agent reaches the frontier of human ADOFAI play, or finds a physically reproducible route beyond it.

## About the Title

The project title and final-chapter title are inspired by recruitment titles from *Blue Archive*. This is an independent project and is not affiliated with, endorsed by, or an official project of Yostar, NEXON Games, 7th Beat Games, or their respective rights holders.

[日本語](日本語.md)
