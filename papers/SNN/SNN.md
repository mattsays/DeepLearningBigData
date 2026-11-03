# Spiking Neural Networks: How They Work, How They Are Trained, and What They Are Used For

This guide introduces **Spiking Neural Networks (SNNs)** and follows the five points of the presentation:

1. General Idea
2. Training / Inference Procedures
3. Current Applications
4. Current Limitations
5. Future Improvements

The sources include papers published through 2026. Experimental applications are distinguished from research directions.

## 1. General Idea

### 1.1 The Core Intuition

An SNN is a neural network in which neurons communicate through **discrete events in time**, called *spikes*. Each neuron maintains an **internal state** that changes when it receives input and continues to evolve between successive inputs.

Understanding the network's behavior therefore requires considering both **which neurons fire** and **when they fire**.

In a conventional network, a neuron is often described as follows:

$$
y = f\left(\sum_j w_j x_j + b\right)
$$

The inputs are weighted, summed, and transformed by an activation function. In SNNs, inputs instead modify a variable that retains a trace of the past: the **membrane potential**.

This introduces temporal memory even when connections between layers are purely feedforward. Conventional networks can also process sequences, for example through recurrence: memory is not exclusive to SNNs.

### 1.2 The Leaky Integrate-and-Fire Neuron

One of the most widely used models is the **Leaky Integrate-and-Fire** neuron, abbreviated **LIF**. Its name describes three operations:

- **Integrate:** the neuron accumulates the effects of incoming inputs.
- **Leaky:** in the absence of new inputs, the potential tends toward its resting value.
- **Fire:** when the potential reaches a threshold, the neuron emits a spike and the potential is reduced or returned to a reset value.

An analogy is a container with a small leak. Each input adds water; the leak lets it escape; reaching a certain level triggers firing and empties the container.

The LIF model simplifies a biological neuron: it represents spike times without reconstructing the full shape of the electrical impulse. [1]

### 1.3 A Simplified Mathematical Model

Consider a version with discrete time steps and resting and reset potentials both equal to zero.

First, calculate the potential before reset:

$$
v_i[t] = \beta u_i[t-1] + \sum_j w_{ij}s_j[t]
$$

Then, determine whether to emit a spike:

$$
s_i[t] =
\begin{cases}
1 & \text{if } v_i[t] \geq \theta_i \\
0 & \text{otherwise}
\end{cases}
$$

Finally, apply a **hard reset**:

$$
u_i[t] =
\begin{cases}
0 & \text{if } s_i[t] = 1 \\
v_i[t] & \text{otherwise}
\end{cases}
$$

| Symbol | Meaning |
| --- | --- |
| $u_i[t]$ | The neuron's potential after any reset |
| $v_i[t]$ | The potential before any reset |
| $s_j[t]$ | The presence or absence of an incoming spike |
| $w_{ij}$ | The weight of the connection from neuron $j$ to neuron $i$ |
| $\beta$ | The potential retention factor, with $0 < \beta < 1$ |
| $\theta_i$ | The firing threshold |

A high value of $\beta$ preserves the effect of inputs for longer; a lower value causes it to decay more quickly. In the exponential model, $\beta = e^{-\Delta t/\tau_m}$, where $\tau_m$ is the membrane time constant.

These equations describe a simple modeling choice: other models include synaptic currents, delays, adaptation, or a different reset mechanism. [1][2]

### 1.4 Numerical Example

Suppose we have:

- an initial potential of $u[0] = 0$;
- a threshold of $\theta = 1$;
- a retention factor of $\beta = 0.8$;
- an input of amplitude $0.4$ at every time step.

| Step | Potential Before Reset | Spike Emitted? | Potential After Reset |
| --- | --- | --- | --- |
| 1 | $0.8 \cdot 0 + 0.4 = 0.4$ | No | $0.4$ |
| 2 | $0.8 \cdot 0.4 + 0.4 = 0.72$ | No | $0.72$ |
| 3 | $0.8 \cdot 0.72 + 0.4 = 0.976$ | No | $0.976$ |
| 4 | $0.8 \cdot 0.976 + 0.4 = 1.1808$ | Yes | $0$ |
| 5 | $0.8 \cdot 0 + 0.4 = 0.4$ | No | $0.4$ |

The neuron fires at the fourth step because the inputs have accumulated quickly enough.

If sufficiently long intervals are inserted between inputs, the potential decays and the threshold may never be reached. **The same number of inputs can therefore produce different results depending on their distribution over time.**

### 1.5 From Spikes to a Network

A spike emitted by a neuron is transmitted to connected neurons. Its effect depends on the synaptic weight: an excitatory connection promotes firing, while an inhibitory connection suppresses it.

In the most common binary SNNs, a spike indicates whether an event is present or absent. This does not necessarily make the weights or membrane potential binary.

Information can be represented in different ways:

- **Rate coding:** through the number or frequency of spikes within a time window.
- **Temporal coding:** through spike times, their temporal relationships, or the time of the first spike.

In directly trained networks, hidden layers can also develop a representation without explicitly imposing a coding scheme of this kind. [3]

## 2. Training / Inference Procedures

**Inference** means using the model to produce an answer. **Training** means modifying its parameters so that it produces better answers.

### 2.1 Inference: How to Use an SNN

Consider a classifier that recognizes a spoken word.

1. **Input preparation.** The audio signal is transformed into a representation suitable for the model. It can become a sequence of spikes or feed the first layer through numerical values, such as input currents.
2. **Temporal processing.** At each step, neurons integrate inputs, update their potentials, and emit spikes if the threshold is reached.
3. **Propagation.** Spikes influence subsequent neurons through the learned connections.
4. **Output readout.** A decoding mechanism transforms the final activity into a class or a numerical value.

The readout can use spike counts, first-spike times, or a continuous output state. For example, if each output neuron represents a word, the neuron with the most spikes within the observation window can be selected.

During ordinary inference, **the weights remain fixed while the internal states change**. Systems with online learning can also update their weights during use. [2]

The number of time steps is an important choice: observing for longer can provide more evidence for a decision, but may increase latency and activity.

### 2.2 Why Training Is Difficult

Spike generation can be written as a step function:

$$
s[t] = H(v[t]-\theta)
$$

Its derivative is zero almost everywhere and undefined at the threshold. Standard backpropagation therefore does not obtain a useful signal indicating how small changes in the weights would affect spike generation.

There is also a temporal dependency: a weight can influence the potential now and contribute to a spike several steps later. The error must be attributed to both the connections and earlier time steps. [3]

### 2.3 Method A: Direct Training with Surrogate Gradients

A widely used approach keeps spikes discrete during the forward pass and uses a **substitute derivative**, or *surrogate gradient*, during the backward pass.

The procedure is:

1. Present a sequence to the network.
2. Calculate states and spikes over time.
3. Compare the output with the target using a loss function.
4. Propagate the error signal across layers and time.
5. Update the parameters using an optimizer.

Temporal backpropagation is often performed using **Backpropagation Through Time (BPTT)**, as in recurrent networks. The surrogate gradient replaces the problematic derivative of the threshold function: it does not necessarily coincide with the exact gradient of the discrete network.

**The neuron continues to emit spikes during the forward pass; the approximation concerns the backward pass.** [3]

### 2.4 Method B: ANN → SNN Conversion

A conventional network is trained first, and a spiking version is then constructed to approximate its behavior.

One example is representing a high activation with a high spike rate. Conversion makes it possible to reuse established training procedures, but may require calibration and sufficiently long time windows. [4]

### 2.5 Method C: Local Rules and STDP

**Spike-Timing-Dependent Plasticity (STDP)** modifies a connection based on the temporal relationship between presynaptic and postsynaptic spikes.

In its classical form, a presynaptic spike occurring shortly before a postsynaptic spike can strengthen the connection; the reverse order can weaken it. The rule uses local information, but on its own does not guarantee that a complex supervised objective will be learned.

Variants modulated by rewards or other signals also exist. Therefore, not all SNNs are trained with STDP, and not all learn during inference. [4]

## 3. Current Applications

SNNs are particularly interesting when data evolve over time, events are sparse, and the device must respond within a limited energy budget.

### 3.1 Event-Based Vision and Robotics

An event camera records local changes in brightness rather than only producing complete images at regular intervals. This provides a temporal stream suitable for spiking processing.

A study published in *Science Robotics* in 2024 demonstrated a pipeline using an event camera and an SNN on Intel Loihi to control a real drone, including hovering, landing, and lateral maneuvers. It is an experimental demonstration of neuromorphic perception and control. [5]

### 3.2 Audio and Keyword Spotting

Recognizing a small set of words in a continuous audio stream is an interesting use case for devices that are always listening.

A paper published online in 2025 and included in a 2026 IEEE volume proposes a *streaming keyword spotting* system with **resonate-and-fire** neurons, quantization, and early stopping of the decision process. It is an example of research on compact models for edge devices. [6]

### 3.3 Tactile Processing

Contact with a surface generates temporal sequences that can be processed by an SNN. A study on Braille reading compared spiking classifiers on Loihi with conventional models, considering accuracy, power consumption, and delay. [7]

These examples demonstrate concrete tasks. A result obtained in one specific system does not automatically imply the same advantage in every application.

## 4. Current Limitations

### 4.1 The Cost of Temporal Training

BPTT may require storing many intermediate states. Increasing the sequence length or network size can increase memory and computation requirements. Surrogate gradients make optimization possible, but remain approximations. [3]

### 4.2 The Trade-Off Between Accuracy, Latency, and Activity

A longer window can improve an estimate based on spike counts, but delays the response. Reducing spikes too aggressively can instead make it harder to transmit the necessary information.

The objective depends on the task: a useful balance must be found between response quality, time, and resources. [4]

### 4.3 Encoding Data That Are Not Event-Based

For static data, such as a photograph, converting information into a temporal sequence introduces a design choice and can add work. Direct numerical input is an alternative, but the overall benefit still needs to be assessed. [2][4]

### 4.4 Hardware Dependence

An SNN can be simulated on a CPU or GPU. Exploiting sparse events, however, requires execution that can avoid unnecessary work.

Neuromorphic hardware aims to support this behavior and reduce data movement between memory and computation. The advantage still depends on the model, platform, and workload. A network simulated using dense operations may not exploit spike sparsity. [8]

### 4.5 Energy Comparisons and Benchmarks

Claiming that an SNN consumes less energy requires a measurable comparison. It is necessary to clarify:

- whether accuracy and latency are comparable;
- whether measurements cover only the chip or the complete system;
- whether encoding, communication, and baseline power consumption are included;
- whether the conventional baseline is optimized for that device.

**A low spike count alone is not an energy measurement.** NeuroBench proposes a framework for evaluating neuromorphic algorithms and systems more systematically. [9]

## 5. Future Improvements

### 5.1 Training with Less Memory

One direction is to reduce the need to store the entire temporal history by using online updates, local signals, or gradient approximations. The challenge is to maintain good learning performance. [3]

### 5.2 On-Device Learning

Updating the model directly on the device can enable adaptation to new users or environments. However, this requires hardware and algorithms compatible with memory and energy constraints.

A March 2026 study presented a multicore architecture for direct SNN training, demonstrated on FPGAs, including distributed configurations. This is a research result toward that goal, rather than a capability already available in all neuromorphic processors. [10]

### 5.3 Fewer Spikes and Faster Responses

Making better use of spike timing can reduce the need for long counting windows. Stopping processing when the decision is sufficiently reliable can also avoid additional work. These strategies must be evaluated together with accuracy. [3][6]

### 5.4 Joint Algorithm and Hardware Design

The network, weight precision, memory, and communication can be designed together. A model that is accurate in simulation must also fit the resources of the platform on which it will run. [8][10]

### 5.5 More Reproducible Evaluations

Shared benchmarks and measurements of the complete system can clarify where SNNs offer practical advantages and where there is still room for improvement. This includes accuracy, latency, memory, and energy, with explicit comparison conditions. [9]

## Key Points to Remember for the Presentation

- An SNN processes events through neurons that maintain a state over time.
- The LIF neuron integrates inputs, gradually loses potential, and emits a spike at the threshold.
- Input timing can change the response even when the number of spikes is the same.
- Training can use surrogate gradients, conversion from ANNs, or local rules.
- Energy efficiency depends on the entire system and must be measured.

## References

1. **Gerstner et al.: Neuronal Dynamics, Section 1.3: Integrate-and-Fire Models.** The LIF model, integration, decay, and threshold. <https://neuronaldynamics.epfl.ch/online/Ch1.S3.html>
2. **Eshraghian: Tutorial 5: Training Spiking Neural Networks with snnTorch.** Recurrent state, inputs, training, and output readout. <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html>
3. **Neftci, Mostafa, and Zenke (2019): Surrogate Gradient Learning in Spiking Neural Networks.** The derivative problem, temporal coding, and learning methods. <https://arxiv.org/html/1901.09948v2>
4. **Eshraghian et al. (2023): Training Spiking Neural Networks Using Lessons From Deep Learning.** Overview of models, encoding, conversion, and training. The initial preprint is from 2021; the cited version was updated in 2023. <https://arxiv.org/abs/2109.12894>
5. **Paredes-Vallés et al. (2024): Fully neuromorphic vision and control for autonomous drone flight.** Science Robotics. <https://pubmed.ncbi.nlm.nih.gov/38748781/>
6. **Lightweight Streaming Keyword Spotting System Based on Resonate-and-Fire Neurons.** IEEE Transactions on Cognitive and Developmental Systems; published online in 2025, included in a 2026 volume. <https://ieeexplore.ieee.org/document/11120352/>
7. **Braille Letter Reading: A Benchmark for Spatio-Temporal Pattern Recognition on Neuromorphic Hardware (2022).** <https://arxiv.org/abs/2205.15864>
8. **Davies et al. (2021): Advancing Neuromorphic Computing With Loihi: A Survey of Results and Outlook.** Proceedings of the IEEE. <https://doi.org/10.1109/JPROC.2021.3067593>
9. **The NeuroBench Framework for Benchmarking Neuromorphic Computing Algorithms and Systems (2025).** Nature Communications. <https://www.nature.com/articles/s41467-025-56739-4>
10. **Li et al. (2026): A Highly Energy-Efficient Multi-Core Neuromorphic Architecture for Training Deep Spiking Neural Networks.** Nature Communications. <https://pubmed.ncbi.nlm.nih.gov/41881988/>
