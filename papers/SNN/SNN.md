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

### 2.2 What Training Actually Changes: A Running Example

During inference, the membrane potential changes as the network processes a sequence. During training, an additional process changes the **learned parameters**, usually the synaptic weights, so that future sequences produce more useful responses. Some models also learn thresholds or time constants, but we will keep those fixed here.

Consider one neuron receiving **two consecutive input spikes**. We want it to emit an output spike at the second time step. This is a deliberately small supervised example, not a complete classification task.

We choose:

- initial state $u[0]=0$;
- fixed threshold $\theta=1$;
- fixed retention factor $\beta=0.8$;
- one learned scalar weight $w$, reused at both time steps;
- observed inputs $x[1]=x[2]=1$, indicating that an input spike arrives at each step.

The potential before reset follows:

$$
v[t]=0.8u[t-1]+wx[t]
$$

Here $x[t]$ is the incoming spike, $v[t]$ is the potential before reset, and $u[t]$ is the state after reset. As in Section 1.3, the neuron emits $s[t]=1$ when $v[t]\geq1$ and then resets its state to zero.

Start with $w=0.5$:

| Time Step | Potential Before Reset | Output Spike | State After Reset |
| --- | --- | --- | --- |
| 1 | $v[1]=0.8\cdot0+0.5=0.5$ | $s[1]=0$ | $u[1]=0.5$ |
| 2 | $v[2]=0.8\cdot0.5+0.5=0.9$ | $s[2]=0$ | $u[2]=0.9$ |

The first input leaves a trace that persists into the second time step. However, the combined contribution does not reach the threshold. **Training should change the weight so that this input sequence produces the requested spike.**

To quantify the mistake, let the desired output at the second step be $s^{\ast}[2]=1$. The actual prediction is $s[2]=0$. We choose a simple squared-error loss:

$$
L=\frac{1}{2}\left(s[2]-s^{\ast}[2]\right)^2
$$

The difference measures the prediction error; squaring makes either direction of error positive. The factor $1/2$ simplifies differentiation. For the current prediction:

$$
L=\frac{1}{2}(0-1)^2=0.5
$$

If the neuron emitted the desired spike, the loss would be zero. This example constrains only the output at the second step. A task requiring a complete spike pattern would also need to penalize incorrect spikes at other times.

For larger tasks, the loss can instead compare spike counts, spike times, continuous readout values, or class probabilities. The objective determines which behavior the network learns. [2][3]

### 2.3 Why the Spike Threshold Makes Training Difficult

Gradient descent needs a signal indicating how a small parameter change affects the loss. The difficulty is the hard threshold:

$$
s[t]=H(v[t]-\theta)
$$

Here $H$ is the step function: the output is zero below threshold and one at or above it.

In our example, while there is no reset at the first step:

$$
v[2]=0.8w+w=1.8w
$$

Compare nearby weight values:

| Weight | Potential at Step 2 | Output at Step 2 |
| --- | --- | --- |
| $0.50$ | $0.900$ | No spike |
| $0.51$ | $0.918$ | No spike |
| $0.56$ | $1.008$ | Spike |

The output stays unchanged over an interval and then jumps. The derivative of $H$ is zero away from the threshold and undefined at the threshold. Therefore, ordinary differentiation through this operation gives no useful local signal for moving toward the desired spike.

This is also a **temporal credit assignment** problem: the first input affects the second potential through the neuron's retained state. Training must account for both current and earlier contributions. [3]

### 2.4 Method A: Direct Training with Surrogate Gradients

The central idea of a **surrogate gradient** is to keep the hard threshold during the forward pass but substitute its derivative during the backward pass. The forward computation still produces discrete spikes; the backward computation uses an update signal chosen to support optimization. It is generally not the exact gradient of the discrete model. [3]

For our example, choose a simple substitute derivative that is one near the threshold and zero farther away:

$$
g(v)=
\begin{cases}
1 & \text{if } |v-\theta|<0.2 \\
0 & \text{otherwise}
\end{cases}
$$

The width and height here are illustrative design choices. Other surrogates use triangular shapes or derivatives of smooth functions.

At the second step, $v[2]=0.9$ and $\theta=1$, so $g(v[2])=1$. Although no spike was emitted, the backward pass can now assign a nonzero sensitivity to a potential change.

To compute the update, separate the chain rule into three factors.

**First, how does the loss depend on the predicted spike value?** Treating the output as a numerical variable in the loss:

$$
\frac{\partial L}{\partial s[2]}=s[2]-1=-1
$$

The negative sign indicates that increasing the output toward the target would reduce the loss.

**Second, how does the spike depend on the potential?** Replace the problematic derivative in the backward pass:

$$
\frac{\partial s[2]}{\partial v[2]}\approx g(v[2])=1
$$

**Third, how does the potential depend on the weight?** Since $v[2]=1.8w$ in this example:

$$
\frac{\partial v[2]}{\partial w}=1.8
$$

Multiplying the factors gives the surrogate gradient estimate:

$$
\frac{\partial L}{\partial w}\approx(-1)\cdot1\cdot1.8=-1.8
$$

At the first step the potential is far from threshold, so the chosen surrogate derivative is zero there. Together with the absence of an earlier reset, this makes the temporal calculation particularly simple. In general, the reset mechanism must also be handled consistently in the backward computation.

Gradient descent subtracts the estimated gradient, scaled by a **learning rate** $\eta$. With $\eta=0.05$:

$$
w_{\mathrm{new}}=0.5-0.05(-1.8)=0.59
$$

The negative estimated gradient increases the weight. Run the sequence again from $u[0]=0$:

$$
v[1]=0.59
$$

$$
v[2]=0.8\cdot0.59+0.59=1.062
$$

The first step remains below threshold, and the second now produces the desired spike. The loss for this example becomes zero.

> The surrogate derivative provides an update direction even when the hard spike output has not yet changed. The forward pass remains a spiking computation.

This successful update is a teaching example, not a guarantee that every surrogate step reduces the loss. Larger networks require repeated optimization over many examples, with appropriate surrogate scales and learning rates. [3]

### 2.5 Why Backpropagation Through Time Is Also Needed

The factor $1.8$ in our example contains two contributions:

- $1$ from the effect of $w$ on the second input;
- $0.8$ from the effect of the same weight on the first input, retained until the second step.

Ignoring the first contribution in time would miss part of the weight's influence. **Backpropagation Through Time (BPTT)** extends this reasoning across all time steps and layers. Conceptually, it unfolds the state update into a sequence of connected computations and propagates error backward through them. The same physical weight is shared across time, so its update combines contributions from its uses throughout the sequence. [2]

**BPTT and surrogate gradients solve different parts of the problem.** BPTT determines how error flows through the temporal dependencies. Surrogate derivatives supply a usable sensitivity at spike thresholds along those paths.

The same neuron state creates temporal dependencies even in a network whose layer-to-layer connections are feedforward. Explicit recurrent connections can add further dependencies.

Full BPTT typically requires storing intermediate states, so long sequences can be expensive in memory. Truncated BPTT cuts the backward history, potentially losing information about distant causes. Online methods such as **e-prop** instead combine eligibility traces with learning signals to approximate the required updates without a full backward traversal of the sequence. These methods introduce their own approximations and trade-offs. [2][3][13]

### 2.6 Real-Time Recurrent Learning (RTRL)

#### The Idea: Carry Sensitivities Forward with the State

**Real-Time Recurrent Learning (RTRL)** is another way to calculate gradients through temporal dependencies. Alongside the neuron states, it maintains information about **how sensitive the current state is to each learned parameter**. When an error signal becomes available, these sensitivities connect it to the parameters without a backward traversal of the earlier states. [14]

The question it answers is: if a weight had been slightly different throughout the sequence, how would the current state have changed? This sensitivity is different from the state itself. The membrane potential describes what the neuron is doing; its sensitivity describes how a parameter change would affect that behavior.

In our running example, the second potential depends on the weight with factor $1.8$. BPTT recovers this dependence by tracing backward from the error. RTRL can build the same sensitivity as the two inputs arrive.

#### The Same Two-Step Example, Calculated Forward

Keep the parameters fixed during this comparison: $w=0.5$, $\beta=0.8$, and $\theta=1$. Both input spikes are one, and the initial state is independent of the weight.

At step 1:

$$
v[1]=w=0.5
$$

Define the pre-reset sensitivity as $q[t]=\partial v[t]/\partial w$. Since the initial state is fixed:

$$
q[1]=1
$$

This means that increasing $w$ by a sufficiently small amount increases the first potential by that same amount. There is no spike or reset at step 1. Moreover, the chosen surrogate derivative is zero at $v[1]=0.5$, so the post-reset state sensitivity is also one in this example.

At step 2, carry that sensitivity forward through the retained state and add the effect of the new input:

$$
q[2]=0.8\cdot1+1=1.8
$$

The $0.8$ term is the retained influence of the first input. The additional $1$ comes from the current use of the same weight.

| Step | Potential Before Reset | Sensitivity Before Reset |
| --- | --- | --- |
| 1 | $v[1]=0.5$ | $q[1]=1$ |
| 2 | $v[2]=0.9$ | $q[2]=1.8$ |

When the second-step loss becomes available, its derivative with respect to the output spike is $-1$. The surrogate derivative at $v[2]=0.9$ is one. Multiply these by the sensitivity already maintained online:

$$
\frac{\partial L}{\partial w}\approx(-1)\cdot1\cdot1.8=-1.8
$$

The result matches our BPTT calculation. **The temporal influence was accumulated while moving forward, so no backward pass over the two-step history was needed.**

The simple sensitivity update above applies to this particular example. For longer spiking sequences, sensitivities must also account for spikes, resets, and any other state variables.

#### From One Neuron to a General Recurrent Network

For a larger network, write its state update as:

$$
\mathbf{h}[t]=F(\mathbf{h}[t-1],\mathbf{x}[t];\mathbf{w})
$$

Here $\mathbf{h}[t]$ is a vector of $N$ state variables, $\mathbf{x}[t]$ is the observed input, and $\mathbf{w}$ is a vector of $M$ learned parameters. The function $F$ describes the state transition. In an SNN, the state may include membrane potentials, synaptic currents, and adaptation variables.

The sensitivity becomes an $N\times M$ matrix:

$$
P_{ik}[t]=\frac{\partial h_i[t]}{\partial w_k}
$$

The index $i$ selects a state component, and $k$ selects a learned parameter. Each entry tracks a parameter's influence on one component of the current state.

Applying the chain rule gives the RTRL recurrence:

$$
\boxed{\mathbf{P}[t]=\mathbf{J}[t]\mathbf{P}[t-1]+\mathbf{B}[t]}
$$

The two local derivative matrices are:

$$
\mathbf{J}[t]=\frac{\partial F}{\partial\mathbf{h}[t-1]}
$$

$$
\mathbf{B}[t]=\left.\frac{\partial F}{\partial\mathbf{w}}\right|_{\mathbf{h}[t-1]\ \mathrm{fixed}}
$$

The $N\times N$ matrix $\mathbf{J}[t]$ describes how the previous state affects the next state. Multiplying it by $\mathbf{P}[t-1]$ transports the sensitivities already accumulated. The $N\times M$ matrix $\mathbf{B}[t]$ adds the parameters' direct effect at the current step, holding the previous state fixed. If the initial state does not depend on the parameters, $\mathbf{P}[0]=\mathbf{0}$. [15]

For a current loss $\ell[t]$ that depends on these parameters only through the current state, the gradient is:

$$
\frac{\mathrm{d}\ell[t]}{\mathrm{d}\mathbf{w}}=
\frac{\partial\ell[t]}{\partial\mathbf{h}[t]}\mathbf{P}[t]
$$

The loss derivative is a row vector, so the result has one entry per parameter. If the loss also depends directly on a parameter, its direct derivative must be added. Contributions from multiple time-step losses can be accumulated for an update.

#### BPTT and RTRL: What Changes and What Stays the Same

| Aspect | Full BPTT | RTRL |
| --- | --- | --- |
| Temporal calculation | Propagates error backward through stored or recomputed states | Propagates state-to-parameter sensitivities forward |
| Main information retained | State history or checkpoints | Current state and sensitivity matrix |
| Streaming use | Requires a backward computation over a chosen interval | Can form an update as each loss becomes available |
| Long temporal context | Full history is costly; truncation discards paths | No explicit temporal truncation is required |
| Main scaling difficulty | Length of the backward history | Number of state-parameter dependencies |

For differentiable dynamics, fixed parameter values, the same trajectory, and the same objective, full BPTT and exact RTRL calculate the same gradient. They organize the chain rule differently. Updating weights after every step changes the training trajectory, so arbitrary online updates should not be equated with a single gradient update after a full sequence. [14][15]

An online method also cannot use an error signal before it exists. If a target arrives only at the end, RTRL can maintain sensitivities while waiting, then combine them with that target's loss. It does not obtain future information early.

#### What Changes for Spiking Neurons

RTRL does not remove the hard-threshold differentiation problem. With discrete spikes, derivative calculations usually still need a surrogate or another suitable treatment. Thus, **RTRL and BPTT are alternatives for tracking temporal influence; surrogate derivatives address the spike nonlinearity**. [3]

The reset belongs to the state transition too. For our zero-valued hard reset, let $p[t]$ denote the post-reset state sensitivity. The pre-reset sensitivity follows:

$$
q[t]=\beta p[t-1]+x[t]
$$

If the reset is differentiated through using the same surrogate $g$, then:

$$
p[t]\approx\left(1-s[t]-v[t]g(v[t])\right)q[t]
$$

This follows by differentiating $u[t]=(1-s[t])v[t]$: both the potential and the spike-dependent reset factor depend on the weight. In the first step of our example, $s[1]=0$ and $g(v[1])=0$, giving $p[1]=q[1]=1$, as used above. Implementations that detach the reset use a different differentiation convention; a comparison with BPTT must use the same convention.

With surrogate derivatives, the propagated sensitivities and resulting update are surrogate quantities, rather than guaranteed exact derivatives of the hard-spiking system.

#### Why Exact RTRL Can Be Expensive

In a conventional densely recurrent network with $N$ state variables and approximately $N^2$ parameters, the sensitivity matrix contains approximately $N^3$ entries. A standard dense implementation requires approximately $O(N^4)$ computation per time step. These costs do not grow with the elapsed sequence length, but they grow steeply with network size. [15]

The word *real-time* describes the forward, causal availability of the calculation; it does not guarantee fast execution. Particular recurrent structures can reduce the cost, while approximate methods compress or simplify the sensitivities. Work on element-wise recurrent architectures illustrates opportunities and limitations of making exact RTRL practical. [16]

RTRL is not automatically biologically local: a general parameter can influence many state variables. Methods such as e-prop pursue more practical online updates with eligibility traces and learning signals, with their own approximations. Online learning therefore does not mean that every synapse can independently calculate the full exact gradient. [13][15]

> BPTT traces an error backward to its earlier causes. RTRL continually carries forward how parameters influence the current state, ready to combine that information with an error when it becomes available.

### 2.7 The Complete Direct-Training Cycle

For a supervised task, the operational procedure is:

1. **Select an input sequence and its target.** The target might be a class, a trajectory, or a desired response.
2. **Initialize the neuron states.** Independent examples usually start with reset states; continuous streams need an explicit policy for carrying state between segments.
3. **Run the forward pass over time.** Compute membrane states, spikes, resets, and readout values.
4. **Calculate the loss.** Compare the relevant outputs with the target.
5. **Run the backward pass.** Apply BPTT and surrogate derivatives through the spiking computation.
6. **Update the parameters.** An optimizer changes the weights and any other parameters selected for learning.
7. **Repeat across the training data.** Evaluate on separate data to check whether the learned behavior generalizes.

After training, ordinary inference uses the learned parameters without the loss calculation or backward pass. The membrane states still evolve as inputs arrive. Training on a GPU and deploying the resulting model on a neuromorphic processor are separate stages; low-power inference does not imply equally low-cost training. [2][4][8]

### 2.8 Method B: ANN → SNN Conversion

Conversion starts by training a conventional ANN with its usual differentiable operations. A spiking model is then constructed to approximate that trained network's behavior.

For an illustrative rate-based mapping, suppose an ANN activation of $0.6$ corresponds to a firing rate of $60$ spikes per second. Over $100$ ms, the corresponding expected count would be six spikes. A shorter observation window gives less opportunity to represent that rate accurately. This numerical scale is only an example; the appropriate mapping must be calibrated for the model.

Conversion therefore involves more than copying weights. Activation scales, thresholds, reset behavior, and the observation window affect how closely the SNN approximates the original network. Depending on the architecture, additional adaptation or fine-tuning may be needed.

This route reuses established ANN training procedures, while direct SNN training optimizes the spiking dynamics themselves. Conversion can offer good accuracy, but longer windows needed for approximation may increase inference latency and activity. [4][11]

### 2.9 Method C: Local Rules and STDP

**Spike-Timing-Dependent Plasticity (STDP)** updates a synapse using the relative timing of presynaptic and postsynaptic spikes.

Suppose neuron A fires at $10$ ms and connected neuron B fires at $15$ ms. Define the timing difference as postsynaptic time minus presynaptic time:

$$
\Delta t=t_B-t_A=5\ \mathrm{ms}
$$

In the classical asymmetric STDP rule, this order can strengthen the A-to-B connection. If B fires before A, the rule can weaken it. The size of the update depends on the timing difference and the chosen plasticity rule. The classic experimental result concerns particular biological conditions; STDP is not a universal rule with the same shape at every synapse. [12]

The important computational distinction is that a basic pair-based update uses **local spike information**, without requiring a global classification error to be propagated backward through every layer.

However, temporal association alone does not specify the task. A connection can strengthen because A regularly precedes B even if this does not improve a desired classification. Reward-modulated or other three-factor rules add a further learning signal to guide plasticity toward an objective. [3][4]

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
11. **Rueckauer et al. (2017): Conversion of Continuous-Valued Deep Networks to Efficient Event-Driven Networks for Image Classification.** ANN-to-SNN conversion, calibration, and inference trade-offs. <https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2017.00682/full>
12. **Bi and Poo (1998): Synaptic Modifications in Cultured Hippocampal Neurons: Dependence on Spike Timing, Synaptic Strength, and Postsynaptic Cell Type.** Experimental basis for classical STDP. <https://pmc.ncbi.nlm.nih.gov/articles/PMC6793365/>
13. **Bellec et al. (2020): A Solution to the Learning Dilemma for Recurrent Networks of Spiking Neurons.** E-prop, eligibility traces, and online learning. <https://www.nature.com/articles/s41467-020-17236-y>
14. **Williams and Zipser (1989): A Learning Algorithm for Continually Running Fully Recurrent Neural Networks.** Neural Computation, 1(2), 270-280. Original RTRL formulation. <https://doi.org/10.1162/neco.1989.1.2.270>
15. **Marschall, Cho, and Savin (2020): A Unified Framework of Online Learning Algorithms for Training Recurrent Neural Networks.** Journal of Machine Learning Research, 21(135), 1-34. RTRL sensitivities, computational complexity, and online approximations. <https://jmlr.org/papers/v21/19-562.html>
16. **Irie, Gopalakrishnan, and Schmidhuber (2024): Exploring the Promise and Limits of Real-Time Recurrent Learning.** ICLR 2024; initial preprint 2023. Structured recurrence and practical RTRL trade-offs. <https://arxiv.org/abs/2305.19044>
