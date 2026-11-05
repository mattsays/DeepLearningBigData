# Presentation Script — *Beyond Conventional Deep Learning*

**Spiking Neural Networks and Graph Neural Networks: Temporal Dynamics and Relational Structure**
Big Data & Deep Learning · Single narrator · English · **20:00 of speaking** (Q&A not included)

---

## How to read this script

- **Pace:** about 125 words per minute (2,496 spoken words), plus one 10 s pause on the last slide. Under each slide title you will find its target clock (start – end), its duration and its number of clicks.
- **`[click N]`** = press → once (the N-th press on that slide). Start the sentence before the click, finish it after.
- **`[on entry]`** = on the SNN slides from "One neuron" onwards, the first build is already on screen when the slide opens, so there is nothing to press: start speaking as soon as the slide appears. The clicks that follow are numbered from 1.
- Slides marked **no clicks** have nothing to reveal (the two title slides and the SNN conclusion).
- **Math is spoken, never read.** Say what a formula *does*, not its symbols.
- Plain text is spoken aloud. Lines starting with `>` are notes to the speaker.

## Timing at a glance

| Part | Slides | Time | Clock |
|---|---|---|---|
| Introduction | 1–2 | 1:45 | 0:00 → 1:45 |
| Spiking Neural Networks | 3–17 | 8:26 | 1:45 → 10:11 |
| Graph Neural Networks | 18–31 | 8:40 | 10:11 → 18:51 |
| Perspectives and references | 32–34 | 1:09 | 18:51 → 20:00 |

Checkpoints for rehearsal: **1:45 · 10:11 · 18:51 · 20:00** (end of Introduction, SNN part, GNN part, and talk)

---

# PART 1 — Introduction

### 01 · Title — Beyond Conventional Deep Learning
`0:00 – 0:21 · 21 s · 3 clicks`

[click 1] Hello everyone. [click 2] Today we go beyond conventional deep learning, with two families of networks: Spiking Neural Networks, which add temporal dynamics, and Graph Neural Networks, which add relational structure. [click 3] We are the Big Data and Deep Learning group, and we will guide you through both.

### 02 · Limitations of conventional neural architectures
`0:21 – 1:45 · 84 s · 4 clicks`

[click 1] MLPs, CNNs and Transformers are built around vectors, grids and sequences. A forward pass sends the input through layers of weighted connections, and backpropagation computes gradients to update those weights. This recipe is hugely successful, but it hides two assumptions that real data often breaks.

[click 2] The first is time. Frame-based models compute at every step even when nothing has happened, they lose the exact timing of events, and they waste effort on sparse, asynchronous signals. A biological neuron works differently: it keeps an internal state, its membrane potential, that evolves over time. That is the equation on screen, and we will return to it shortly.

[click 3] The second is relational structure. Social networks, molecules, roads, knowledge graphs: entities plus relations, which standard models do not exploit explicitly. The key property is permutation equivariance: reorder the nodes, and the outputs reorder in the same way. The graph has not changed, only our labeling.

[click 4] Each limitation has an answer. For time, Spiking Neural Networks. For relations, Graph Neural Networks. They address different challenges: they are not competing alternatives.

---

# PART 2 — Spiking Neural Networks

### 03 · Spiking Neural Networks — From brain to machine
`1:45 – 1:54 · 9 s · no clicks`

Let us start with Spiking Neural Networks, from brain to machine. The natural starting point is the biological neuron.

### 04 · The biological neuron
`1:54 – 2:34 · 40 s · 4 clicks`

[click 1] A neuron receives signals through its dendrites and integrates them in the soma. If enough builds up at the axon hillock, a pulse travels down the myelinated axon to the synapses of the next cells. [click 2] Electrically, the membrane is an RC circuit: a capacitor, the lipid bilayer, and a resistor, the ion channels. [click 3] The potential leaks towards rest while the input current pushes it up. At a threshold, the neuron spikes and resets. [click 4] The plot shows it: leaky integration, a spike, a reset.

### 05 · One neuron, three operations
`2:34 – 3:11 · 37 s · 3 clicks`

Machines use a discrete-time version of this neuron. [on entry] It performs three operations. [click 1] First, integrate and retain: weighted inputs enter the cell, and a decay factor, beta, keeps part of the previous state. [click 2] Second, compare with the threshold: when the potential reaches theta, the neuron emits a spike, a one. Otherwise it outputs zero. [click 3] Third, reset by subtraction: one step after a spike we subtract theta. Integrate, fire, reset. Everything that follows is built on these three steps.

### 06 · Inference over time
`3:11 – 3:40 · 29 s · 4 clicks`

[on entry] At inference, input events arrive as spikes over time. [click 1] Each neuron integrates, retains, fires and resets. [click 2] A readout decodes the answer: for classification, the largest spike count wins, here six against two. [click 3] States evolve at every step, but the learned weights stay fixed. [click 4] For regression, the readout is continuous, such as an angle of thirty point seven degrees.

### 07 · Forward inference and backpropagation
`3:40 – 4:12 · 32 s · 1 click`

[on entry] Here is the catch. In the forward pass the neuron applies a hard threshold: zero point nine is below the threshold of one, so no spike. [click 1] But training needs the derivative of the spike with respect to the potential. At the threshold the function jumps, so the derivative is undefined, and everywhere else it is zero. Backpropagation gets no signal, and the network cannot learn.

### 08 · How do we solve this issue? — Surrogate gradient
`4:12 – 4:40 · 28 s · 1 click`

[on entry] So how do we solve this issue? [click 1] With a surrogate gradient. The forward pass keeps the hard threshold and binary spikes. The backward pass swaps the missing derivative for a smooth stand-in: here a rectangle, equal to one within zero point two of the threshold and zero elsewhere. So zero point nine now passes a learning signal.

### 09 · Backpropagation through time (BPTT)
`4:40 – 5:31 · 51 s · 7 clicks`

How do we send that gradient back through time? First: backpropagation through time. [on entry] Follow one neuron as time advances: its state evolves, but the neuron and its weights stay the same. [click 1] At step one an input updates the state, and we save it. [click 2] At step two the same neuron updates again, and we save that. [click 3] At step three we compute the output and the loss. [click 4] Now we open the saved history. [click 5] Starting from the final loss, we use the surrogate derivative wherever a spike was crossed, [click 6] and move backward through the stored states. [click 7] Finally we sum the three contributions and update the shared weights once.

### 10 · Real-time recurrent learning (RTRL)
`5:31 – 6:11 · 40 s · 7 clicks`

BPTT stores the whole history. Real-time recurrent learning does not. [on entry] Again, one neuron with shared weights. [click 1] Next to the state we track its sensitivity to the weight. [click 2] At step one, both update together. [click 3] At step two the sensitivity is carried forward, with past influence already inside. [click 4] Same at step three. [click 5] When the loss arrives, the temporal information is already summarized. [click 6] We combine the current error with that sensitivity, [click 7] and update the weight immediately. The price: a sensitivity for every state and weight pair.

### 11 · ANN-to-SNN conversion
`6:11 – 6:49 · 38 s · 7 clicks`

A third route skips spike-based training. [on entry] Train an ordinary ANN with standard backpropagation. [click 1] Reuse its weights, [click 2] and swap compatible activations for spiking neurons. [click 3] Calibrate scales and thresholds so firing rates approximate the ANN's activity. [click 4] For one neuron, a normalized activation becomes a firing rate. [click 5] We count spikes over an observation window. [click 6] After three steps, one spike means a rate of about zero point three three: coarse. [click 7] A longer window refines it, at the cost of latency.

### 12 · Local plasticity: STDP
`6:49 – 7:17 · 28 s · 4 clicks`

A fourth, brain-inspired option is local plasticity. [on entry] Neuron A sends to neuron B through a weight w. Local means the synapse only uses the two neurons' spike times. [click 1] If A fires first, [click 2] the weight increases. [click 3] If B fires first, it decreases. [click 4] A larger weight makes A's future spikes count more for B. Causal pairs are reinforced: spike-timing-dependent plasticity.

### 13 · Neuromorphic vision and drone control
`7:17 – 7:51 · 34 s · 3 clicks`

Now, applications. [on entry] An event camera has no frames: each pixel reports local brightness changes. [click 1] Those events feed an SNN running on Intel Loihi, which turns them into motion estimates and flight-control commands. [click 2] It runs on board at two hundred hertz, hovering and landing a real drone. [click 3] It is a research prototype, published in Science Robotics in 2024, but it shows perception and control working together in a neuromorphic system.

### 14 · Energy savings depend on the system
`7:51 – 8:23 · 32 s · 3 clicks`

A reality check: are energy savings automatic? [on entry] Sparse spikes reduce activity, but baseline and system costs remain. [click 1] On this chip, idle power is already zero point nine four watts, and activity adds only seven to twelve milliwatts. [click 2] The total, about zero point nine five watts, excludes sensors, communication and actuation. [click 3] So compare energy per task at matched accuracy and latency, and state the measurement boundary.

### 15 · Streaming keyword spotting
`8:23 – 8:49 · 26 s · 3 clicks`

Keyword spotting turns a continuous sound stream into a small-vocabulary decision. [on entry] A spoken yes arrives over time. [click 1] Neurons respond to sound frequencies and express them as spikes. [click 2] SNN states accumulate evidence: eight for yes, two for no, in this illustrative case. [click 3] And the system can decide as soon as the evidence is sufficient.

### 16 · Tactile sensing: Braille recognition
`8:49 – 9:25 · 36 s · 2 clicks`

Our last application is touch. [on entry] A fingertip slides across Braille, and pressure changes form a temporal pattern. We turn pressure into events and classify them with a recurrent SNN on Loihi, over twenty-seven classes. [click 1] It reaches seventy-eight point three percent; an LSTM on a Jetson, with frame input, reaches ninety-six point nine. [click 2] But the comparison mixes encoding, model and hardware. The lesson is not that SNNs are worse: accuracy depends on the whole system.

### 17 · Conclusion: spikes trade density for efficiency
`9:25 – 10:11 · 46 s · no clicks`

To close the SNN part: spikes trade density for efficiency. SNNs compute with timed binary events, so silence costs nothing. Training is now solved well enough to scale, with surrogate-gradient BPTT, conversion and hybrid pipelines. The wins land first at the edge, on neuromorphic chips, where the slide reports energy advantages of thirty to a thousand times. Next come spiking foundation models, on-chip continual learning and hybrid ANN-SNN systems. Bottom line: a working, measurable route to AI that computes only when there is something worth computing. That was the temporal side. Now, the relational side.

---

# PART 3 — Graph Neural Networks

### 18 · Graph Neural Networks — Learning through connections
`10:11 – 10:18 · 7 s · no clicks`

Now the second pillar: Graph Neural Networks, learning through connections. From time, we move to structure.

### 19 · Graph Neural Networks — what they do
`10:18 – 10:39 · 21 s · 1 click`

[click 1] Graph Neural Networks learn from graph-structured data by exchanging information between connected nodes. Each node combines its own features with messages from its neighbors. Repeat that, and the network uses both attributes and relationships to predict things about nodes, connections, or whole graphs.

### 20 · GNN: The unfair advantage
`10:39 – 11:05 · 26 s · 2 clicks`

Why graphs? They can model almost any kind of data. [click 1] Even traditional data fits: an image is a grid graph of pixels, and text is a chain with a node per word. [click 2] Heterogeneous data is more natural still: atoms and bonds form molecules, people and relationships form social networks. GNNs learn from these connections.

### 21 · GNN: Improvements over classical NNs
`11:05 – 11:43 · 38 s · 2 clicks`

Compared with classical networks, GNNs lift two assumptions. [click 1] First, size. An MLP or a CNN expects a fixed-size input, while graphs have different numbers of nodes, edges and neighbors. A GNN applies the same shared weights at every node, so any size works. [click 2] Second, order. In a classical network, input position carries meaning: shuffle the entries and the output changes. A graph has no canonical node order, so a GNN aggregates independently of order, and the graph-level prediction stays the same.

### 22 · How GNNs see a graph
`11:43 – 12:25 · 42 s · 4 clicks`

Formally, a graph is a set of nodes and a set of edges. [click 1] Node features go into a matrix X: one row per node, one column per feature. [click 2] The adjacency matrix A has a one wherever two nodes are connected; here node three is linked to nodes one, two and four. [click 3] The degree matrix D simply counts each node's neighbors: two, two, three, one. [click 4] For graph convolutional networks we add self-loops, so each node keeps its own features, and we normalize by degree so that no neighbor dominates.

### 23 · Neural message passing
`12:25 – 13:15 · 50 s · 3 clicks`

All GNNs build on neural message passing, and one round has three phases. [click 1] Message: every node sends a message to each of its neighbors, built by a learnable function from the two node states and the edge features. [click 2] Aggregation: take node three. It gathers the messages of nodes one, two and four into a single vector, with a sum, a mean or a max, and the result cannot depend on the order of arrival. [click 3] Update: node three combines its old state with the aggregate through a learnable MLP and an activation. It learns how to mix, not just copy. Stacking layers simply repeats this round.

### 24 · GNN prediction tasks
`13:15 – 14:05 · 50 s · 3 clicks`

One embedding, three kinds of task, from a single node to a whole graph. [click 1] Node: what is this entity? Messages gather around a node, a softmax head turns its embedding into class probabilities, and every node gets a class, such as a paper's topic or a fraudulent user. [click 2] Edge: what is the relationship between these entities? A scoring function takes two node embeddings and returns the probability of a link, as in friend suggestions. [click 3] Graph: what property does the whole structure have? A readout pools all the node embeddings into one vector, and a head predicts a property, like the toxicity of a molecule.

### 25 · GNN architectures
`14:05 – 14:47 · 42 s · 3 clicks`

Message passing tells us what a GNN does. The architectures differ in how they weight the neighbors. [click 1] The graph convolutional network, GCN, gives every neighbor the same weight, fixed by the node degrees: simple and efficient. [click 2] GraphSAGE samples a fixed number of neighbors, here three out of six, and aggregates only those. It scales to large graphs and handles nodes it has never seen. [click 3] The graph attention network, GAT, learns a weight for each neighbor, so the most relevant ones count more. Let us look at each of them.

### 26 · Graph Convolutional Networks (GCN)
`14:47 – 15:32 · 45 s · 4 clicks`

The graph convolutional network updates every node with fixed weights: the graph decides how much each neighbor counts, and only one matrix is learned. [click 1] First, a self-loop: every node is also its own neighbor, so it keeps its own features. [click 2] Second, normalize: each connection is weighted by one over the square root of the two degrees. Well-connected nodes are damped, so hubs cannot dominate. Here the green neighbor, with few links, weighs the most. [click 3] Third, aggregate: sum the features, each times its weight. [click 4] Finally, a learned matrix and a nonlinearity give the new representation.

### 27 · GraphSAGE
`15:32 – 16:13 · 41 s · 4 clicks`

GraphSAGE learns how to build an embedding from a neighborhood, not one embedding per node. [click 1] Sample: pick a fixed number of neighbors at random, here three out of six, so the cost per node stays constant. [click 2] Aggregate: merge their vectors into one, with a mean, a max-pooling or an LSTM. [click 3] Concatenate: attach the result to the node's own vector. [click 4] Transform: a learned matrix and a nonlinearity give the new embedding. These are shared by all nodes, so unseen nodes can be embedded too: GraphSAGE is inductive.

### 28 · Graph Attention Network (GAT)
`16:13 – 16:49 · 36 s · 3 clicks`

GAT lets the network decide which neighbors matter. [click 1] Score: a small learned function compares node i with each neighbor, and returns a score. [click 2] Normalize: a softmax turns the scores into attention weights that sum to one. The most relevant neighbor gets the largest share, and the thickest edge. [click 3] Weighted sum: neighbor vectors are scaled by their weights and summed, so the result leans toward the most relevant neighbor. Several attention heads can run in parallel.

### 29 · Limitation: Over-squashing
`16:49 – 17:27 · 38 s · 6 clicks`

Now the limitations. First, over-squashing. [click 1] Reaching distant nodes needs more layers, and the receptive field grows exponentially: one, two, four, eight. [click 2] Distant parts must funnel through narrow bridges: a topological bottleneck. [click 3] And everything is compressed into one fixed-size vector per node, erasing detail. [click 4] The neighborhood grows like b to the K, while memory per node stays fixed, so distant messages are lost. [click 5] The trade-off: depth buys reach, but triggers over-squashing and over-smoothing. [click 6] An exponential neighborhood, squeezed into a fixed vector.

### 30 · Limitation: Over-smoothing
`17:27 – 18:07 · 40 s · 6 clicks`

Over-smoothing is the other side of the same dilemma. [click 1] A shallow network keeps each node's unique features, but lacks long-range context. [click 2] With two to four layers, repeated averaging acts as a low-pass filter. [click 3] In a deep stack, all embeddings become indistinguishable. [click 4] Every layer averages a node with its neighbors, so stacking layers keeps damping the differences. [click 5] A short-blanket dilemma: too few layers miss distant nodes, too many erase node identity. [click 6] In the limit, the Dirichlet energy goes to zero: all nodes are the same.

### 31 · Future directions
`18:07 – 18:51 · 44 s · 4 clicks`

Three directions look promising. [click 1] Graph foundation models: one model pre-trained on social, biological and financial graphs, then applied zero-shot to a graph it has never seen. [click 2] Topological deep learning: where ordinary edges need four hops, a hyperedge links the whole group at once. It is an information highway that eases over-squashing. [click 3] Streaming continual learning: when a new node arrives, only the weights around it are updated, in real time, without forgetting. [click 4] Together: universal transfer, topological expressivity and dynamic efficiency. That last point, real-time and efficient, is where the two worlds begin to meet.

---

# PART 4 — Perspectives and references

### 32 · Two different perspectives on neural networks
`18:51 – 19:24 · 33 s · 3 clicks`

Let us step back. [click 1] Spiking Neural Networks ask when, and how, information evolves over time. Neurons communicate through discrete spikes and integrate inputs, so information lives in spike timing and temporal patterns. [click 2] SNNs give us the temporal dimension, GNNs the relational one. [click 3] Graph Neural Networks ask how information is organized through relationships: nodes and edges, features and neighborhoods, messages aggregated to update representations. Information is organized by connectivity.

### 33 · References — Spiking Neural Networks
`19:24 – 19:36 · 12 s · 3 clicks`

> Click through the three reference cards while you speak; do not read them.

[click 1] [click 2] [click 3] For the spiking part: Maass on the third generation of neural networks, Gerstner and colleagues on neuron models, and Neftci and colleagues on surrogate gradients.

### 34 · References — Graph Neural Networks
`19:36 – 20:00 · 24 s · 2 clicks`

> Click through the two reference cards. Hold this last slide on screen while you take questions; its time budget includes a 10 s pause.

[click 1] [click 2] For the graph part: the Distill introduction to graph neural networks, and a recent survey in IEEE Access. Thank you for your attention. We are happy to take your questions.

---

## If you run long — cut in this order (about 75 seconds)

1. Slide 12 (STDP): keep only "A fires first, the weight goes up" — saves ~15 s
2. Slide 15 (keyword spotting): drop the accumulation example — saves ~15 s
3. Slide 22 (how GNNs see a graph): skip the degree matrix — saves ~10 s
4. Slide 30 (over-smoothing): skip the last click, the Dirichlet energy — saves ~10 s
5. Slide 31 (future): keep only the first and third items — saves ~15 s
6. Slides 33–34 (references): just say thank you — saves ~10 s

## If you run short

Add one sentence of your own to slide 13 (drone), slide 23 (message-passing animation) and slide 32 (two perspectives). Those three have the most visual material to hang a pause on.

## Rehearsal tips

- Do one full pass with `pmaker.html` → **Present** and a stopwatch, and compare your clock with the table above at each checkpoint.
- If you are more than 20 s off at a checkpoint, correct on the *next* slide rather than by speeding up.
- The slides with seven fast clicks (BPTT, RTRL, ANN-to-SNN) should be the quickest to deliver: one short clause per click.

---

## Spare text — no slide in the deck at the moment

The conclusion slide was removed from the deck. Its script is kept here in case it comes back; it is not part of the 20 minutes.

### (spare) Conclusion — Beyond Conventional Deep Learning
6 clicks

[click 1] To summarize. Spiking Neural Networks give us temporal information processing, spike-based communication and bio-inspired dynamics. [click 2] Graph Neural Networks give us learning from non-Euclidean data, relational information processing, and message passing with aggregation. [click 3] Spiking Graph Neural Networks combine the two: temporal and relational processing together, spatiotemporal dependencies, and efficient graph computation.

[click 4] Three takeaways. SNNs extend neural computation with temporal dynamics. GNNs extend deep learning to relational and irregular data. And their integration opens new possibilities for spatiotemporal learning. [click 5] It is a move from learning when information occurs, to understanding how information is connected.

[click 6] That leaves us with an open question: can neural networks become more efficient and expressive by combining brain-inspired computation with structured representations? We think it is worth pursuing. Thank you.

> Hold the final question on screen for a few seconds before saying "Thank you". This slide includes a 10 s pause in its time budget.
