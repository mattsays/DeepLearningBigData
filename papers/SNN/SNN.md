# Spiking Neural Networks: come funzionano, come si addestrano e a cosa servono

Questa guida introduce le **Spiking Neural Networks (SNN)** e segue i cinque punti della presentazione:

1. General Idea
2. Training / Inference Procedures
3. Current Applications
4. Current Limitations
5. Future Improvements

Le fonti includono lavori pubblicati fino al 2026. Le applicazioni sperimentali sono distinte dalle direzioni di ricerca.

## 1. General Idea

### 1.1 L'intuizione fondamentale

Una SNN è una rete neurale in cui i neuroni comunicano attraverso **eventi discreti nel tempo**, chiamati *spike*. Ogni neurone mantiene uno **stato interno** che cambia quando riceve input e continua a evolvere fra un input e il successivo.

Per capire il comportamento della rete bisogna quindi considerare sia **quali neuroni si attivano**, sia **quando si attivano**.

In una rete convenzionale, un neurone viene spesso descritto così:

$$
y = f\left(\sum_j w_j x_j + b\right)
$$

Gli input vengono pesati, sommati e trasformati da una funzione di attivazione. Nelle SNN, invece, gli input modificano una variabile che conserva traccia del passato: il **potenziale di membrana**.

Questo introduce una memoria temporale anche quando i collegamenti fra i livelli sono soltanto in avanti. Le reti convenzionali possono a loro volta elaborare sequenze, per esempio attraverso ricorrenza: la memoria non è un'esclusiva delle SNN.

### 1.2 Il neurone Leaky Integrate-and-Fire

Uno dei modelli più usati è il **Leaky Integrate-and-Fire**, abbreviato **LIF**. Il nome descrive tre operazioni:

- **Integrate:** il neurone accumula l'effetto degli input ricevuti.
- **Leaky:** in assenza di nuovi input, il potenziale tende al valore di riposo.
- **Fire:** quando il potenziale raggiunge una soglia, il neurone emette uno spike e il potenziale viene ridotto o riportato a un valore di reset.

Un'analogia è un recipiente con una piccola perdita. Ogni input aggiunge acqua; la perdita la fa uscire; raggiungere un certo livello provoca un'attivazione e uno svuotamento.

Il LIF è una semplificazione del neurone biologico: rappresenta gli istanti degli spike senza ricostruire la forma completa dell'impulso elettrico. [1]

### 1.3 Un modello matematico semplificato

Consideriamo una versione a passi temporali discreti, con potenziale di riposo e reset pari a zero.

Prima si calcola il potenziale prima del reset:

$$
v_i[t] = \beta u_i[t-1] + \sum_j w_{ij}s_j[t]
$$

Poi si decide se emettere uno spike:

$$
s_i[t] =
\begin{cases}
1 & \text{se } v_i[t] \geq \theta_i \\
0 & \text{altrimenti}
\end{cases}
$$

Infine si applica un **hard reset**:

$$
u_i[t] =
\begin{cases}
0 & \text{se } s_i[t] = 1 \\
v_i[t] & \text{altrimenti}
\end{cases}
$$

| Simbolo | Significato |
| --- | --- |
| $u_i[t]$ | Potenziale del neurone dopo l'eventuale reset |
| $v_i[t]$ | Potenziale prima dell'eventuale reset |
| $s_j[t]$ | Presenza o assenza di uno spike in ingresso |
| $w_{ij}$ | Peso del collegamento dal neurone $j$ al neurone $i$ |
| $\beta$ | Fattore di mantenimento del potenziale, con $0 < \beta < 1$ |
| $\theta_i$ | Soglia di attivazione |

Un valore elevato di $\beta$ mantiene più a lungo l'effetto degli input; un valore più basso lo fa decadere più rapidamente. Nel modello esponenziale, $\beta = e^{-\Delta t/\tau_m}$, dove $\tau_m$ è la costante di tempo della membrana.

Queste equazioni descrivono una scelta semplice: altri modelli includono correnti sinaptiche, ritardi, adattamento o un diverso reset. [1][2]

### 1.4 Esempio numerico

Supponiamo di avere:

- potenziale iniziale $u[0] = 0$;
- soglia $\theta = 1$;
- mantenimento $\beta = 0{,}8$;
- un input di ampiezza $0{,}4$ a ogni passo.

| Passo | Potenziale prima del reset | Spike emesso? | Potenziale dopo il reset |
| --- | --- | --- | --- |
| 1 | $0{,}8 \cdot 0 + 0{,}4 = 0{,}4$ | No | $0{,}4$ |
| 2 | $0{,}8 \cdot 0{,}4 + 0{,}4 = 0{,}72$ | No | $0{,}72$ |
| 3 | $0{,}8 \cdot 0{,}72 + 0{,}4 = 0{,}976$ | No | $0{,}976$ |
| 4 | $0{,}8 \cdot 0{,}976 + 0{,}4 = 1{,}1808$ | Sì | $0$ |
| 5 | $0{,}8 \cdot 0 + 0{,}4 = 0{,}4$ | No | $0{,}4$ |

Il neurone si attiva al quarto passo perché gli input si sono accumulati abbastanza rapidamente.

Se fra gli input inseriamo intervalli sufficientemente lunghi, il potenziale decade e la soglia può non essere mai raggiunta. **Lo stesso numero di input può quindi produrre risultati diversi a seconda della loro distribuzione temporale.**

### 1.5 Dallo spike alla rete

Uno spike emesso da un neurone viene trasmesso ai neuroni collegati. Il suo effetto dipende dal peso sinaptico: un collegamento eccitatorio favorisce l'attivazione, uno inibitorio la ostacola.

Nelle SNN binarie più comuni, lo spike indica un evento presente o assente. Questo non rende necessariamente binari i pesi o il potenziale di membrana.

L'informazione può essere rappresentata in modi diversi:

- **Rate coding:** attraverso il numero o la frequenza degli spike in una finestra temporale.
- **Temporal coding:** attraverso gli istanti degli spike, la loro relazione temporale o il tempo del primo spike.

Nelle reti addestrate direttamente, i livelli nascosti possono anche sviluppare una rappresentazione senza imporre esplicitamente un codice di questo tipo. [3]

## 2. Training / Inference Procedures

**Inference** significa usare il modello per produrre una risposta. **Training** significa modificare i suoi parametri affinché produca risposte migliori.

### 2.1 Inference: come si usa una SNN

Consideriamo un classificatore che riconosce una parola pronunciata.

1. **Preparazione dell'input.** Il segnale audio viene trasformato in una rappresentazione adatta al modello. Può diventare una sequenza di spike oppure alimentare il primo livello attraverso valori numerici, per esempio correnti d'ingresso.
2. **Elaborazione temporale.** A ogni passo, i neuroni integrano gli input, aggiornano il potenziale ed eventualmente emettono spike.
3. **Propagazione.** Gli spike influenzano i neuroni successivi attraverso i collegamenti appresi.
4. **Lettura dell'output.** Un meccanismo di decodifica trasforma l'attività finale in una classe o in un valore numerico.

La lettura può usare il conteggio degli spike, il tempo del primo spike o uno stato continuo di uscita. Per esempio, se ogni neurone finale rappresenta una parola, si può scegliere quello con più spike nella finestra osservata.

Durante l'inferenza ordinaria **i pesi rimangono fissi, mentre gli stati interni cambiano**. Sistemi con apprendimento online possono invece aggiornare anche i pesi durante l'uso. [2]

Il numero di passi temporali è una scelta importante: osservare più a lungo può dare più evidenza alla decisione, ma può aumentare latenza e attività.

### 2.2 Perché il training è difficile

La generazione di uno spike può essere scritta come una funzione a gradino:

$$
s[t] = H(v[t]-\theta)
$$

La sua derivata è zero quasi ovunque e non è definita alla soglia. La backpropagation standard non ottiene quindi un segnale utile per capire come piccoli cambiamenti dei pesi modifichino l'emissione degli spike.

Esiste anche una dipendenza temporale: un peso può influenzare il potenziale adesso e contribuire a uno spike diversi passi dopo. Bisogna attribuire l'errore sia ai collegamenti sia agli istanti precedenti. [3]

### 2.3 Metodo A: addestramento diretto con surrogate gradients

Un approccio molto usato mantiene gli spike discreti nel passaggio in avanti e usa una **derivata sostitutiva**, o *surrogate gradient*, nel passaggio all'indietro.

La procedura è:

1. Presentare una sequenza alla rete.
2. Calcolare gli stati e gli spike nel tempo.
3. Confrontare l'output con il target tramite una funzione di perdita.
4. Propagare il segnale di errore attraverso livelli e tempo.
5. Aggiornare i parametri con un ottimizzatore.

La propagazione temporale viene spesso effettuata con **Backpropagation Through Time (BPTT)**, come nelle reti ricorrenti. Il surrogate gradient sostituisce la derivata problematica della soglia: non coincide necessariamente con il gradiente esatto della rete discreta.

**Il neurone continua a emettere spike durante il forward pass; l'approssimazione riguarda il backward pass.** [3]

### 2.4 Metodo B: conversione ANN → SNN

Si addestra prima una rete convenzionale e poi se ne costruisce una versione spiking che ne approssima il comportamento.

Un esempio è rappresentare un'attivazione elevata con una frequenza elevata di spike. La conversione permette di riutilizzare procedure di training consolidate, ma può richiedere calibrazione e finestre temporali sufficientemente lunghe. [4]

### 2.5 Metodo C: regole locali e STDP

La **Spike-Timing-Dependent Plasticity (STDP)** modifica un collegamento in funzione della relazione temporale fra gli spike presinaptici e postsinaptici.

Nella forma classica, uno spike presinaptico poco prima di quello postsinaptico può rafforzare il collegamento; l'ordine inverso può indebolirlo. La regola usa informazioni locali, ma da sola non garantisce l'apprendimento di un obiettivo supervisionato complesso.

Esistono anche varianti modulate da ricompense o altri segnali. Non tutte le SNN sono quindi addestrate con STDP e non tutte apprendono durante l'inferenza. [4]

## 3. Current Applications

Le SNN sono particolarmente interessanti quando i dati evolvono nel tempo, gli eventi sono sparsi e il dispositivo deve rispondere con un budget energetico limitato.

### 3.1 Visione a eventi e robotica

Una camera a eventi registra cambiamenti locali di luminosità anziché produrre soltanto immagini complete a intervalli regolari. Questo offre un flusso temporale adatto all'elaborazione spiking.

Uno studio pubblicato su *Science Robotics* nel 2024 ha dimostrato una pipeline con camera a eventi e SNN su Intel Loihi per il controllo di un drone reale, includendo hovering, atterraggio e manovre laterali. È una dimostrazione sperimentale di percezione e controllo neuromorfici. [5]

### 3.2 Audio e keyword spotting

Riconoscere poche parole in un flusso audio continuo è un caso d'uso interessante per dispositivi che rimangono sempre in ascolto.

Un lavoro pubblicato online nel 2025 e incluso in un volume IEEE del 2026 propone un sistema di *streaming keyword spotting* con neuroni **resonate-and-fire**, quantizzazione e arresto anticipato della decisione. Si tratta di un esempio di ricerca su modelli compatti per dispositivi edge. [6]

### 3.3 Elaborazione tattile

Il contatto con una superficie genera sequenze temporali che possono essere elaborate da una SNN. Un lavoro sulla lettura del Braille ha confrontato classificatori spiking su Loihi con modelli convenzionali, considerando anche accuratezza, consumo e ritardo. [7]

Questi esempi mostrano compiti concreti. Il risultato ottenuto in uno specifico sistema non implica automaticamente lo stesso vantaggio in ogni applicazione.

## 4. Current Limitations

### 4.1 Costo del training temporale

La BPTT può richiedere di conservare molti stati intermedi. Allungare la sequenza o aumentare la dimensione della rete può aumentare memoria e calcolo. I surrogate gradients rendono possibile l'ottimizzazione, ma restano approssimazioni. [3]

### 4.2 Compromesso fra accuratezza, latenza e attività

Una finestra più lunga può migliorare una stima basata sul conteggio degli spike, ma ritarda la risposta. Ridurre troppo gli spike può invece rendere più difficile trasmettere l'informazione necessaria.

L'obiettivo dipende dal compito: bisogna trovare un equilibrio utile fra qualità della risposta, tempo e risorse. [4]

### 4.3 Codifica di dati non basati su eventi

Per dati statici, come una fotografia, convertire l'informazione in una sequenza temporale introduce una scelta progettuale e può aggiungere lavoro. L'input diretto numerico è un'alternativa, ma il vantaggio complessivo va comunque valutato. [2][4]

### 4.4 Dipendenza dall'hardware

Una SNN può essere simulata su CPU o GPU. Per sfruttare gli eventi sparsi, tuttavia, serve un'esecuzione capace di evitare lavoro inutile.

L'hardware neuromorfico cerca di supportare questo comportamento e di ridurre il movimento dei dati fra memoria e calcolo. Il vantaggio dipende comunque dal modello, dalla piattaforma e dal carico di lavoro. Una rete simulata con operazioni dense può non sfruttare la sparsità degli spike. [8]

### 4.5 Confronti energetici e benchmark

Dire che una SNN consuma meno richiede un confronto misurabile. Bisogna chiarire:

- se l'accuratezza e la latenza sono comparabili;
- se si misura il solo chip oppure il sistema completo;
- se sono inclusi codifica, comunicazione e consumi di base;
- se il riferimento convenzionale è ottimizzato per quel dispositivo.

**Un basso numero di spike non equivale da solo a una misura di energia.** NeuroBench propone un framework per valutare algoritmi e sistemi neuromorfici in modo più sistematico. [9]

## 5. Future Improvements

### 5.1 Training con meno memoria

Una direzione è ridurre la dipendenza dalla conservazione dell'intera storia temporale, usando aggiornamenti online, segnali locali o approssimazioni del gradiente. La sfida è mantenere una buona qualità dell'apprendimento. [3]

### 5.2 Apprendimento sul dispositivo

Aggiornare il modello direttamente sul dispositivo può permettere l'adattamento a nuovi utenti o ambienti. Sono però necessari hardware e algoritmi compatibili con i vincoli di memoria ed energia.

Uno studio di marzo 2026 ha presentato un'architettura multicore per il training diretto delle SNN, dimostrata su FPGA anche in configurazioni distribuite. È un risultato di ricerca verso questa direzione, non una capacità già presente in tutti i processori neuromorfici. [10]

### 5.3 Meno spike e risposte più rapide

Sfruttare meglio il timing degli spike può ridurre la necessità di lunghe finestre di conteggio. Anche interrompere l'elaborazione quando la decisione è sufficientemente affidabile può evitare lavoro aggiuntivo. Queste strategie devono essere valutate insieme all'accuratezza. [3][6]

### 5.4 Progettazione congiunta di algoritmo e hardware

Rete, precisione dei pesi, memoria e comunicazione possono essere progettate insieme. Un modello accurato in simulazione deve anche adattarsi alle risorse della piattaforma su cui verrà eseguito. [8][10]

### 5.5 Valutazioni più riproducibili

Benchmark comuni e misure dell'intero sistema possono chiarire dove le SNN offrono vantaggi pratici e dove rimangono margini di miglioramento. Questo comprende accuratezza, latenza, memoria ed energia, con condizioni di confronto esplicite. [9]

## Concetti da ricordare per la presentazione

- Una SNN elabora eventi attraverso neuroni che mantengono uno stato nel tempo.
- Il neurone LIF integra gli input, perde gradualmente potenziale ed emette uno spike alla soglia.
- Il timing degli input può cambiare la risposta anche a parità di numero di spike.
- Il training può usare surrogate gradients, conversione da ANN o regole locali.
- L'efficienza energetica dipende dall'intero sistema e deve essere misurata.

## Fonti

1. **Gerstner et al. — Neuronal Dynamics, sezione 1.3: Integrate-and-Fire Models.** Modello LIF, integrazione, decadimento e soglia. <https://neuronaldynamics.epfl.ch/online/Ch1.S3.html>
2. **Eshraghian — Tutorial 5: Training Spiking Neural Networks with snnTorch.** Stato ricorrente, input, training e lettura dell'output. <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html>
3. **Neftci, Mostafa e Zenke (2019) — Surrogate Gradient Learning in Spiking Neural Networks.** Problema della derivata, codifica temporale e metodi di apprendimento. <https://arxiv.org/html/1901.09948v2>
4. **Eshraghian et al. (2023) — Training Spiking Neural Networks Using Lessons From Deep Learning.** Panoramica su modelli, codifica, conversione e training. Il preprint iniziale è del 2021; la versione citata è aggiornata al 2023. <https://arxiv.org/abs/2109.12894>
5. **Paredes-Vallés et al. (2024) — Fully neuromorphic vision and control for autonomous drone flight.** Science Robotics. <https://pubmed.ncbi.nlm.nih.gov/38748781/>
6. **Lightweight Streaming Keyword Spotting System Based on Resonate-and-Fire Neurons.** IEEE Transactions on Cognitive and Developmental Systems; pubblicazione online 2025, volume 2026. <https://ieeexplore.ieee.org/document/11120352/>
7. **Braille Letter Reading: A Benchmark for Spatio-Temporal Pattern Recognition on Neuromorphic Hardware (2022).** <https://arxiv.org/abs/2205.15864>
8. **Davies et al. (2021) — Advancing Neuromorphic Computing With Loihi: A Survey of Results and Outlook.** Proceedings of the IEEE. <https://doi.org/10.1109/JPROC.2021.3067593>
9. **The NeuroBench Framework for Benchmarking Neuromorphic Computing Algorithms and Systems (2025).** Nature Communications. <https://www.nature.com/articles/s41467-025-56739-4>
10. **Li et al. (2026) — A Highly Energy-Efficient Multi-Core Neuromorphic Architecture for Training Deep Spiking Neural Networks.** Nature Communications. <https://pubmed.ncbi.nlm.nih.gov/41881988/>
