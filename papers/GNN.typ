#set document(title: "Report: Graph Neural Networks", author: "Gemini")
#set page(paper: "a4", margin: (x: 2.5cm, y: 3cm))
#set text(font: "Linux Libertine", size: 11pt, lang: "it")
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.1")

// Formattazione titoli principali
#show heading.where(level: 1): it => block(width: 100%, below: 1.5em, above: 2em)[
  #set text(size: 16pt, weight: "bold", fill: rgb("#1a73e8"))
  #it
]

// Titolo del Documento
#align(center)[
  #block(text(size: 22pt, weight: "bold")[A Gentle Introduction to \ Graph Neural Networks])
  #v(1em)
  #text(size: 14pt, style: "italic")[Report di sintesi basato sull'articolo di Distill.pub]
  #v(2em)
]

#outline(title: "Indice", indent: auto)
#v(2em)
#pagebreak()

= Introduzione
Le reti neurali profonde sono state storicamente progettate per operare su dati dalla struttura regolare, come le immagini (interpretabili come griglie 2D) o il testo (interpretabile come sequenze 1D). Tuttavia, gran parte dei dati nel mondo reale possiede una struttura relazionale, complessa e irregolare. L'articolo *A Gentle Introduction to Graph Neural Networks* di Distill esplora come le *Graph Neural Networks (GNN)* permettano di applicare il deep learning a questi domini, elaborando informazioni rappresentate sotto forma di grafi.

Un *grafo* è una struttura dati che descrive le relazioni tra un insieme di entità. Nello specifico, è composto da:
- *Nodi (Vertices):* Le singole entità (es. utenti, atomi, parole).
- *Archi (Edges):* Le connessioni o relazioni tra i nodi (es. amicizie, legami chimici).
- *Contesto Globale (Master Node):* Informazioni macroscopiche che descrivono l'intero grafo o sistema.

= I Grafi nel Mondo Reale
L'articolo dimostra come l'astrazione del grafo sia sufficientemente flessibile da modellare quasi ogni tipo di dato.

== Dati Tradizionali visti come Grafi
Anche i dati strutturati classici possono essere modellati come grafi:
- *Immagini:* Rappresentabili come grafi con una struttura a griglia molto rigida, dove ogni pixel è un nodo connesso ai suoi 8 pixel adiacenti (le informazioni di colore RGB costituiscono l'embedding del nodo).
- *Testo:* Modellabile come un grafo lineare orientato, in cui ogni parola/token è un nodo connesso esclusivamente alla parola successiva.

== Dati Strutturati Intrinsecamente (Heterogeneous Data)
Le GNN mostrano il loro vero potenziale su dataset intrinsecamente eterogenei:
- *Molecole:* I nodi rappresentano gli atomi e gli archi rappresentano i legami covalenti. Strutture chimiche o 3D si traducono perfettamente in grafi.
- *Reti Sociali:* Le persone o istituzioni sono nodi, le loro interazioni sociali sono gli archi. In queste reti il numero di connessioni per nodo è estremamente variabile.
- *Reti di Citazioni:* I paper scientifici (nodi) sono collegati tra loro da citazioni (archi orientati).

= Tipologie di Task Predittivi sui Grafi
Le Graph Neural Networks sono costruite per risolvere tre livelli fondamentali di compiti predittivi:

+ *Graph-level (Livello di intero grafo):* Lo scopo è predire una proprietà globale del grafo. 
  - *Esempio:* Data la struttura a grafo di una molecola, predire se risulterà tossica per l'uomo o se si legherà a un determinato recettore.
+ *Node-level (Livello di nodo):* Lo scopo è predire un attributo o un ruolo di un nodo specifico.
  - *Esempio:* In una rete sociale, classificare se un determinato nodo (utente) è un bot o una persona reale, o prevedere il suo ruolo in un gruppo.
+ *Edge-level (Livello di arco):* Lo scopo è predire la presenza, la forza o la natura di una connessione tra due nodi.
  - *Esempio:* Nei sistemi di raccomandazione, predire se esisterà un arco (interazione/acquisto) tra il nodo "Utente" e il nodo "Prodotto".

= Le Sfide: Perché non usare le reti neurali standard?
Utilizzare le Convolutional Neural Networks (CNN) o le Recurrent Neural Networks (RNN) su grafi complessi risulta inefficace per due motivi principali:
1. *Dimensioni e topologia variabili:* I grafi non hanno un numero fisso di nodi né una struttura omogenea.
2. *Assenza di un ordinamento spaziale (Permutation Invariance):* Non esiste un ordine univoco per elencare i nodi in una matrice. Una GNN deve garantire che, cambiando l'ordine dei nodi nella matrice di adiacenza in input, il risultato finale della rete rimanga identico.

= Architettura di una GNN e Message Passing
Il nucleo computazionale delle GNN moderne si basa sul concetto di *Message Passing* (scambio di messaggi). Nodi, archi e contesto globale mantengono ciascuno un proprio "stato" sotto forma di un vettore denso (embedding). Ad ogni strato (layer) della rete, queste rappresentazioni vengono aggiornate in due fasi:

== 1. Fase di Aggregazione (Pooling/Gather)
Un nodo raccoglie (aggrega) gli embedding dei nodi a esso direttamente collegati. Siccome il numero di vicini varia per ogni nodo e non c'è un ordine preciso tra loro, la funzione di aggregazione deve essere invariante rispetto all'ordine. Le operazioni più comuni sono:
- *Sum* (Somma di tutti i vettori vicini)
- *Mean* (Media aritmetica)
- *Max* (Valore massimo per componente)

== 2. Fase di Aggiornamento (Update)
Dopo aver raccolto le informazioni dal suo vicinato locale, il nodo unisce queste informazioni aggregate al suo stato attuale e le elabora passando attraverso una rete neurale (generalmente un Multi-Layer Perceptron, MLP) per generare il suo nuovo embedding.
Ripetendo questo processo per $K$ layer sequenziali, ogni nodo sarà in grado di integrare informazioni provenienti da una distanza massima di $K$ "salti" (hops) nel grafo.

= Gestione del Contesto e del Master Node
Una criticità comune nei grafi di grandi dimensioni è che i nodi situati agli estremi opposti necessitano di moltissimi layer di message passing prima che le loro informazioni si incontrino. 

Per mitigare questo problema, i ricercatori introducono spesso un *Contesto Globale (Master Node o U-node)*. Questo è un nodo virtuale connesso simultaneamente a tutti i nodi e a tutti gli archi del grafo. Durante le fasi di message passing, il Master Node agisce da "memoria centrale", raccogliendo informazioni da tutto il sistema e ritrasmettendole istantaneamente a ogni elemento, facilitando enormemente lo scambio di informazioni a lunga distanza e potenziando la capacità espressiva globale del modello.

= Conclusioni
Le GNN costituiscono uno strumento duttile ed espressivo, in grado di generalizzare i concetti fondamentali del deep learning classico portandoli su domini strutturati non euclidei. Grazie al bilanciamento accurato delle operazioni di aggregazione e aggiornamento multivariato, le GNN sono la spina dorsale di innovazioni contemporanee nella scoperta di farmaci, nelle simulazioni di fisica avanzata e nei sistemi di raccomandazione su larga scala.