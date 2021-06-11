from utils import *
import logging
import logging_configuration
from network_analysis_utils import NetworkAnalyzer
from lesson4 import *
import numpy as np
from centrality_measures import *

NETWORK_PATH = "nets/net_1"
NETWORK_NUM_NODES = 10000
DEGREE_MEAN_THRESHOLD = 15
DEGREE_STD_THRESHOLD = 10
DIAMETER_THRESHOLD = 2
AVG_CLUSTERING_COEF_THRESHOLD = 0.20
logger = logging.getLogger(__name__)


def main():
    # {'n': 10000, 'r': 5, 'k': 5, 'q': 2}
    target_network = load_network(NETWORK_PATH)

    """
    clusters = nx.algorithms.community.girvan_newman(target_network)
    clusters = list(sorted(cluster) for cluster in next(clusters))
    logger.info(f"The network net_1 has : {len(clusters)} that are : {clusters}")
    """

    network_analyzer = NetworkAnalyzer(degree_mean_threshold=DEGREE_MEAN_THRESHOLD,
                                       degree_std_threshold=DEGREE_STD_THRESHOLD,
                                       network_diameter_threshold=DIAMETER_THRESHOLD,
                                       network_clustering_coefficient_threshold=AVG_CLUSTERING_COEF_THRESHOLD,
                                       target_network=target_network, network_generation_algorithms_with_kwargs=[
            (affiliationG, {"n": NETWORK_NUM_NODES, "m": 3, "q": 0.40, "c": 1, "p": 0.005, "s": 5}),
            (affiliationG, {"n": NETWORK_NUM_NODES, "m": 5, "q": 0.20, "c": 1, "p": 0.005, "s": 5}),
            (affiliationG, {"n": NETWORK_NUM_NODES, "m": 7, "q": 0.20, "c": 1, "p": 0.005, "s": 5}),
            (affiliationG, {"n": NETWORK_NUM_NODES, "m": 10, "q": 0.20, "c": 1, "p": 0.005, "s": 5}),
            (affiliationG, {"n": NETWORK_NUM_NODES, "m": 10, "q": 0.20, "c": 3, "p": 0.005, "s": 5}),
            (affiliationG, {"n": NETWORK_NUM_NODES, "m": 15, "q": 0.20, "c": 3, "p": 0.005, "s": 5}),
            (affiliationG, {"n": NETWORK_NUM_NODES, "m": 20, "q": 0.20, "c": 5, "p": 0.005, "s": 5}),
        ])

    network_analyzer.analyze_all()


if __name__ == "__main__":
    main()

"""
La rete presenta le seguenti caratteristiche:
1) Numero di nodi 10000
2) Numero di archi 388147
3) La distribuzione dei gradi dei nodi non segue una power law, ma sembra seguire una gaussiana
con media 69.4 e varianza 27.3. 
4) il diametro della rete è 6 quindi è uno SMALL WORLD e il suo calcolo impiega circa 7 minuti
5) La rete è connessa.
6) il suo coefficiente di clustering è 0.5745321712547844

ALTA CLUSTERIZZAZIONE : SEMBRA DI SI.
SMALL WORLD : SI
POWER LAW : NO

- randomG in quanto questo tende a creare una rete con coefficente di clustering basso, con un diametro dell'
ordine di log(n) (quindi non SMALL WORLD) e di solito in questi grafi i nodi hanno quasi tutti lo stesso grado.
ALTA CLUSTERIZZAZIONE : NO
SMALL WORLD : NO (DIPENDE DALLA PROBABILITà CON P=0.005 IL DIAMETRO è 4, CON P=0.20 è 2)
POWER LAW : NO

del modello random abbiamo provato diversi valori p : 
Per p troppo piccli (5e-05, 0.0005): la media e la deviazione standard dei gradi tende ad essere molto molto bassa 
                                     (il grafo è anche non connesso == diametro infinito)
                                     e inoltre il coefficiente di clustering è bassissimo come ci si aspettava 0.0003.
Per p un pò più grandi (0.005) : la media e la deviazione standard dei gradi tende a valori accettabili intorno ai 50 la media
                                 e 15 la std, anche se ancora lontani dalle nostre (69 e 27), il diametro è piccolo 4 quindi la rete
                                 è uno Small world, ma il coefficiente di clusterizzazione è molto molto basso 0.005034732258503456
                                 due ordini di grandezza inferiore rispetto al nostro.
Per p più grandi ancora (0.05) : la media e la deviazione standard dei gradi esplodono, 500 di media e 43 di std, il diametro è 2
                                 questo implica che la rete è molto connessa, ma il coefficiente di clustering rimane un ordine di grandezza
                                 inferiore al nostro 0.04992027359136008.
Sulla base delle considerazioni teoriche e quelle sperimentali, è quindi possibile escludere questo modello.

- configurationG con una degree sequence che è una power law, in quanto questa rete avrebbe una distribuzione dei
gradi dei nodi che rispecchierebbe appunto una power law.

ALTA CLUSTERIZZAZIONE : NO
SMALL WORLD : NO
POWER LAW : SI
DOVREMMO GIA POTERLO ESCLUDERE



- preferentialG in quanto le reti create con questo modello avrebbero una distribuzione dei gradi dei nodi che rispecchierebbe una power law.
ALTA CLUSTERIZZAZIONE : NO
SMALL WORLD : NO
POWER LAW : SI

DOVREMMO GIA POTERLO ESCLUDERE
Sulla base delle considerazioni teoriche e sperimentali che possiamo escludere questo modello, in quanto le reti generate da esso
impongono una distribuzione dei gradi dei nodi che è una power_law, mentre la nostra rete non rispecchia questa proprietà. Inoltre
non restituisce in genere una rete che è uno Small world e con alto indice di clusterizzazione cose entrambe vere per la rete in esame.

- Generalized Watts-Strogatz 
ALTA CLUSTERIZZAZIONE : SE I NODI SONO BEN DISPOSTI NELLO SPAZIO SI
SMALL WORLD : SI
POWER LAW : NO
SECONDO ME è QUESTO MODELLO


- affiliationG in queste reti ci sono molti parametri ma se sono tunati bene si possono avere reti:
ALTA CLUSTERIZZAZIONE : SI
SMALL WORLD : SI
POWER LAW : SI

Per ora il modello candidato è Generalized Watts-Strogatz in quanto sembra rispettare tutti i requisiti.
potremmo già escludere i modelli che implicano power law come : configurationG con una degree sequence che è una power law (per POWER LAW),
preferentialG (PER POWER LAW), affiliationG (PER POWER LAW,ANCHE SE VANNO FATTI ESPERIMENTI), randomG (PER SMALL WORLD)
"""
