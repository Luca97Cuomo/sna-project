from utils import *
import logging
import logging_configuration
from network_analysis_utils import NetworkAnalyzer
from lesson4 import *

NETWORK_PATH = "nets/net_1"
NETWORK_NUM_NODES = 10000
DEGREE_MEAN_THRESHOLD = 100
DEGREE_STD_THRESHOLD = 50
DIAMETER_THRESHOLD = 3
AVG_CLUSTERING_COEF_THRESHOLD = 0.20
logger = logging.getLogger(__name__)


def main():
    target_network = load_network(NETWORK_PATH)

    network_analyzer = NetworkAnalyzer(degree_mean_threshold=DEGREE_MEAN_THRESHOLD,
                                       degree_std_threshold=DEGREE_STD_THRESHOLD,
                                       network_diameter_threshold=DIAMETER_THRESHOLD,
                                       network_clustering_coefficient_threshold=AVG_CLUSTERING_COEF_THRESHOLD,
                                       target_network=target_network, network_generation_algorithms_with_kwargs=[
            (preferentialG, {"n": NETWORK_NUM_NODES, "p": 0.20})
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

- configurationG con una degree sequence che è una power law, in quanto questa rete avrebbe una distribuzione dei
gradi dei nodi che rispecchierebbe appunto una power law.
ALTA CLUSTERIZZAZIONE : NO
SMALL WORLD : NO
POWER LAW : SI

- preferentialG in quanto le reti create con questo modello avrebbero una distribuzione dei gradi dei nodi che rispecchierebbe una power law.
ALTA CLUSTERIZZAZIONE : NO
SMALL WORLD : NO
POWER LAW : SI

- Generalized Watts-Strogatz 
ALTA CLUSTERIZZAZIONE : SE I NODI SONO BEN DISPOSTI NELLO SPAZIO SI
SMALL WORLD : SI
POWER LAW : NO


- affiliationG in queste reti ci sono molti parametri ma se sono tunati bene si possono avere reti:
ALTA CLUSTERIZZAZIONE : SI
SMALL WORLD : SI
POWER LAW : SI

Per ora il modello candidato è Generalized Watts-Strogatz in quanto sembra rispettare tutti i requisiti.
potremmo già escludere i modelli che implicano power law come : configurationG con una degree sequence che è una power law (per POWER LAW),
preferentialG (PER POWER LAW), affiliationG (PER POWER LAW,ANCHE SE VANNO FATTI ESPERIMENTI), randomG (PER SMALL WORLD)
"""