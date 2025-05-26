from Proposed_CDBO import Proposed_run
import EEO_Dynamic_Mode_Selection.run
import Beefly_pattern_based_resource_allocation.run
import Dcdd_MCTS.run
import Joint_resource_allocation.run
import random


def call_main(Channel, users):

    print("\nNo of Cellular Users: ", users)

    ENERGY, THROUGHPUT = [], []                             # Energy Efficiency

    n_cluster = 2                                           # Value used for user grouping

    # Proposed Method
    print("\n\n########### Proposed Method Running ###########")
    Proposed_run.call_model(Channel, users, n_cluster, ENERGY, THROUGHPUT)
    print(ENERGY, THROUGHPUT)

    print("\n\n########### Comparative Methods Running ###########")
    Beefly_pattern_based_resource_allocation.run.call_model(Channel, users, n_cluster, ENERGY, THROUGHPUT)
    print(ENERGY, THROUGHPUT)
    Dcdd_MCTS.run.call_model(Channel, users, n_cluster, ENERGY, THROUGHPUT)
    print(ENERGY, THROUGHPUT)
    Joint_resource_allocation.run.call_model(Channel, users, n_cluster, ENERGY, THROUGHPUT)
    print(ENERGY, THROUGHPUT)
    EEO_Dynamic_Mode_Selection.run.call_model(Channel, users, n_cluster, ENERGY, THROUGHPUT)

    print(ENERGY, THROUGHPUT)
    return ENERGY, THROUGHPUT
