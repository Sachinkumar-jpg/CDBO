from Proposed_CDBO import D2D_Hetnet


def call_model(Channel, users, n_cluster, ENERGY, THROUGHPUT):
    print("\n CDBO is running")
    # Device to Device Communication under Hetnet model
    D2D_Hetnet.call_model(Channel, users, n_cluster, ENERGY, THROUGHPUT)

    return ENERGY, THROUGHPUT

