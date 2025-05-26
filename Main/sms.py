import numpy as np


def Distance(x, y):
    distpow = np.power(x, 2) + np.power(y, 2)
    Dist = np.sqrt(distpow)
    return Dist


def resource_allocate(users, x, y, total_power):
    reuse_distance_threshold = 50  # Example: reuse resources within 50 meters

    class D2DUser:
        def __init__(self, user_id, position):
            self.user_id = user_id
            self.position = position
            self.transmit_power = 0  # Placeholder for transmit power

    users1 = [D2DUser(i, (np.array([x[i], y[i]]))) for i in range(len(users))]

    # Resource allocation for reuse mode
    # Sequential Max Search (SMS) algorithm
    power = []
    for user in users1:
        # Determine neighboring users within reuse distance threshold
        neighboring_users = [other_user for other_user in users1 if other_user != user
                             and Distance(user.position, other_user.position).any() < reuse_distance_threshold]

        # Calculate allocated power for reuse mode users
        if neighboring_users:
            allocated_power = total_power / (1 + len(neighboring_users))
        else:
            allocated_power = total_power  # If no neighboring users, allocate full power

        user.transmit_power = allocated_power
        power.append(user.transmit_power)

    return power

