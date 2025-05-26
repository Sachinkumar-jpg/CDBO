# CDBO
Caviar dung beetle optimization based power control and FCM based user grouping for D2D Communication under HetNet.
CDBO: Power Control and User Grouping in D2D Communication under HetNet
This project implements a hybrid optimization framework combining Caviar Dung Beetle Optimization (CDBO) for power control and Fuzzy C-Means (FCM) clustering for user grouping in Device-to-Device (D2D) communication within Heterogeneous Networks (HetNet). The approach aims to enhance spectrum efficiency, reduce interference, and optimize resource allocation in 5G networks.

📌 Features
CDBO-Based Power Control: Employs a bio-inspired optimization algorithm to manage transmission power among D2D users, ensuring optimal Signal-to-Interference-plus-Noise Ratio (SINR) levels.

FCM-Based User Grouping: Utilizes Fuzzy C-Means clustering to group users based on proximity and channel conditions, facilitating efficient resource sharing.

Interference Mitigation: Implements strategies to minimize cross-tier and co-channel interference between D2D and cellular users.

Resource Allocation: Optimizes the assignment of Resource Blocks (RBs) to D2D pairs, enhancing overall network throughput.

Simulation and Analysis: Includes simulation scripts and result graphs to evaluate the performance of the proposed methods.

🛠️ Technologies Used
Programming Language: Python

Optimization Algorithm: Caviar Dung Beetle Optimization (CDBO)

Clustering Technique: Fuzzy C-Means (FCM)

Simulation Tools: Custom Python scripts for network simulation and performance analysis
arXiv

📁 Project Structure
css
Copy
Edit
CDBO/
├── Beefly_pattern_based_resource_allocation/
├── Dcdd_MCTS/
├── EEO_Dynamic_Mode_Selection/
├── Joint_resource_allocation/
├── Main/
├── Proposed_CDBO/
├── Result_graphs/
├── README.md
└── requirements.txt
Proposed_CDBO/: Contains the main implementation of the CDBO algorithm for power control.

Result_graphs/: Includes graphs and plots illustrating the performance metrics obtained from simulations.

requirements.txt: Lists the Python dependencies required to run the project.

🚀 Getting Started
Prerequisites
Python 3.x

Required Python libraries (listed in requirements.txt)
arXiv

Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/Sachinkumar-jpg/CDBO.git
cd CDBO
Install the dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Navigate to the Proposed_CDBO directory and run the main script:

bash
Copy
Edit
cd Proposed_CDBO
python main.py
📊 Results
The Result_graphs directory contains visualizations demonstrating the effectiveness of the proposed methods in terms of:

SINR improvement

Throughput enhancement

Interference reduction

Energy efficiency
arXiv

These results validate the superiority of the CDBO and FCM-based approach over traditional optimization techniques in D2D communication scenarios.

📄 License
This project is licensed under the MIT License.

🤝 Acknowledgments
Special thanks to the contributors and researchers whose work laid the foundation for this project.
